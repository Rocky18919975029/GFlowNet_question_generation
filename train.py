# train.py

from types import MethodType
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training
import os

from lightning_data_scalable import ScalableDataModule
from lightning_module import ContradictionGFNTask
from reward import ContradictionReward
from gflownet.replay_buffer_scalable import RedisReplayBuffer

torch.set_float32_matmul_precision('high')

def get_separated_models_and_tokenizer(config: DictConfig):
    bnb_config = None
    if config.model.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True
        )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name, add_bos_token=False, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("--- Loading frozen base model for reward calculation ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name, trust_remote_code=True
    )
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    print("--- Loading sampler model for GFlowNet fine-tuning ---")
    sampler_model = AutoModelForCausalLM.from_pretrained(
        config.model.name, quantization_config=bnb_config, trust_remote_code=True
    )

    sampler_model.gradient_checkpointing_enable()
    
    if config.model.use_4bit:
        sampler_model = prepare_model_for_kbit_training(sampler_model)
    lora_config = hydra.utils.instantiate(config.model.lora_config)
    sampler_model = get_peft_model(sampler_model, lora_config)
    sampler_model.print_trainable_parameters()
    return base_model, sampler_model, tokenizer

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def train(config: DictConfig):
    pl.seed_everything(config.seed, workers=True)
    base_model, sampler_model, tokenizer = get_separated_models_and_tokenizer(config)
    try:
        end_of_question_token_id = tokenizer.encode("?", add_special_tokens=False)[-1]
    except:
        end_of_question_token_id = tokenizer.convert_tokens_to_ids("?")
    
    # --- CHANGE: Pass new config parameters to ContradictionReward ---
    reward_model = ContradictionReward(
        base_model=base_model,
        base_tokenizer=tokenizer,
        nli_model_name=config.reward.nli_model_name,
        likelihood_weight=config.task.likelihood_weight,
        answer_prompt_template=config.task.answer_prompt_template,
        termination_token_id=end_of_question_token_id,
        min_len=config.task.min_question_len,
        temperature=config.task.reward_temp_start,
        nli_batch_size=config.reward.nli_batch_size,
        use_hybrid_reward=config.task.use_hybrid_reward,
        penalized_reward_end_step=config.task.penalized_reward_end_step,
        contradiction_threshold=config.task.contradiction_threshold,
        failure_penalty=config.task.failure_penalty,
    )
    
    reward_buffer = RedisReplayBuffer(
        host=config.buffer.redis_host,
        port=config.buffer.redis_port,
        buffer_size=config.task.replay_buffer_size,
        termination_token_id=end_of_question_token_id
    )
    
    if int(os.environ.get("RANK", 0)) == 0:
        print("--- Rank 0 process is resetting the Redis buffer. ---", flush=True)
        reward_buffer.reset()
        
    data_module = ScalableDataModule(
        data_path=config.data.path,
        tokenizer=tokenizer,
        train_size=config.data.train_size,
        limit_data=config.data.limit_data,
        batch_size=1,
        num_workers=2
    )
    
    # --- CHANGE: Pass new config parameters to ContradictionGFNTask ---
    task = ContradictionGFNTask(
        model=sampler_model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        reward_buffer=reward_buffer,
        task_prompt=config.task.task_prompt,
        likelihood_weight=config.task.likelihood_weight,
        n_samples=config.task.n_samples,
        lr=config.training.lr,
        subtb_lambda=config.task.subtb_lambda,
        pf_temp_high=config.task.pf_temp_high,
        pf_temp_low=config.task.pf_temp_low,
        pf_temp_prob=config.task.pf_temp_prob,
        use_buffer_prob=config.task.use_buffer_prob,
        min_question_len=config.task.min_question_len,
        max_question_len=config.task.max_question_len,
        reward_temp_start=config.task.reward_temp_start,
        reward_temp_end=config.task.reward_temp_end,
        reward_temp_horizon=config.task.reward_temp_horizon,
        n_probes=config.task.n_probes,
        use_4bit=config.model.use_4bit,
        checkpoint_save_interval=config.task.checkpoint_save_interval,
        log_every_n_steps=config.training.log_every_n_steps,
        use_hybrid_reward=config.task.use_hybrid_reward,
        penalized_reward_end_step=config.task.penalized_reward_end_step,
        contradiction_threshold=config.task.contradiction_threshold,
        failure_penalty=config.task.failure_penalty,
    )
    
    logger = hydra.utils.instantiate(config.logger)
    
    strategy = config.trainer.strategy
    if strategy == "ddp":
        from lightning_fabric.plugins.environments import LightningEnvironment
        strategy = pl.strategies.DDPStrategy(
            cluster_environment=LightningEnvironment(),
            find_unused_parameters=True
        )

    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        num_nodes=config.trainer.num_nodes,
        strategy=strategy,
        max_epochs=config.trainer.max_epochs,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        logger=logger,
        callbacks=[hydra.utils.instantiate(c) for c in config.callbacks],
        log_every_n_steps=config.training.log_every_n_steps,
    )
    if config.model.use_4bit:
        task.to = MethodType(lambda s, *args, **kwargs: s, task)
        task.cuda = MethodType(lambda s, *args, **kwargs: s, task)
    
    trainer.fit(model=task, datamodule=data_module)

if __name__ == "__main__":
    train()