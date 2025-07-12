from types import MethodType
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training

from lightning_data import StatementPairDataModule
from lightning_module import ContradictionGFNTask
from reward import ContradictionReward
from gflownet.replay_buffer import ReplayBuffer

torch.set_float32_matmul_precision('high')

def get_model(config: DictConfig):
    if config.model.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    else: bnb_config = None
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name, add_bos_token=False, padding_side="left")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name, quantization_config=bnb_config, trust_remote_code=True)
    if config.model.use_4bit: model = prepare_model_for_kbit_training(model)
    lora_config = hydra.utils.instantiate(config.model.lora_config)
    model = get_peft_model(model, lora_config)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0.0
    model.print_trainable_parameters()
    return model, tokenizer

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def train(config: DictConfig):
    pl.seed_everything(config.seed, workers=True)
    model, tokenizer = get_model(config)
    try:
        end_of_question_token_id = tokenizer.encode("?", add_special_tokens=False)[-1]
    except:
        end_of_question_token_id = tokenizer.convert_tokens_to_ids("?")

    reward_model = ContradictionReward(
        base_model=model, base_tokenizer=tokenizer,
        nli_model_name=config.reward.nli_model_name,
        likelihood_weight=config.task.likelihood_weight,
        answer_prompt_template=config.task.answer_prompt_template, # Pass the new template
        termination_token_id=end_of_question_token_id,
        min_len=config.task.min_question_len,
        temperature=config.task.reward_temp_start,
        nli_batch_size=config.reward.nli_batch_size,
    )
    reward_buffer = ReplayBuffer(
        buffer_size=config.task.replay_buffer_size,
        termination_token_id=end_of_question_token_id)

    data_module = StatementPairDataModule(
        data_path=config.data.path, tokenizer=tokenizer,
        train_size=config.data.train_size, limit_data=config.data.limit_data)
    data_module.setup("fit") 
    val_probes = [data_module.val_data[i] for i in range(min(config.task.n_probes, len(data_module.val_data)))]
    
    task = ContradictionGFNTask(
        model=model, tokenizer=tokenizer, reward_model=reward_model, reward_buffer=reward_buffer,
        task_prompt=config.task.task_prompt, likelihood_weight=config.task.likelihood_weight,
        n_samples=config.task.n_samples, lr=config.training.lr, subtb_lambda=config.task.subtb_lambda,
        pf_temp_high=config.task.pf_temp_high, pf_temp_low=config.task.pf_temp_low,
        pf_temp_prob=config.task.pf_temp_prob, use_buffer_prob=config.task.use_buffer_prob,
        min_question_len=config.task.min_question_len, max_question_len=config.task.max_question_len,
        reward_temp_start=config.task.reward_temp_start, reward_temp_end=config.task.reward_temp_end,
        reward_temp_horizon=config.task.reward_temp_horizon, val_probes=val_probes, use_4bit=config.model.use_4bit)

    logger = hydra.utils.instantiate(config.logger)
    
    strategy = config.trainer.strategy
    if strategy == "ddp":
        strategy = pl.strategies.DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        num_nodes=config.trainer.num_nodes,
        strategy=strategy,
        max_epochs=config.training.epochs,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        logger=logger,
        callbacks=[hydra.utils.instantiate(c) for c in config.callbacks]
    )

    if config.model.use_4bit:
        task.to = MethodType(lambda s, *args, **kwargs: s, task)
        task.cuda = MethodType(lambda s, *args, **kwargs: s, task)

    trainer.fit(model=task, datamodule=data_module)

if __name__ == "__main__":
    train()