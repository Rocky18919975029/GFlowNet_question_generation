# evaluate.py

from types import MethodType
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf # Keep OmegaConf for potential debugging prints
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training

# Import our custom modules
from lightning_data import StatementPairDataModule
from lightning_module import ContradictionGFNTask
from reward import ContradictionReward
from gflownet.replay_buffer import ReplayBuffer
from utils import get_termination_vals, lora_to_base, base_to_lora 

torch.set_float32_matmul_precision('high')

def get_model(config: DictConfig):
    """
    Initializes the tokenizer and the LoRA-wrapped language model.
    (This function is copied directly from train.py)
    """
    if config.model.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        add_bos_token=False,
        padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    if config.model.use_4bit:
        # Prepare model for k-bit training also sets up the correct modules for LoRA
        model = prepare_model_for_kbit_training(model)
    
    lora_config = hydra.utils.instantiate(config.model.lora_config)

    # Wrap the base model with PEFT adapters
    model = get_peft_model(model, lora_config)
    
    # Set dropout to 0 for evaluation consistency (usually done via model.eval() but explicit here)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    
    model.print_trainable_parameters()

    return model, tokenizer

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def evaluate(config: DictConfig):
    # DEBUG: Print the full loaded config to verify contents
    print(f"DEBUG: Hydra config loaded. Full config:\n{OmegaConf.to_yaml(config)}")

    # Set a global seed for reproducibility
    pl.seed_everything(config.seed, workers=True)

    # Access checkpoint_path from the 'evaluation' block
    checkpoint_path = config.evaluation.checkpoint_path
    
    if not torch.cuda.is_available():
        print("CUDA not available. Evaluation will run on CPU, which may be very slow for LLMs.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # 1. --- Initialize Model and Tokenizer (same as in train.py) ---
    # This will initialize the base model and apply PEFT, including BitsAndBytes quantization setup.
    model, tokenizer = get_model(config)
    # The model should already be on the correct device due to BitsAndBytes or prepare_model_for_kbit_training
    # but an explicit .to(device) for the outer PeftModel might sometimes be needed if issues arise.
    # For bnb models, the model is often already on GPU.
    # model.to(device) # Uncomment if you get device mismatch errors.

    # Define the termination token (end of question)
    try:
        end_of_question_token_id = tokenizer.encode("?", add_special_tokens=False)[-1]
    except:
        end_of_question_token_id = tokenizer.convert_tokens_to_ids("?")

    # 2. --- Initialize Reward Components (same as in train.py) ---
    reward_model = ContradictionReward(
        base_model=model, # Pass the initialized model (which is now a PeftModel)
        base_tokenizer=tokenizer,
        nli_model_name=config.reward.nli_model_name,
        task_prompt=config.task.task_prompt,
        termination_token_id=end_of_question_token_id,
        min_len=config.task.min_question_len,
        temperature=config.task.reward_temp_end, # Use final temperature for evaluation
        nli_batch_size=config.reward.nli_batch_size,
        device=device, 
    )

    reward_buffer = ReplayBuffer( # Required for ContradictionGFNTask's __init__, even if not actively used for eval
        buffer_size=config.task.replay_buffer_size,
        termination_token_id=end_of_question_token_id,
    )

    # 3. --- Initialize DataModule ---
    data_module = StatementPairDataModule(
        data_path=config.data.path,
        tokenizer=tokenizer,
        train_size=config.data.train_size, 
        limit_data=config.data.limit_data,
    )
    data_module.setup("fit") 

    val_probes = [data_module.val_data[i] for i in range(min(config.task.n_probes, len(data_module.val_data)))]
    
    # 4. --- Load the Main LightningModule from Checkpoint with Filtering ---
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Manually load the checkpoint
    # Ensure map_location is set correctly, typically to the device the model will run on
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    # Filter out bitsandbytes specific keys from the state_dict
    # These keys are automatically generated/managed by bnb during model initialization
    # and cause "Unexpected key(s)" errors when trying to load them.
    filtered_state_dict = {}
    for k, v in state_dict.items():
        # Check if the key corresponds to the model's base_model and contains bnb metadata
        if k.startswith("model.base_model.model.") and \
           ("absmax" in k or "quant_map" in k or "quant_state" in k):
            # print(f"DEBUG: Filtering bnb key: {k}") # Uncomment for verbose debugging
            continue # Skip these keys
        filtered_state_dict[k] = v

    # Instantiate ContradictionGFNTask with the newly created (quantized, PEFT-wrapped) model
    task = ContradictionGFNTask(
        model=model, # This 'model' is the one we created with get_model()
        tokenizer=tokenizer,
        reward_model=reward_model,
        reward_buffer=reward_buffer,
        task_prompt=config.task.task_prompt,
        n_samples=config.task.n_samples,
        lr=config.training.lr, # This value is not used for eval but required for __init__
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
        val_probes=val_probes,
        use_4bit=config.model.use_4bit,
    )
    
    # Load the filtered state_dict into the task.
    # Set strict=False just in case there are other minor mismatches not caught by filtering,
    # but ideally, the filtering should make strict=True possible for non-bnb keys.
    # For robustness, we'll keep it False.
    task.load_state_dict(filtered_state_dict, strict=False) 

    # Apply bitsandbytes workaround for .to() and .cuda() methods
    if config.model.use_4bit:
        task.to = MethodType(lambda s, *args, **kwargs: s, task)
        task.cuda = MethodType(lambda s, *args, **kwargs: s, task)

    # Set model to evaluation mode
    task.eval()
    
    # 5. --- Initialize the PyTorch Lightning Trainer for Validation ---
    logger = pl.loggers.WandbLogger(
        project=config.logger.project,
        name=f"evaluation_run_from_ckpt_{checkpoint_path.split('/')[-1]}",
        mode=config.logger.mode, 
        log_model=False
    ) if config.evaluation.log_eval_to_wandb else None 

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
    )

    # 6. --- Run Evaluation ---
    print("Running validation over the entire validation dataset...")
    trainer.validate(model=task, datamodule=data_module)

    # 7. --- Generate and Log Qualitative Samples (Optional) ---
    print("\nGenerating qualitative samples (probes):")
    if task.hparams.val_probes:
        samples_df = task.sample_probes(task.hparams.val_probes, n_samples=config.task.n_samples) 
        print(samples_df.to_string())
        if logger:
            logger.log_table(key="samples/evaluation_probes", dataframe=samples_df)
    else:
        print("No validation probes configured.")


if __name__ == "__main__":
    evaluate()