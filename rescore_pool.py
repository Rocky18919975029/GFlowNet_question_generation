# rescore_pool.py (Memory Optimized)

import os
import sys
import hydra
import torch
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# This script should be in the root directory
# We are NOT importing from train.py anymore to avoid loading the sampler_model
from reward import ContradictionReward

def get_scoring_models_and_tokenizer(config: DictConfig):
    """A leaner version that only loads the models needed for scoring."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name, add_bos_token=False, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("--- Loading frozen base model for answer generation ---")
    base_model = AutoModelForCausalLM.from_pretrained(config.model.name)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    print("--- Loading NLI model for contradiction scoring ---")
    nli_model = AutoModelForSequenceClassification.from_pretrained(config.reward.nli_model_name)
    nli_model.eval()
    for param in nli_model.parameters():
        param.requires_grad = False
        
    return base_model, nli_model, tokenizer

@hydra.main(version_base=None, config_path="configs", config_name="dataseed")
def rescore_pool(config: DictConfig):
    main_config = OmegaConf.merge(
        OmegaConf.load("configs/config.yaml"),
        {"model": OmegaConf.load(f"configs/model/{OmegaConf.load('configs/config.yaml').defaults[1].model}.yaml"),
         "reward": OmegaConf.load(f"configs/reward/{OmegaConf.load('configs/config.yaml').defaults[2].reward}.yaml")}
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_pool_path = "candidate_pool.jsonl"
    output_pool_path = "clean_pool.jsonl"

    print("--- 1. Setting up models with lean loading... ---")
    base_model, nli_model, tokenizer = get_scoring_models_and_tokenizer(main_config)
    
    # We will manually move models to/from the GPU.
    # So we don't pass the models to the Reward class constructor.
    reward_model = ContradictionReward(
        base_model=None, base_tokenizer=tokenizer, nli_model_name=main_config.reward.nli_model_name,
        likelihood_weight=0, answer_prompt_template=main_config.task.answer_prompt_template,
        termination_token_id=tokenizer.eos_token_id, min_len=0, 
        nli_batch_size=config.evaluation_batch_size, device=device
    )
    # Overwrite the NLI model in the reward class with our pre-loaded one
    reward_model.nli_model = nli_model

    df_pool = pd.read_json(input_pool_path, lines=True)
    if os.path.exists(output_pool_path):
        os.remove(output_pool_path)

    print("--- 3. Re-scoring all candidates with memory cycling... ---")
    with open(output_pool_path, 'a') as f_out:
        for group_key, group_df in tqdm(df_pool.groupby(['subject', 'edit_fact']), desc="Re-scoring Pool"):
            subject, edit_fact = group_key
            questions_in_group = group_df['question'].tolist()
            
            # --- STEP A: Generate Answers (base_model on GPU) ---
            base_model.to(device)
            reward_model.base_model = base_model # Temporarily attach to the reward model
            prompts_for_answers = [main_config.task.answer_prompt_template.format(question=q) for q in questions_in_group]
            answers = reward_model._generate_answers(prompts_for_answers)
            base_model.to('cpu') # Move base_model OFF the GPU
            torch.cuda.empty_cache() # Clear cache

            # --- STEP B: Score Contradiction (nli_model on GPU) ---
            nli_model.to(device)
            contradiction_scores = reward_model._calculate_contradiction_score(answers, edit_fact)
            nli_model.to('cpu') # Move nli_model OFF the GPU
            torch.cuda.empty_cache() # Clear cache
            
            results_df = group_df.copy()
            results_df['generated_answer'] = answers
            results_df['contradiction_score'] = contradiction_scores.detach().cpu().numpy()
            
            f_out.write(results_df.to_json(orient='records', lines=True))
            f_out.flush()

    print(f"\n--- Re-scoring complete. Clean, valid results saved to {output_pool_path}. ---")


if __name__ == "__main__":
    rescore_pool()