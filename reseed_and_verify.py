# reseed_and_verify.py

import os
import hydra
import torch
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
import redis # <--- THE MISSING IMPORT

from gflownet.replay_buffer_scalable import RedisReplayBuffer

def get_redis_key(prompt_text: str, tokenizer) -> str:
    """The key generation function we are debugging."""
    prompt_tokens = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
    prompt_hash = hash(str(prompt_tokens.cpu().tolist()))
    return f"gfn_buffer:{prompt_hash}"

@hydra.main(version_base=None, config_path="configs", config_name="dataseed")
def reseed_and_verify(config: DictConfig):
    main_config = OmegaConf.merge(
        OmegaConf.load("configs/config.yaml"),
        {"model": OmegaConf.load(f"configs/model/{OmegaConf.load('configs/config.yaml').defaults[1].model}.yaml")}
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pool_path = "clean_pool.jsonl"
    df_pool = pd.read_json(pool_path, lines=True)
    
    tokenizer = AutoTokenizer.from_pretrained(main_config.model.name, add_bos_token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    replay_buffer = RedisReplayBuffer(
        host=main_config.buffer.redis_host,
        port=main_config.buffer.redis_port,
        buffer_size=main_config.task.replay_buffer_size,
        termination_token_id=tokenizer.eos_token_id
    )
    print("--- Clearing Redis buffer completely before re-seeding... ---")
    replay_buffer.reset()
    
    task_prompt_template = main_config.task.task_prompt

    # --- Step 1: Reseed the buffer and STORE the keys we generate ---
    print("--- Step 1: Re-seeding Redis and recording generated keys... ---")
    generated_keys_during_seeding = {} # A dictionary to store {subject: key}
    total_seeded = 0

    for subject_text, group_df in tqdm(df_pool.groupby('subject'), desc="Re-seeding Buffer"):
        top_k_df = group_df.nlargest(config.num_top_to_select, 'contradiction_score')
        if top_k_df.empty:
            continue
            
        top_questions_text = top_k_df['question'].tolist()
        top_rewards = torch.tensor(top_k_df['contradiction_score'].values, device=device)
        
        prompt_text = task_prompt_template.format(subject=subject_text)
        prompt_tokens = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
        
        key_for_subject = get_redis_key(prompt_text, tokenizer)
        generated_keys_during_seeding[subject_text] = key_for_subject

        top_questions_toks = tokenizer(top_questions_text, return_tensors='pt', padding=True, truncation=True).input_ids
        
        replay_buffer.add_batch(
            prompt_tokens=prompt_tokens,
            generated_questions=top_questions_toks,
            logrewards=top_rewards,
            tokenizer=tokenizer
        )
        total_seeded += len(top_k_df)

    print(f"\n--- Re-seeding complete. Added {total_seeded} questions to Redis. ---")

    # --- Step 2: Immediately Verify the Keys ---
    print("\n--- Step 2: Immediately verifying the keys we just wrote... ---")
    r_str = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    all_redis_keys = set(r_str.keys("gfn_buffer:*"))
    
    successful_checks = 0
    failed_checks = 0
    
    for subject, generated_key in list(generated_keys_during_seeding.items())[:20]:
        print("\n" + "-"*50)
        print(f"Verifying subject: '{subject}'")
        print(f"  - Key that was written: {generated_key}")
        
        if generated_key in all_redis_keys:
            print("  - ✅ VERIFICATION PASSED: This key exists in Redis.")
            successful_checks += 1
        else:
            print("  - 🚨🚨🚨 VERIFICATION FAILED: This key DOES NOT EXIST in Redis. 🚨🚨🚨")
            failed_checks += 1
    
    print("\n" + "="*50)
    print("--- Verification Complete ---")
    print(f"Passed checks: {successful_checks}")
    print(f"Failed checks: {failed_checks}")
    if failed_checks > 0:
        print("CONCLUSION: The bug is confirmed. The keys being written are not the keys being stored.")
    else:
        print("CONCLUSION: Seeding and verification successful. The data is now ready.")


if __name__ == "__main__":
    reseed_and_verify()