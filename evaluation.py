# evalaution.py

import hydra
import torch
import pandas as pd
from omegaconf import DictConfig
from functools import partial

# Import your project's classes and functions
from train import get_separated_models_and_tokenizer
from reward import ContradictionReward
from gflownet.trajectory import generate_and_return_termination_logprob

@hydra.main(version_base=None, config_path="./configs", config_name="config")
def evaluate(config: DictConfig):
    print("--- Setting up models and tokenizer for evaluation ---")
    
    # Use your existing function to load and prepare all models
    base_model, sampler_model, tokenizer = get_separated_models_and_tokenizer(config)
    
    # --- IMPORTANT: Load your fine-tuned LoRA weights ---
    # You need to provide the path to your final checkpoint file.
    # It's best to pass this from the command line for flexibility.
    # Example: python evaluation.py +checkpoint_path=/path/to/your/last.ckpt
    if 'checkpoint_path' not in config:
        raise ValueError("Please provide the path to your checkpoint via command line, e.g., '+checkpoint_path=path/to/your.ckpt'")
    
    print(f"--- Loading checkpoint from: {config.checkpoint_path} ---")
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    sampler_model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    sampler_model.to("cuda")
    base_model.to("cuda")
    sampler_model.eval()
    base_model.eval()

    # --- Set up the Reward Model for scoring ---
    # We don't need the hybrid reward parameters for evaluation, just the core components.
    try:
        end_of_question_token_id = tokenizer.encode("?", add_special_tokens=False)[-1]
    except:
        end_of_question_token_id = tokenizer.convert_tokens_to_ids("?")

    reward_model = ContradictionReward(
        base_model=base_model,
        base_tokenizer=tokenizer,
        nli_model_name=config.reward.nli_model_name,
        likelihood_weight=config.task.likelihood_weight,
        answer_prompt_template=config.task.answer_prompt_template,
        termination_token_id=end_of_question_token_id,
        min_len=config.task.min_question_len,
        nli_batch_size=config.reward.nli_batch_size,
        use_hybrid_reward=False, # Disable hybrid reward for consistent scoring
    )
    # Move reward sub-models to the correct device
    reward_model.nli_model.to("cuda")
    reward_model.diversity_model.to("cuda")
    reward_model.device = "cuda"

    # --- Define your test cases here ---
    test_cases = [
        {
            "subject": "William Shakespeare",
            "edit_fact": "The play 'Hamlet' was written by Christopher Marlowe."
        },
        {
            "subject": "The Earth",
            "edit_fact": "The Earth is the largest planet in the solar system."
        },
        {
            "subject": "Mount Everest",
            "edit_fact": "Mount Everest is located in the Andes mountain range."
        },
        # Add more test cases as you like
    ]

    # --- Generation Parameters ---
    n_samples_to_generate = 10
    generation_temp = 1.2 # Use a slightly higher temp for more creative outputs

    for i, case in enumerate(test_cases):
        print("\n" + "="*80)
        print(f"--- Test Case #{i+1} ---")
        print(f"Subject: {case['subject']}")
        print(f"Edit Fact: {case['edit_fact']}")
        print("="*80 + "\n")

        # Prepare the prompt
        formatted_task_prompt = config.task.task_prompt.format(subject=case['subject'])
        prompt_tokens = tokenizer(formatted_task_prompt, return_tensors="pt")["input_ids"].to("cuda")
        prompt_tokens = prompt_tokens.expand(n_samples_to_generate, -1)
        prompt_len = prompt_tokens.shape[1]

        # Generate questions
        with torch.no_grad():
            generated_trajectories, _, _, _, _ = generate_and_return_termination_logprob(
                model=sampler_model,
                encoded_prompt=prompt_tokens,
                reward_fn=None, # We will score them afterwards
                termination_token_id=end_of_question_token_id,
                max_len=config.task.max_question_len,
                min_len=config.task.min_question_len,
                temperature=generation_temp,
                skip_rewards=True
            )

        # Decode the generated questions
        generated_questions_toks = generated_trajectories[:, prompt_len:]
        decoded_questions = tokenizer.batch_decode(generated_questions_toks, skip_special_tokens=True)

        # Score the generated questions
        _, unscaled_rewards, contradiction_scores, likelihood_scores = reward_model.score(
            generated_trajectories=generated_trajectories,
            z_prime_text=case['edit_fact'],
            prompt_length=prompt_len
        )
        
        # Calculate semantic diversity
        diversity_score = reward_model.calculate_semantic_diversity(decoded_questions)

        # Create a report using pandas DataFrame
        report_df = pd.DataFrame({
            "Generated Question": decoded_questions,
            "Unscaled Reward": unscaled_rewards.squeeze().cpu().numpy(),
            "Contradiction Score": contradiction_scores.squeeze().cpu().numpy(),
            "Likelihood Score": likelihood_scores.squeeze().cpu().numpy()
        })
        
        report_df = report_df.sort_values(by="Unscaled Reward", ascending=False).reset_index(drop=True)
        
        print(f"--- Generated Questions & Scores (Diversity: {diversity_score:.4f}) ---")
        print(report_df.to_string())
        print("\n")


if __name__ == "__main__":
    evaluate()