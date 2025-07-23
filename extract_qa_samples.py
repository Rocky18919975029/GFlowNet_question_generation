# extract_qa_samples.py

import os
import re
import random

# --- Configuration ---
QA_LOG_FILE = "training_probes/qa_log.txt"
NUM_SAMPLES_PER_EXPERIMENT = 5 # Number of random Q&A pairs to extract per experiment

# --- Helper to parse a single experiment block ---
def parse_experiment_block(block_content):
    """Parses a string block corresponding to one experiment's QA log."""
    samples = []
    current_entry = {}
    
    lines = block_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue

        step_subject_match = re.match(r"--- Step: (\d+), Subject: (.+) ---", line)
        if step_subject_match:
            if current_entry: # Save previous entry if exists
                samples.append(current_entry)
            current_entry = {
                "step": int(step_subject_match.group(1)),
                "subject": step_subject_match.group(2)
            }
        elif line.startswith("Q:"):
            current_entry["question"] = line[2:].strip()
        elif line.startswith("A:"):
            current_entry["answer"] = line[2:].strip()
    
    if current_entry: # Add the last entry
        samples.append(current_entry)
        
    return samples

# --- Main Script ---
def main():
    if not os.path.exists(QA_LOG_FILE):
        print(f"ERROR: QA log file not found at '{QA_LOG_FILE}'.")
        print("Please ensure the training runs have completed and the path is correct.")
        exit(1)

    print(f"--- Extracting random QA samples from '{QA_LOG_FILE}' ---")

    with open(QA_LOG_FILE, 'r', encoding='utf-8') as f:
        full_log_content = f.read()

    # Split the log into blocks by the "NEW RUN STARTED AT" header
    # The regex includes the boundary lines (===...)
    experiment_blocks = re.split(r"={80}\nNEW RUN STARTED AT:.+\n={80}\n\n", full_log_content)
    
    # The first split part might be empty or contain only old headers if the file was appended to
    # We filter out empty blocks.
    experiment_blocks = [block.strip() for block in experiment_blocks if block.strip()]

    if not experiment_blocks:
        print("No distinct experiment blocks found in the log file. Is the log file empty or malformed?")
        exit(1)

    print(f"Found {len(experiment_blocks)} distinct experiment logs.")

    all_sampled_qa = {}

    for i, block_content in enumerate(experiment_blocks):
        experiment_id = f"Experiment {i+1}" # Assuming sequential order from the sweep script
        print(f"\n--- Processing {experiment_id} ---")
        
        all_samples_in_block = parse_experiment_block(block_content)
        
        if not all_samples_in_block:
            print(f"  No Q&A entries found for {experiment_id}.")
            continue

        # Randomly select samples
        num_to_sample = min(NUM_SAMPLES_PER_EXPERIMENT, len(all_samples_in_block))
        sampled_qa = random.sample(all_samples_in_block, num_to_sample)
        
        all_sampled_qa[experiment_id] = sampled_qa
        print(f"  Sampled {num_to_sample} Q&A pairs from {len(all_samples_in_block)} total entries.")

    # --- Print all sampled QA for review ---
    print("\n" + "="*80)
    print("           Randomly Sampled Questions and Answers           ")
    print("="*80)

    for exp_id, samples in all_sampled_qa.items():
        print(f"\n### {exp_id} Samples (Answer Quality Weight: {get_aqw_from_exp_id(exp_id)}) ###")
        for j, sample in enumerate(samples):
            print(f"  --- Sample {j+1} (Step: {sample.get('step', 'N/A')}) ---")
            print(f"  Subject: {sample.get('subject', 'N/A')}")
            print(f"  Q: {sample.get('question', 'N/A')}")
            print(f"  A: {sample.get('answer', 'N/A')}")
            print("-" * 30)
            
    print("\nReady for qualitative analysis!")

# Function to get AQW from experiment ID (manual mapping based on sweep script)
def get_aqw_from_exp_id(exp_id):
    if "Experiment 1" in exp_id: return "0.1 (inferred)"
    if "Experiment 2" in exp_id: return "0.25 (inferred)"
    if "Experiment 3" in exp_id: return "0.5 (inferred)"
    if "Experiment 4" in exp_id: return "1.0 (inferred)"
    return "Unknown"

if __name__ == "__main__":
    main()