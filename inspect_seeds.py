# inspect_seeds.py

import pandas as pd
import textwrap
import os

# --- Configuration ---
TRAIN_DATA_PATH = "data/zsre_1000.jsonl"
POOL_DATA_PATH = "clean_pool.jsonl"  # <-- We are checking this file
NUM_TO_SHOW = 5

# --- Main Script ---
print("--- Interactive Local Seed Data Inspector ---")
print(f"--- Reading from: {os.path.abspath(POOL_DATA_PATH)} ---")

# 1. Load both data files into pandas DataFrames
try:
    print(f"Loading training data from '{TRAIN_DATA_PATH}'...")
    df_train = pd.read_json(TRAIN_DATA_PATH, lines=True)
    print(f"Loaded {len(df_train)} records.")

    print(f"Loading CLEAN candidate pool from '{POOL_DATA_PATH}'...")
    df_pool = pd.read_json(POOL_DATA_PATH, lines=True)
    print(f"Loaded {len(df_pool)} total clean candidates.")
    if df_pool.empty:
        print("\nERROR: The clean_pool.jsonl file is empty!")
        exit(1)

except FileNotFoundError as e:
    print(f"\nERROR: Could not find a required data file: {e}")
    exit(1)

# 2. Main interactive loop
while True:
    # Randomly select one row from the original training data
    task_row = df_train.sample(1).iloc[0]
    
    subject = task_row['subject']
    original_fact = task_row['original_fact']
    edit_fact = task_row['edit_fact']
    
    # --- Display the Task Details ---
    print("\n" + "="*80)
    print("                      RANDOM TASK                           ")
    print("="*80)
    print(f"SUBJECT:        {subject}")
    print(textwrap.fill(f"ORIGINAL FACT:  {original_fact}", width=80))
    print(textwrap.fill(f"EDIT FACT:      {edit_fact}", width=80))
    print("-"*80)

    # --- Find, Sort, and Display the Matching Candidates ---
    # Match candidates based on BOTH subject and edit_fact to be precise
    matched_candidates = df_pool[
        (df_pool['subject'] == subject) & 
        (df_pool['edit_fact'] == edit_fact)
    ]
    
    if matched_candidates.empty:
        print("!! No candidate questions were found for this specific task in the clean pool. !!")
        print("!! This is expected if all generated answers for this task were invalid. !!")
    else:
        # Sort the matched candidates by their score (highest first)
        top_candidates = matched_candidates.sort_values(
            by='contradiction_score', ascending=False
        ).head(NUM_TO_SHOW)
        
        print(f"Found {len(matched_candidates)} candidates in the clean pool. Showing top {len(top_candidates)}:")
        print("-"*80)
        
        for index, row in top_candidates.iterrows():
            question = row['question']
            score = row['contradiction_score']
            answer_col = 'generated_answer' if 'generated_answer' in row else 'answer'
            answer = row[answer_col]

            print(f"  ▶ [Score: {score:8.4f}] Question: \"{question}\"")
            print(f"                     Answer:   \"{answer}\"")
            print()

    print("="*80)

    # Ask the user to continue or quit
    user_input = input("Press Enter to see another random sample, or type 'q' to quit: ")
    if user_input.lower() == 'q':
        break

print("\nInspector closed.")