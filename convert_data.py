# convert_data.py

import pandas as pd

# The path to your original pickle file
input_pkl_path = "data/ZSRE_1000.pkl"

# The path for the new JSON Lines file
output_jsonl_path = "data/zsre_1000.jsonl"

print(f"Reading data from {input_pkl_path}...")
df = pd.read_pickle(input_pkl_path)

print(f"Converting and writing to {output_jsonl_path}...")
# The 'records' orientation creates one JSON object per row
df.to_json(output_jsonl_path, orient='records', lines=True)

print("Conversion complete.")