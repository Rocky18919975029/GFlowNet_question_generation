# download_models.py

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# --- CHANGE gpt2-large to gpt2 ---
SAMPLER_MODEL = "gpt2"
NLI_MODEL = "NDugar/v3-Large-mnli" # Keep the same NLI model

print(f"--- Downloading and caching model: {SAMPLER_MODEL} ---")
AutoTokenizer.from_pretrained(SAMPLER_MODEL)
AutoModelForCausalLM.from_pretrained(SAMPLER_MODEL)
print("...done.")

# This will just confirm the NLI model is already cached from previous runs
print(f"\n--- Verifying cache for model: {NLI_MODEL} ---")
AutoTokenizer.from_pretrained(NLI_MODEL)
AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
print("...done.")

print("\nAll models have been downloaded/verified in the local cache.")