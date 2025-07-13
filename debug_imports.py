# debug_imports.py
import os
import socket

# --- Print initial environment info ---
rank = os.environ.get("SLURM_PROCID", "N/A")
hostname = socket.gethostname()
print(f"--- [Rank {rank} on {hostname}] Starting import debug script. ---")

try:
    import hydra
    print(f"--- [Rank {rank}] Imported 'hydra' successfully. ---")

    import torch
    print(f"--- [Rank {rank}] Imported 'torch' successfully. ---")
    
    # Check if torch can see the GPUs
    print(f"--- [Rank {rank}] torch.cuda.is_available(): {torch.cuda.is_available()} ---")
    print(f"--- [Rank {rank}] torch.cuda.device_count(): {torch.cuda.device_count()} ---")

    import pytorch_lightning as pl
    print(f"--- [Rank {rank}] Imported 'pytorch_lightning' successfully. ---")

    import transformers
    print(f"--- [Rank {rank}] Imported 'transformers' successfully. ---")
    
    import peft
    print(f"--- [Rank {rank}] Imported 'peft' successfully. ---")

    import bitsandbytes
    print(f"--- [Rank {rank}] Imported 'bitsandbytes' successfully. ---")

    print(f"--- [Rank {rank}] All major imports successful. ---")

except Exception as e:
    print(f"--- [Rank {rank}] AN ERROR OCCURRED: {e} ---")