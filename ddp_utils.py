import os
import time

def ddp_print(rank, message):
    """
    A DDP-safe print function that includes the rank and a timestamp,
    and flushes the output to ensure it's visible immediately.
    """
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{timestamp}][Rank {rank}] {message}", flush=True)