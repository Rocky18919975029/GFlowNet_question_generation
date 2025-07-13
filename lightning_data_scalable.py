import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
# --- FIX: Removed ddp_print import ---
# from ddp_utils import ddp_print 

class ScalableDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        train_size: float = 0.9,
        limit_data: int = None,
        batch_size: int = 1,
        num_workers: int = 2
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.train_size_percent = int(train_size * 100)
        self.limit_data = limit_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Removed rank and ddp_print call from init
        # self.rank = int(os.environ.get("LOCAL_RANK", 0))
        # ddp_print(self.rank, "DataModule: __init__ finished.")

    def prepare_data(self):
        # Removed ddp_print calls
        # ddp_print(self.rank, f"DataModule: In prepare_data(), creating cache from {self.data_path}...")
        load_dataset('json', data_files=self.data_path, split='train')
        # ddp_print(self.rank, "DataModule: Finished prepare_data().")

    def setup(self, stage: str):
        # Removed ddp_print calls
        # ddp_print(self.rank, f"DataModule: In setup(stage='{stage}'), loading from cache...")
        dataset = load_dataset('json', data_files=self.data_path, split='train')

        if self.limit_data:
            dataset = dataset.select(range(self.limit_data))

        split_dataset = dataset.train_test_split(
            train_size=self.train_size_percent / 100.0,
            shuffle=True,
            seed=42
        )
        self.train_data = split_dataset['train']
        self.val_data = split_dataset['test']
        # ddp_print(self.rank, "DataModule: Finished setup().")

    def _collate_fn(self, batch):
        item = batch[0]
        return {
            "z": self.tokenizer(item["original_fact"], return_tensors="pt")["input_ids"].squeeze(0),
            "z_prime": self.tokenizer(item["edit_fact"], return_tensors="pt")["input_ids"].squeeze(0),
            "subject": self.tokenizer(item["subject"], return_tensors="pt")["input_ids"].squeeze(0),
        }

    def train_dataloader(self):
        # Removed ddp_print calls
        # ddp_print(self.rank, "DataModule: Creating train_dataloader()...")
        loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers
        )
        # ddp_print(self.rank, "DataModule: train_dataloader() created and returned.")
        return loader

    def val_dataloader(self):
        # Removed ddp_print calls
        # ddp_print(self.rank, "DataModule: Creating val_dataloader()...")
        loader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers
        )
        # ddp_print(self.rank, "DataModule: val_dataloader() created and returned.")
        return loader