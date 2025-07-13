import warnings
import pandas as pd
from torch.utils.data import DataLoader, MapDataPipe
import pytorch_lightning as pl
import os

warnings.filterwarnings("ignore", ".*does not have many workers.*")

class StatementPairDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, tokenizer, train_size: float = 0.9, limit_data: int = None):
        super().__init__()
        self.save_hyperparameters(ignore="tokenizer")
        self.tokenizer = tokenizer
        self.train_data = None
        self.val_data = None

    def setup(self, stage: str):
        rank = os.environ.get("LOCAL_RANK", "N/A")
        print(f"--- [DataModule on Rank {rank}] Entering setup()... ---")
        
        print(f"--- [DataModule on Rank {rank}] Reading pickle file: {self.hparams.data_path} ---")
        df = pd.read_pickle(self.hparams.data_path)
        print(f"--- [DataModule on Rank {rank}] Pickle file read successfully. ---")
        
        if self.hparams.limit_data is not None:
            df = df.head(self.hparams.limit_data)
        if len(df) == 1:
            num_train = 1
        else:
            num_train = int(len(df) * self.hparams.train_size)
            if len(df) > 0 and num_train == 0:
                num_train = 1
                
        train_df = df.iloc[:num_train].reset_index(drop=True)
        val_df = df.iloc[num_train:].reset_index(drop=True)
        self.train_data = StatementPairDataPipe(train_df, self.tokenizer)
        self.val_data = StatementPairDataPipe(val_df, self.tokenizer)
        
        print(f"--- [DataModule on Rank {rank}] Finished setup(). ---")

    def train_dataloader(self):
        rank = os.environ.get("LOCAL_RANK", "N/A")
        print(f"--- [DataModule on Rank {rank}] Creating train_dataloader(). ---")
        return DataLoader(self.train_data, shuffle=True, batch_size=None, num_workers=0)

    def val_dataloader(self):
        rank = os.environ.get("LOCAL_RANK", "N/A")
        print(f"--- [DataModule on Rank {rank}] Creating val_dataloader(). ---")
        return DataLoader(self.val_data, batch_size=None, num_workers=0)

class StatementPairDataPipe(MapDataPipe):
    def __init__(self, dataframe: pd.DataFrame, tokenizer):
        super().__init__()
        self.data = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        z_text = row["original_fact"]
        z_prime_text = row["edit_fact"]
        subject_text = row["subject"]

        tokenized_z = self.tokenizer(z_text, return_tensors="pt")["input_ids"]
        tokenized_z_prime = self.tokenizer(z_prime_text, return_tensors="pt")["input_ids"]
        tokenized_subject = self.tokenizer(subject_text, return_tensors="pt")["input_ids"]

        return {
            "z": tokenized_z.squeeze(0),
            "z_prime": tokenized_z_prime.squeeze(0),
            "subject": tokenized_subject.squeeze(0),
        }