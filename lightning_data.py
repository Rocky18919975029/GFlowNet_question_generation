# lightning_data.py

import warnings
import pandas as pd
from torch.utils.data import DataLoader, MapDataPipe
import pytorch_lightning as pl

warnings.filterwarnings("ignore", ".*does not have many workers.*")

class StatementPairDataModule(pl.LightningDataModule):
    """
    A LightningDataModule for loading (original_fact, edit_fact, subject) tuples from a pickle file.
    """
    def __init__(self, data_path: str, tokenizer, train_size: float = 0.9, limit_data: int = None):
        super().__init__()
        self.save_hyperparameters(ignore="tokenizer")
        self.tokenizer = tokenizer
        self.train_data = None
        self.val_data = None

    def setup(self, stage: str):
        df = pd.read_pickle(self.hparams.data_path)
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

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=None, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=None, num_workers=0)

class StatementPairDataPipe(MapDataPipe):
    """
    A custom DataPipe that processes DataFrame rows into tokenized tensors for z, z_prime, and subject.
    """
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