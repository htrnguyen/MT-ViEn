# datasets.py
import torch
from torch.utils.data import DataLoader, Dataset

from config import BPE_VOCAB_SIZE, MAX_LEN_GPT, MAX_LEN_SCRATCH
from tokenization_utils import load_bpe_tokenizer


class TranslationDataset(Dataset):
    def __init__(self, df, src_lang_col, tgt_lang_col, tokenizer, max_length=512):
        self.df = df
        self.src_texts = df[src_lang_col].tolist()
        self.tgt_texts = df[tgt_lang_col].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        src_encoded = self.tokenizer.encode(
            src_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tgt_encoded = self.tokenizer.encode(
            tgt_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": src_encoded.squeeze(0),
            "labels": tgt_encoded.squeeze(0),
            "attention_mask": (src_encoded != self.tokenizer.pad_token_id).squeeze(0),
        }


def create_dataloaders(
    train_df, val_df, test_df, src_lang_col, tgt_lang_col, tokenizer, batch_size=16
):
    train_dataset = TranslationDataset(train_df, src_lang_col, tgt_lang_col, tokenizer)
    val_dataset = TranslationDataset(val_df, src_lang_col, tgt_lang_col, tokenizer)
    test_dataset = TranslationDataset(test_df, src_lang_col, tgt_lang_col, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
