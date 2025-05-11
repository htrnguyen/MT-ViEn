# tokenization_utils.py
import os

from tokenizers import ByteLevelBPETokenizer

from config import SPECIAL_TOKENS_BPE


def train_bpe_tokenizer(
    data,
    save_path,
    vocab_size=30000,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS_BPE,
):
    os.makedirs(save_path, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        data,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )
    tokenizer.save_model(save_path)
    return tokenizer


def load_bpe_tokenizer(save_path):
    return ByteLevelBPETokenizer.from_file(save_path)
