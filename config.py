# config.py

import os

# Đường dẫn dữ liệu
RAW_DATA_PATH = "data/eng-vie.csv"
OUTPUT_DIR = "output"

# Tokenizer BPE
BPE_VOCAB_SIZE = 30000
BPE_MIN_FREQUENCY = 2
SPECIAL_TOKENS_BPE = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# Model Transformer từ đầu
TRANSFORMER_CONFIG = {
    "d_model": 256,
    "nhead": 4,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "dim_feedforward": 512,
    "dropout": 0.2,
    "max_len": 40,
}

# GPT từ đầu
GPT_SCRATCH_CONFIG = {
    "d_model": 256,
    "nhead": 4,
    "num_decoder_layers": 4,
    "dim_feedforward": 512,
    "dropout": 0.2,
    "max_len": 80,
}

# Huấn luyện chung
EPOCHS_SCRATCH_TRANSFORMER = 3
EPOCHS_SCRATCH_GPT = 3
EPOCHS_FINETUNE_MARIAN = 3
EPOCHS_FINETUNE_GPT2 = 3

LEARNING_RATE_SCRATCH = 1e-3
LEARNING_RATE_SCRATCH_GPT = 5e-4
LEARNING_RATE_MARIAN = 3e-5
LEARNING_RATE_GPT2 = 2e-5

BATCH_SIZE_SCRATCH_TRANSFORMER = 16
BATCH_SIZE_SCRATCH_GPT = 16
BATCH_SIZE_PRETRAINED = 8

WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 3

# Đánh giá
DIRECTIONS = [("en", "vi"), ("vi", "en")]
MAX_SENTENCES = 50000
