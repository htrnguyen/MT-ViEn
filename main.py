# main.py
import os

import torch

from config import *
from data_utils import load_and_clean_data, split_data
from datasets import create_dataloaders
from eval_utils import calculate_bleu, calculate_rouge, generate_translations
from models.gpt_scratch import GPTScratchModel
from models.pretrained_models import load_gpt2_model, load_marian_model
from models.transformer_scratch import TransformerScratchModel
from tokenization_utils import load_bpe_tokenizer, train_bpe_tokenizer
from train_utils import train_model
from utils import clear_gpu_cache, plot_training_history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Tiền xử lý dữ liệu
df = load_and_clean_data(RAW_DATA_PATH, MAX_SENTENCES)
train_df, val_df, test_df = split_data(df)

# 2. Huấn luyện BPE Tokenizer
bpe_texts = list(train_df["en"]) + list(train_df["vi"])
tokenizer_dir = os.path.join(OUTPUT_DIR, "bpe_tokenizer")
os.makedirs(tokenizer_dir, exist_ok=True)
bpe_tokenizer = train_bpe_tokenizer(
    bpe_texts, tokenizer_dir, vocab_size=BPE_VOCAB_SIZE, min_frequency=BPE_MIN_FREQUENCY
)

# 3. Tạo DataLoaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_df, "en", "vi", bpe_tokenizer, batch_size=BATCH_SIZE_SCRATCH_TRANSFORMER
)

# 4. Huấn luyện Transformer từ đầu
for src, tgt in DIRECTIONS:
    print(f"\n{'='*40}\nTraining Transformer from Scratch: {src} -> {tgt}\n{'='*40}")

    model = TransformerScratchModel(
        vocab_size=bpe_tokenizer.get_vocab_size(),
        d_model=TRANSFORMER_CONFIG["d_model"],
        nhead=TRANSFORMER_CONFIG["nhead"],
        num_encoder_layers=TRANSFORMER_CONFIG["num_encoder_layers"],
        num_decoder_layers=TRANSFORMER_CONFIG["num_decoder_layers"],
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_SCRATCH)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    model_dir = os.path.join(OUTPUT_DIR, "models", "transformer_scratch")
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        bpe_tokenizer,
        device,
        optimizer,
        scheduler,
        EPOCHS_SCRATCH_TRANSFORMER,
        model_dir,
        f"{src}_{tgt}",
    )

    # Vẽ biểu đồ
    plot_path = os.path.join(
        OUTPUT_DIR, "plots", f"transformer_{src}_{tgt}_history.png"
    )
    plot_training_history(history, f"Transformer ({src}->{tgt})", plot_path)

    # Đánh giá
    predictions, references = generate_translations(
        model, test_loader, bpe_tokenizer, device
    )
    bleu = calculate_bleu(predictions, references)
    rouge = calculate_rouge(predictions, references)
    print(f"\nTest Results - BLEU: {bleu:.2f}, ROUGE: {rouge}")

    clear_gpu_cache()
