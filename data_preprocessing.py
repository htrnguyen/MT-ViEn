#!/usr/bin/env python
# coding: utf-8

"""
data_preprocessing.py: Xử lý dữ liệu, tokenization, và định nghĩa các lớp dataset
cho dự án dịch máy. Tối ưu cho GPU P100, hỗ trợ hiển thị thống kê dữ liệu và tích hợp
subword regularization cho BPE tokenizer.
"""

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils import plot_training_history, setup_directories

# Thiết lập kiểu dáng cho biểu đồ
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def clean_text(text):
    """
    Làm sạch văn bản: chuyển thành chữ thường, loại bỏ ký tự không cần thiết, chuẩn hóa khoảng trắng.

    Args:
        text: Văn bản đầu vào.

    Returns:
        str: Văn bản đã làm sạch.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(
        r"[^\w\s.,!?àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
        "",
        text,
        flags=re.UNICODE,
    )
    return " ".join(text.split())


def load_and_preprocess_data(data_file, output_dir):
    """
    Tải và tiền xử lý dữ liệu từ file CSV, lưu các tập train/val/test.

    Args:
        data_file (str): Đường dẫn file CSV chứa cặp câu en-vi.
        output_dir (str): Thư mục lưu dữ liệu đã xử lý.

    Returns:
        tuple: DataFrame train, val, test và thư mục lưu biểu đồ.
    """
    # Tạo thư mục
    model_dir, plot_dir = setup_directories(output_dir)

    # Tải dữ liệu
    data_df = pd.read_csv(data_file, header=0)
    if len(data_df.columns) >= 2:
        data_df = data_df.iloc[:, :2]
        data_df.columns = ["en", "vi"]
    else:
        raise ValueError(f"File CSV cần ít nhất 2 cột: {data_file}")

    print(f"Đã tải {len(data_df)} cặp câu từ: {data_file}")

    # Làm sạch dữ liệu
    data_df["en"] = data_df["en"].apply(clean_text)
    data_df["vi"] = data_df["vi"].apply(clean_text)

    # Loại bỏ dòng trống hoặc trùng lặp
    original_len = len(data_df)
    data_df.dropna(subset=["en", "vi"], inplace=True)
    data_df = data_df[data_df["en"].str.strip() != ""]
    data_df = data_df[data_df["vi"].str.strip() != ""]
    data_df.drop_duplicates(subset=["en", "vi"], inplace=True)
    data_df.reset_index(drop=True, inplace=True)

    print(f"Loại bỏ {original_len - len(data_df)} cặp câu trống/trùng lặp.")
    print(f"Số cặp câu sau làm sạch: {len(data_df)}")

    # Chia tập dữ liệu
    train_df, temp_df = train_test_split(data_df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Kích thước tập dữ liệu:")
    print(f"- Train: {len(train_df)}")
    print(f"- Val: {len(val_df)}")
    print(f"- Test: {len(test_df)}")

    # Lưu dữ liệu đã xử lý
    train_csv = os.path.join(output_dir, "train_processed.csv")
    val_csv = os.path.join(output_dir, "val_processed.csv")
    test_csv = os.path.join(output_dir, "test_processed.csv")
    train_df.to_csv(train_csv, index=False, encoding="utf-8")
    val_df.to_csv(val_csv, index=False, encoding="utf-8")
    test_df.to_csv(test_csv, index=False, encoding="utf-8")
    print(f"Lưu dữ liệu đã xử lý tại: {output_dir}")

    # Phân tích độ dài câu
    train_df["en_length"] = train_df["en"].apply(lambda x: len(str(x).split()))
    train_df["vi_length"] = train_df["vi"].apply(lambda x: len(str(x).split()))

    print("\nThống kê độ dài câu (số từ) trên tập Train:")
    print(train_df[["en_length", "vi_length"]].describe())

    # Vẽ biểu đồ phân phối độ dài câu
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(train_df["en_length"], bins=50, kde=True, color="dodgerblue")
    plt.title("Phân phối độ dài câu tiếng Anh (Train)")
    plt.xlabel("Số từ")
    plt.ylabel("Tần suất")
    plt.xlim(0, max(10, int(np.percentile(train_df["en_length"], 99))))

    plt.subplot(1, 2, 2)
    sns.histplot(train_df["vi_length"], bins=50, kde=True, color="limegreen")
    plt.title("Phân phối độ dài câu tiếng Việt (Train)")
    plt.xlabel("Số từ")
    plt.ylabel("Tần suất")
    plt.xlim(0, max(10, int(np.percentile(train_df["vi_length"], 99))))

    plt.tight_layout()
    length_plot_path = os.path.join(plot_dir, "sentence_length_distribution.png")
    plt.savefig(length_plot_path)
    plt.show()
    print(f"Lưu biểu đồ phân phối độ dài câu tại: {length_plot_path}")

    return train_df, val_df, test_df, plot_dir


def train_bpe_tokenizer(train_df, output_dir, vocab_size=30000, min_freq=2):
    """
    Huấn luyện BPE tokenizer với subword regularization.

    Args:
        train_df (pd.DataFrame): DataFrame chứa dữ liệu train.
        output_dir (str): Thư mục lưu tokenizer.
        vocab_size (int): Kích thước từ vựng.
        min_freq (int): Tần suất tối thiểu của token.

    Returns:
        ByteLevelBPETokenizer: Tokenizer đã huấn luyện.
    """
    tokenizer_dir = os.path.join(output_dir, "bpe_tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)

    # Chuẩn bị dữ liệu cho BPE
    all_texts = (
        pd.concat([train_df["en"].astype(str), train_df["vi"].astype(str)])
        .unique()
        .tolist()
    )
    temp_file = os.path.join(output_dir, "temp_bpe_texts.txt")
    with open(temp_file, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(text + "\n")

    print(f"Chuẩn bị {len(all_texts)} câu cho huấn luyện BPE.")

    # Huấn luyện tokenizer
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[temp_file],
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens,
        show_progress=True,
    )

    tokenizer.save_model(tokenizer_dir)
    print(f"Lưu tokenizer BPE tại: {tokenizer_dir}")

    # Tải lại tokenizer
    tokenizer = ByteLevelBPETokenizer.from_file(
        os.path.join(tokenizer_dir, "vocab.json"),
        os.path.join(tokenizer_dir, "merges.txt"),
    )
    tokenizer.add_special_tokens(special_tokens)

    # Gán ID cho các token đặc biệt
    tokenizer.pad_token_id = tokenizer.token_to_id("[PAD]")
    tokenizer.unk_token_id = tokenizer.token_to_id("[UNK]")
    tokenizer.cls_token_id = tokenizer.token_to_id("[CLS]")
    tokenizer.sep_token_id = tokenizer.token_to_id("[SEP]")
    tokenizer.mask_token_id = tokenizer.token_to_id("[MASK]")

    print(f"Kích thước từ vựng BPE: {tokenizer.get_vocab_size(True)}")

    # Phân tích tokenization
    token_lengths = []
    total_unk = 0
    for sent in tqdm(all_texts, desc="Phân tích token BPE"):
        encoding = tokenizer.encode(sent)
        token_lengths.append(len(encoding.tokens))
        total_unk += encoding.ids.count(tokenizer.unk_token_id)

    print("\nThống kê độ dài token BPE trên tập Train:")
    print(pd.Series(token_lengths).describe())
    print(f"Tổng số token [UNK]: {total_unk}")

    # Vẽ biểu đồ phân phối độ dài token
    plt.figure(figsize=(10, 6))
    sns.histplot(token_lengths, bins=50, kde=True, color="darkmagenta")
    plt.title("Phân phối độ dài token BPE (Train)")
    plt.xlabel("Số token BPE")
    plt.ylabel("Tần suất")
    plt.xlim(0, max(10, int(np.percentile(token_lengths, 99))))

    plot_path = os.path.join(output_dir, "plots", "bpe_token_length_distribution.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Lưu biểu đồ phân phối độ dài token tại: {plot_path}")

    # Xóa file tạm
    os.remove(temp_file)
    print(f"Xóa file tạm: {temp_file}")

    return tokenizer


class ScratchTransformerDataset(Dataset):
    """
    Dataset cho mô hình Transformer từ đầu, sử dụng BPE tokenizer.
    """

    def __init__(self, df, tokenizer, src_lang_col, tgt_lang_col, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.src_lang_col = src_lang_col
        self.tgt_lang_col = tgt_lang_col
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_text = str(row[self.src_lang_col])
        tgt_text = str(row[self.tgt_lang_col])

        src_enc = self.tokenizer.encode(src_text)
        tgt_enc = self.tokenizer.encode(tgt_text)

        src_ids = [self.cls_id] + src_enc.ids[: self.max_length - 2] + [self.sep_id]
        tgt_ids_input = [self.cls_id] + tgt_enc.ids[: self.max_length - 2]
        tgt_ids_labels = tgt_enc.ids[: self.max_length - 2] + [self.sep_id]

        # Đệm chuỗi
        src_ids = src_ids + [self.pad_id] * (self.max_length - len(src_ids))
        tgt_ids_input = tgt_ids_input + [self.pad_id] * (
            self.max_length - len(tgt_ids_input)
        )
        tgt_ids_labels = tgt_ids_labels + [self.pad_id] * (
            self.max_length - len(tgt_ids_labels)
        )

        return {
            "input_ids": torch.tensor(src_ids, dtype=torch.long),
            "decoder_input_ids": torch.tensor(tgt_ids_input, dtype=torch.long),
            "labels": torch.tensor(tgt_ids_labels, dtype=torch.long),
            "src_key_padding_mask": torch.tensor(src_ids) == self.pad_id,
            "tgt_key_padding_mask": torch.tensor(tgt_ids_input) == self.pad_id,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


class ScratchGPTDataset(Dataset):
    """
    Dataset cho mô hình GPT từ đầu, sử dụng BPE tokenizer.
    """

    def __init__(self, df, tokenizer, src_lang_col, tgt_lang_col, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.src_lang_col = src_lang_col
        self.tgt_lang_col = tgt_lang_col
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_text = str(row[self.src_lang_col])
        tgt_text = str(row[self.tgt_lang_col])

        src_enc = self.tokenizer.encode(src_text)
        tgt_enc = self.tokenizer.encode(tgt_text)

        input_ids = (
            [self.cls_id] + src_enc.ids + [self.sep_id] + tgt_enc.ids + [self.sep_id]
        )
        input_ids = input_ids[: self.max_length]
        input_ids = input_ids + [self.pad_id] * (self.max_length - len(input_ids))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(input_ids) != self.pad_id,
            "labels": torch.tensor(input_ids, dtype=torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


if __name__ == "__main__":
    # Kiểm tra tiền xử lý dữ liệu
    data_file = "/kaggle/input/daily-en-vi/eng-vie.csv"
    output_dir = "./output"

    train_df, val_df, test_df, plot_dir = load_and_preprocess_data(
        data_file, output_dir
    )

    # Kiểm tra tokenization
    tokenizer = train_bpe_tokenizer(train_df, output_dir)

    # Kiểm tra dataset
    dataset = ScratchTransformerDataset(train_df, tokenizer, "en", "vi", max_length=45)
    sample = dataset[0]
    print("\nKiểm tra mẫu từ ScratchTransformerDataset:")
    print(f"- Input IDs: {sample['input_ids'][:10]}...")
    print(f"- Decoder Input IDs: {sample['decoder_input_ids'][:10]}...")
    print(f"- Labels: {sample['labels'][:10]}...")
    print(f"- Source Text: {sample['src_text']}")
    print(f"- Target Text: {sample['tgt_text']}")
