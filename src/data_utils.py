import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text):
    """Chuẩn hóa văn bản tiếng Anh & Việt."""
    text = str(text).lower().strip()
    # Giữ lại chữ cái, số, dấu tiếng Việt và dấu câu cơ bản
    text = re.sub(
        r"[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
        r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
        r"ùúụủũưừứựửữỳýỵỷỹđ\s.,!?'-]",
        "",
        text,
    )
    # Chuẩn hóa khoảng trắng
    text = " ".join(text.split())
    return text


def load_and_clean_csv(path, lang1="en", lang2="vi"):
    """Đọc và làm sạch dữ liệu từ CSV."""
    df = pd.read_csv(path)
    df[lang1] = df[lang1].apply(clean_text)
    df[lang2] = df[lang2].apply(clean_text)
    df = df.dropna().drop_duplicates(subset=[lang1, lang2])
    df = df[(df[lang1].str.strip() != "") & (df[lang2].str.strip() != "")]
    return df.reset_index(drop=True)


def save_splits(train_df, val_df, test_df, out_dir):
    """Lưu các file train/val/test CSV."""
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)


def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """Chia dữ liệu thành train, val, test."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=random_state
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def print_data_stats(df, lang1="en", lang2="vi", name="Data"):
    print(f"---- {name} ----")
    print(f"Số cặp câu: {len(df)}")
    print(f"5 cặp câu mẫu:")
    print(df[[lang1, lang2]].head())
    print("\nThống kê độ dài câu:")
    df[f"{lang1}_len"] = df[lang1].apply(lambda x: len(x.split()))
    df[f"{lang2}_len"] = df[lang2].apply(lambda x: len(x.split()))
    print(df[[f"{lang1}_len", f"{lang2}_len"]].describe())


def analyze_length_distribution(df, lang1="en", lang2="vi"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    df[f"{lang1}_len"] = df[lang1].apply(lambda x: len(x.split()))
    df[f"{lang2}_len"] = df[lang2].apply(lambda x: len(x.split()))
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df[f"{lang1}_len"], bins=30, kde=True, color="blue")
    plt.title(f"Độ dài câu {lang1}")
    plt.xlabel("Số từ")
    plt.subplot(1, 2, 2)
    sns.histplot(df[f"{lang2}_len"], bins=30, kde=True, color="green")
    plt.title(f"Độ dài câu {lang2}")
    plt.xlabel("Số từ")
    plt.tight_layout()
    plt.show()


# --- Ví dụ sử dụng (nên gọi từ main.ipynb) ---

if __name__ == "__main__":
    # Đọc và làm sạch dữ liệu
    df = load_and_clean_csv("../data/eng-vie.csv")
    print_data_stats(df)
    # Chia và lưu dữ liệu
    train_df, val_df, test_df = split_data(df)
    save_splits(train_df, val_df, test_df, "../output")
    # Trực quan hóa phân bố độ dài câu
    analyze_length_distribution(train_df)
