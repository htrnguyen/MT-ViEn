# data_utils.py
import re

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text_detailed(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(
        r"[^\w\s.,!?àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]",
        "",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean_data(path, max_sentences=50000):
    try:
        df = pd.read_csv(path, on_bad_lines="skip", nrows=max_sentences)
        df.columns = ["en", "vi"]
        df["en"] = df["en"].apply(clean_text_detailed)
        df["vi"] = df["vi"].apply(clean_text_detailed)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return None


def split_data(df, test_size=0.1, val_size=0.1):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df, val_df = train_test_split(
        train_df, test_size=val_size / (1 - test_size), random_state=42
    )
    return train_df, val_df, test_df
