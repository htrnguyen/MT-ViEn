import os

from tokenizers import ByteLevelBPETokenizer


def train_bpe_tokenizer(
    texts,
    save_dir,
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    overwrite=True,
):
    """Train và lưu BPE tokenizer từ danh sách câu."""
    os.makedirs(save_dir, exist_ok=True)
    data_file = os.path.join(save_dir, "bpe_texts.txt")
    # Lưu toàn bộ câu ra file txt
    if overwrite or not os.path.exists(data_file):
        with open(data_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text.strip() + "\n")
    # Train BPE
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[data_file],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )
    tokenizer.save_model(save_dir)
    return tokenizer


def load_bpe_tokenizer(save_dir):
    """Load BPE tokenizer đã train."""
    return ByteLevelBPETokenizer(
        os.path.join(save_dir, "vocab.json"), os.path.join(save_dir, "merges.txt")
    )


def test_tokenizer(tokenizer, sample_texts):
    """Kiểm tra encode/decode cho danh sách câu."""
    for text in sample_texts:
        encoding = tokenizer.encode(text)
        print(f"\nCâu gốc: {text}")
        print(f"Tokens: {encoding.tokens[:10]}...")
        print(f"IDs: {encoding.ids[:10]}")
        print(f"Giải mã: {tokenizer.decode(encoding.ids, skip_special_tokens=True)}")


def analyze_token_length(tokenizer, texts, sample_size=1000):
    import matplotlib.pyplot as plt
    import seaborn as sns

    token_lengths = [len(tokenizer.encode(t).ids) for t in texts[:sample_size]]
    plt.figure(figsize=(8, 5))
    sns.histplot(token_lengths, bins=30, kde=True, color="purple")
    plt.title("Phân bố độ dài token BPE")
    plt.xlabel("Số token")
    plt.show()


# --- Ví dụ sử dụng (nên gọi từ main.ipynb) ---

if __name__ == "__main__":
    import pandas as pd

    # Giả sử đã xử lý dữ liệu và lưu tại output/train.csv, output/val.csv, output/test.csv
    train = pd.read_csv("../output/train.csv")
    val = pd.read_csv("../output/val.csv")
    all_texts = (
        pd.concat([train["en"], train["vi"], val["en"], val["vi"]]).dropna().tolist()
    )
    tokenizer = train_bpe_tokenizer(
        all_texts, "../output/bpe_tokenizer", vocab_size=32000
    )
    test_tokenizer(tokenizer, ["chạy đi!", "i love you.", "hãy giúp tôi với!"])
    analyze_token_length(tokenizer, all_texts)
