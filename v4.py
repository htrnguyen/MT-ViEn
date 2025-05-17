#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(
    "pip install transformers sentencepiece datasets evaluate sacrebleu rouge_score -q"
)


# In[2]:


# Import các thư viện cần thiết
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
from transformers import MarianMTModel, MarianTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from transformers import get_linear_schedule_with_warmup
from torch.amp import GradScaler, autocast
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from tqdm.auto import tqdm
import numpy as np
import time
import math
import warnings

# Tắt các cảnh báo
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Thiết lập môi trường
sns.set(style="whitegrid")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "/kaggle/working/"
os.makedirs(output_dir, exist_ok=True)
print(f"Sử dụng thiết bị: {device}")


# Hàm theo dõi tài nguyên
def print_resource_usage():
    try:
        import psutil

        print(f"CPU Usage: {psutil.cpu_percent():.1f}%")
    except:
        print("Không thể truy cập thông tin CPU.")


print_resource_usage()

# Dictionary để lưu metrics của các mô hình
model_metrics = {
    "MarianMT": {
        "train_losses": [],
        "val_losses": [],
        "val_bleus": [],
        "train_times": [],
    },
    "Scratch Transformer": {
        "train_losses": [],
        "val_losses": [],
        "val_bleus": [],
        "train_times": [],
    },
    "GPT-2": {"train_losses": [], "val_losses": [], "val_bleus": [], "train_times": []},
    "Scratch GPT": {
        "train_losses": [],
        "val_losses": [],
        "val_bleus": [],
        "train_times": [],
    },
}

EPOCH_NUM = 50


# # TIỀN XỬ LÝ DỮ LIỆU

# In[3]:


# Đọc dữ liệu
data_df = pd.read_csv("/kaggle/input/daily-en-vi/eng-vie.csv")
print(f"Số cặp câu ban đầu: {len(data_df)}")
print("5 cặp câu mẫu đầu tiên:")
data_df.head()


# In[4]:


# Làm sạch dữ liệu
def clean_text(text):
    text = str(text).lower()
    text = re.sub(
        r"[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s.,!?]",
        "",
        text,
    )
    return " ".join(text.split())


data_df["en"] = data_df["en"].apply(clean_text)
data_df["vi"] = data_df["vi"].apply(clean_text)
data_df = data_df.dropna().drop_duplicates(subset=["en", "vi"])
data_df = data_df[data_df["en"].str.strip() != ""]
data_df = data_df[data_df["vi"].str.strip() != ""]
print(f"Số cặp câu sau khi làm sạch: {len(data_df)}")
print("5 cặp câu mẫu sau khi làm sạch:")
data_df.head()


# In[5]:


# Xử lý nhiều bản dịch tham chiếu
reference_dict = data_df.groupby("en")["vi"].apply(list).to_dict()

# Chọn ngẫu nhiên một bản dịch cho huấn luyện để đơn giản hóa
# data_df = data_df.groupby('en').sample(n=1, random_state=42).reset_index(drop=True)
print(f"Số cặp câu sau khi chọn ngẫu nhiên một bản dịch: {len(data_df)}")


# In[6]:


# Chia dữ liệu
train_df, temp_df = train_test_split(data_df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
print(f"Kích thước tập huấn luyện: {len(train_df)}")
print(f"Kích thước tập xác thực: {len(val_df)}")
print(f"Kích thước tập kiểm tra: {len(test_df)}")


# In[7]:


# Trực quan hóa độ dài câu
train_df["en_length"] = train_df["en"].apply(lambda x: len(x.split()))
train_df["vi_length"] = train_df["vi"].apply(lambda x: len(x.split()))
print("\nThống kê độ dài câu (tập huấn luyện):")
train_df[["en_length", "vi_length"]].describe()


# In[8]:


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(train_df["en_length"], bins=30, kde=True, color="blue")
plt.title("Phân bố độ dài câu tiếng Anh")
plt.xlabel("Số từ")
plt.subplot(1, 2, 2)
sns.histplot(train_df["vi_length"], bins=30, kde=True, color="green")
plt.title("Phân bố độ dài câu tiếng Việt")
plt.xlabel("Số từ")
plt.tight_layout()
plt.show()


# In[9]:


# Lưu dữ liệu đã xử lý
train_df.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val_processed.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)
print(f"Đã lưu dữ liệu đã xử lý vào: {output_dir}")


# # TOKENIZER BPE

# In[10]:


# Chuẩn bị dữ liệu cho BPE
tokenizer_dir = os.path.join(output_dir, "bpe_tokenizer")
os.makedirs(tokenizer_dir, exist_ok=True)
all_texts = pd.concat([train_df["en"], train_df["vi"]]).dropna().tolist()
with open(os.path.join(output_dir, "bpe_texts.txt"), "w", encoding="utf-8") as f:
    for text in all_texts:
        f.write(text + "\n")
print(f"Đã lưu văn bản cho BPE: {len(all_texts)} câu")


# In[11]:


# Huấn luyện BPE
bpe_tokenizer = ByteLevelBPETokenizer()
bpe_tokenizer.train(
    files=[os.path.join(output_dir, "bpe_texts.txt")],
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
)
bpe_tokenizer.save_model(tokenizer_dir)
print(f"Đã lưu tokenizer BPE vào: {tokenizer_dir}")
print(f"Kích thước từ vựng: {bpe_tokenizer.get_vocab_size()}")


# In[12]:


# Kiểm tra tokenizer
sample_sentences = train_df["en"].head(2).tolist() + train_df["vi"].head(2).tolist()
for sentence in sample_sentences:
    encoding = bpe_tokenizer.encode(sentence)
    decoded = bpe_tokenizer.decode(encoding.ids, skip_special_tokens=True)
    print(f"\nCâu gốc: {sentence}")
    print(f"Tokens: {encoding.tokens[:10]}...")
    print(f"Giải mã: {decoded}")


# In[13]:


# Trực quan hóa độ dài token
token_lengths = [len(bpe_tokenizer.encode(text).ids) for text in all_texts[:1000]]
plt.figure(figsize=(8, 5))
sns.histplot(token_lengths, bins=30, kde=True, color="purple")
plt.title("Phân bố độ dài token BPE")
plt.xlabel("Số token")
plt.show()


# # MÔ HÌNH MARIANMT

# In[14]:


# Tải mô hình và tokenizer
marian_model_name = "Helsinki-NLP/opus-mt-en-vi"
marian_tokenizer = MarianTokenizer.from_pretrained(marian_model_name)
marian_model = MarianMTModel.from_pretrained(marian_model_name).to(device)
print(f"Đã tải MarianMT: {marian_model_name}")


# In[15]:


# Dataset và DataLoader
class MarianDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=60):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        en = str(self.df.iloc[idx]["en"])
        vi = str(self.df.iloc[idx]["vi"])
        inputs = self.tokenizer(
            en,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            vi,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels_ids = labels.input_ids.squeeze()
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": labels_ids,
            "en_text": en,
            "vi_text": vi,
        }


marian_train_dataset = MarianDataset(train_df, marian_tokenizer)
marian_val_dataset = MarianDataset(val_df, marian_tokenizer)
marian_test_dataset = MarianDataset(test_df, marian_tokenizer)

marian_train_loader = DataLoader(marian_train_dataset, batch_size=16, shuffle=True)
marian_val_loader = DataLoader(marian_val_dataset, batch_size=16)
marian_test_loader = DataLoader(marian_test_dataset, batch_size=16)

# Kiểm tra mẫu dữ liệu
sample_batch = next(iter(marian_train_loader))
print(f"Input IDs shape: {sample_batch['input_ids'].shape}")
print(f"Attention Mask shape: {sample_batch['attention_mask'].shape}")
print(f"Labels shape: {sample_batch['labels'].shape}")
print(f"Mẫu câu: EN='{sample_batch['en_text'][0]}', VI='{sample_batch['vi_text'][0]}'")


# In[16]:


# Huấn luyện
optimizer = torch.optim.AdamW(marian_model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=500, num_training_steps=len(marian_train_loader) * 5
)
scaler = GradScaler("cuda")

train_losses, val_losses, val_bleus, train_times = [], [], [], []
best_bleu = -float("inf")
betst_loss = float("inf")
patience = 5
patience_counter = 0

for epoch in range(EPOCH_NUM):
    start_time = time.time()
    marian_model.train()
    train_loss = 0
    for batch in tqdm(
        marian_train_loader, desc=f"MarianMT Epoch {epoch+1}/{EPOCH_NUM}"
    ):
        optimizer.zero_grad()
        with autocast("cuda"):
            outputs = marian_model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(marian_train_loader))

    marian_model.eval()
    val_loss = 0
    preds, refs = [], []
    with torch.no_grad():
        for batch in tqdm(marian_val_loader, desc="Validation"):
            outputs = marian_model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            val_loss += outputs.loss.item()
            generated_ids = marian_model.generate(
                batch["input_ids"].to(device),
                max_length=60,
                num_beams=7,
                early_stopping=True,
                top_k=50,
            )
            preds.extend(
                marian_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            )
            refs.extend([reference_dict[en] for en in batch["en_text"]])
    val_losses.append(val_loss / len(marian_val_loader))
    bleu = BLEU().corpus_score(preds, refs).score
    val_bleus.append(bleu)

    epoch_time = time.time() - start_time
    train_times.append(epoch_time)

    print(
        f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, Val BLEU={bleu:.2f}, Time={epoch_time:.2f}s"
    )

    # Early stopping (BLEU và loss)
    # if bleu > best_bleu:
    #     best_bleu = bleu
    #     patience_counter = 0
    #     marian_model.save_pretrained(os.path.join(output_dir, "marian_best"))
    #     marian_tokenizer.save_pretrained(os.path.join(output_dir, "marian_best"))
    # else:
    #     patience_counter += 1
    #     if patience_counter >= patience:
    #         print(f"Early stopping triggered after epoch {epoch+1}")
    #         break
    if bleu > best_bleu:
        best_bleu = bleu
        bleu_counter = 0
    else:
        bleu_counter += 1

    if val_loss < betst_loss:
        betst_loss = val_loss
        loss_counter = 0
    else:
        loss_counter += 1

    # Dừng khi cả BLEU và loss không cải thiện
    if bleu_counter >= patience and loss_counter >= patience:
        print(f"Early stopping triggered after epoch {epoch+1}")
        break

# Lưu metrics
model_metrics["MarianMT"]["train_losses"] = train_losses
model_metrics["MarianMT"]["val_losses"] = val_losses
model_metrics["MarianMT"]["val_bleus"] = val_bleus
model_metrics["MarianMT"]["train_times"] = train_times


# In[17]:


# Đánh giá
marian_model.eval()
test_preds, test_refs = [], []
with torch.no_grad():
    for batch in tqdm(marian_test_loader, desc="MarianMT Testing"):
        generated_ids = marian_model.generate(
            batch["input_ids"].to(device),
            max_length=60,
            num_beams=7,
            early_stopping=True,
            top_k=50,
        )
        test_preds.extend(
            marian_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        )
        test_refs.extend([reference_dict[en] for en in batch["en_text"]])

marian_bleu = BLEU().corpus_score(test_preds, test_refs).score
rouge_scorer_obj = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
)
marian_rouges = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
for pred, refs in zip(test_preds, test_refs):
    scores = [rouge_scorer_obj.score(ref, pred) for ref in refs]
    best_scores = max(scores, key=lambda x: x["rougeL"].fmeasure)
    for key in marian_rouges:
        marian_rouges[key] += best_scores[key].fmeasure
marian_rouges = {k: v / len(test_preds) for k, v in marian_rouges.items()}

print(f"MarianMT Test BLEU: {marian_bleu:.2f}")
print(
    f"MarianMT Test ROUGE: R1={marian_rouges['rouge1']:.4f}, R2={marian_rouges['rouge2']:.4f}, RL={marian_rouges['rougeL']:.4f}"
)


# In[18]:


# Trực quan hóa
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("MarianMT Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(val_bleus, label="Val BLEU", color="blue")
plt.title("MarianMT BLEU")
plt.legend()
plt.tight_layout()
plt.show()


# In[19]:


# Biểu đồ thời gian huấn luyện
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(train_times) + 1), train_times, label="Training Time", color="orange"
)
plt.title("MarianMT Training Time per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Time (seconds)")
plt.legend()
plt.show()


# In[20]:


for i in range(5):
    if i < len(test_preds):
        print(f"EN: {test_df.iloc[i]['en']}")
        print(f"VI (ref): {test_refs[i]}")
        print(f"VI (pred): {test_preds[i]}\n")

# Lưu mô hình cuối cùng
marian_model.save_pretrained(os.path.join(output_dir, "marian_finetuned"))
marian_tokenizer.save_pretrained(os.path.join(output_dir, "marian_finetuned"))


# # MÔ HÌNH TRANSFORMER TỪ ĐẦU

# In[21]:


# Dataset
class ScratchTransformerDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=60):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        en = str(self.df.iloc[idx]["en"])
        vi = str(self.df.iloc[idx]["vi"])
        src = (
            [self.cls_id]
            + self.tokenizer.encode(en).ids[: self.max_len - 2]
            + [self.sep_id]
        )
        tgt = (
            [self.cls_id]
            + self.tokenizer.encode(vi).ids[: self.max_len - 2]
            + [self.sep_id]
        )
        src = src + [self.pad_id] * (self.max_len - len(src))
        tgt = tgt + [self.pad_id] * (self.max_len - len(tgt))
        return {
            "input_ids": torch.tensor(src, dtype=torch.long),
            "decoder_input_ids": torch.tensor(tgt[:-1], dtype=torch.long),
            "labels": torch.tensor(tgt[1:], dtype=torch.long),
            "en_text": en,
            "vi_text": vi,
        }


scratch_train_dataset = ScratchTransformerDataset(train_df, bpe_tokenizer)
scratch_val_dataset = ScratchTransformerDataset(val_df, bpe_tokenizer)
scratch_test_dataset = ScratchTransformerDataset(test_df, bpe_tokenizer)

scratch_train_loader = DataLoader(scratch_train_dataset, batch_size=32, shuffle=True)
scratch_val_loader = DataLoader(scratch_val_dataset, batch_size=32)
scratch_test_loader = DataLoader(scratch_test_dataset, batch_size=32)

# Kiểm tra mẫu dữ liệu
sample_batch = next(iter(scratch_train_loader))
print(f"Input IDs shape: {sample_batch['input_ids'].shape}")
print(f"Decoder Input IDs shape: {sample_batch['decoder_input_ids'].shape}")
print(f"Labels shape: {sample_batch['labels'].shape}")
print(f"Mẫu câu: EN='{sample_batch['en_text'][0]}', VI='{sample_batch['vi_text'][0]}'")


# In[22]:


# Mô hình
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class ScratchTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_layers,
            num_layers,
            d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt):
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(
            device
        )
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc(output)

    def generate(self, src, max_len, start_id, end_id):
        batch_size = src.size(0)
        generated = torch.full(
            (batch_size, 1), start_id, dtype=torch.long, device=device
        )
        for _ in range(max_len - 1):
            output = self.forward(src, generated)
            next_token = output[:, -1].argmax(dim=-1).unsqueeze(1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == end_id).all():
                break
        return generated


scratch_model = ScratchTransformer(bpe_tokenizer.get_vocab_size()).to(device)


# In[23]:


# Huấn luyện
optimizer = torch.optim.AdamW(
    scratch_model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01
)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=4000, num_training_steps=len(scratch_train_loader) * 20
)
criterion = nn.CrossEntropyLoss(
    ignore_index=bpe_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1
)
scaler = GradScaler("cuda")

train_losses, val_losses, val_bleus, train_times = [], [], [], []
best_bleu = -float("inf")
patience = 7
patience_counter = 0

for epoch in range(EPOCH_NUM):
    start_time = time.time()
    scratch_model.train()
    train_loss = 0
    for batch in tqdm(
        scratch_train_loader, desc=f"Scratch Transformer Epoch {epoch+1}/{EPOCH_NUM}"
    ):
        optimizer.zero_grad()
        with autocast("cuda"):
            output = scratch_model(
                batch["input_ids"].to(device), batch["decoder_input_ids"].to(device)
            )
            loss = criterion(
                output.view(-1, output.size(-1)), batch["labels"].to(device).view(-1)
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(scratch_train_loader))

    scratch_model.eval()
    val_loss = 0
    preds, refs = [], []
    with torch.no_grad():
        for batch in tqdm(scratch_val_loader, desc="Validation"):
            output = scratch_model(
                batch["input_ids"].to(device), batch["decoder_input_ids"].to(device)
            )
            loss = criterion(
                output.view(-1, output.size(-1)), batch["labels"].to(device).view(-1)
            )
            val_loss += loss.item()
            generated_ids = scratch_model.generate(
                batch["input_ids"].to(device),
                max_len=60,
                start_id=bpe_tokenizer.token_to_id("[CLS]"),
                end_id=bpe_tokenizer.token_to_id("[SEP]"),
            )
            preds.extend(
                [
                    bpe_tokenizer.decode(ids.cpu().tolist(), skip_special_tokens=True)
                    for ids in generated_ids
                ]
            )
            refs.extend([reference_dict[en] for en in batch["en_text"]])
    val_losses.append(val_loss / len(scratch_val_loader))
    bleu = BLEU().corpus_score(preds, refs).score
    val_bleus.append(bleu)

    epoch_time = time.time() - start_time
    train_times.append(epoch_time)

    print(
        f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, Val BLEU={bleu:.2f}, Time={epoch_time:.2f}s"
    )

    # Early stopping
    if bleu > best_bleu:
        best_bleu = bleu
        patience_counter = 0
        torch.save(
            scratch_model.state_dict(),
            os.path.join(output_dir, "scratch_transformer_best.pt"),
        )
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

# Lưu metrics
model_metrics["Scratch Transformer"]["train_losses"] = train_losses
model_metrics["Scratch Transformer"]["val_losses"] = val_losses
model_metrics["Scratch Transformer"]["val_bleus"] = val_bleus
model_metrics["Scratch Transformer"]["train_times"] = train_times


# In[24]:


# Đánh giá
scratch_model.eval()
test_preds, test_refs = [], []
with torch.no_grad():
    for batch in tqdm(scratch_test_loader, desc="Scratch Transformer Testing"):
        generated_ids = scratch_model.generate(
            batch["input_ids"].to(device),
            max_len=60,
            start_id=bpe_tokenizer.token_to_id("[CLS]"),
            end_id=bpe_tokenizer.token_to_id("[SEP]"),
        )
        test_preds.extend(
            [
                bpe_tokenizer.decode(ids.cpu().tolist(), skip_special_tokens=True)
                for ids in generated_ids
            ]
        )
        test_refs.extend([reference_dict[en] for en in batch["en_text"]])

scratch_bleu = BLEU().corpus_score(test_preds, test_refs).score
scratch_rouges = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
for pred, refs in zip(test_preds, test_refs):
    scores = [rouge_scorer_obj.score(ref, pred) for ref in refs]
    best_scores = max(scores, key=lambda x: x["rougeL"].fmeasure)
    for key in scratch_rouges:
        scratch_rouges[key] += best_scores[key].fmeasure
scratch_rouges = {k: v / len(test_preds) for k, v in scratch_rouges.items()}

print(f"\nScratch Transformer Test BLEU: {scratch_bleu:.2f}")
print(
    f"Scratch Transformer Test ROUGE: R1={scratch_rouges['rouge1']:.4f}, R2={scratch_rouges['rouge2']:.4f}, RL={scratch_rouges['rougeL']:.4f}"
)


# In[25]:


# Trực quan hóa
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Scratch Transformer Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(val_bleus, label="Val BLEU", color="blue")
plt.title("Scratch Transformer BLEU")
plt.legend()
plt.tight_layout()
plt.show()


# In[26]:


# Biểu đồ thời gian huấn luyện
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(train_times) + 1), train_times, label="Training Time", color="orange"
)
plt.title("Scratch Transformer Training Time per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Time (seconds)")
plt.legend()
plt.show()


# In[27]:


for i in range(5):
    if i < len(test_preds):
        print(f"EN: {test_df.iloc[i]['en']}")
        print(f"VI (ref): {test_refs[i]}")
        print(f"VI (pred): {test_preds[i]}\n")

# Lưu mô hình cuối cùng
torch.save(
    scratch_model.state_dict(), os.path.join(output_dir, "scratch_transformer.pt")
)


# # MÔ HÌNH GPT-2

# In[28]:


# Tải mô hình và tokenizer
gpt_model_name = "gpt2"
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name).to(device)
gpt_model.config.pad_token_id = gpt_tokenizer.pad_token_id
print(f"Đã tải GPT-2: {gpt_model_name}")


# In[29]:


# Dataset
class GPT2Dataset(Dataset):
    def __init__(self, df, tokenizer, max_len=60):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        en = str(self.df.iloc[idx]["en"])
        vi = str(self.df.iloc[idx]["vi"])
        prompt = f"English: {en} Vietnamese: {vi}{self.tokenizer.eos_token}"
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": inputs.input_ids.squeeze(),
            "en_text": en,
            "vi_text": vi,
        }


gpt_train_dataset = GPT2Dataset(train_df, gpt_tokenizer)
gpt_val_dataset = GPT2Dataset(val_df, gpt_tokenizer)
gpt_test_dataset = GPT2Dataset(test_df, gpt_tokenizer)

gpt_train_loader = DataLoader(gpt_train_dataset, batch_size=8, shuffle=True)
gpt_val_loader = DataLoader(gpt_val_dataset, batch_size=8)
gpt_test_loader = DataLoader(gpt_test_dataset, batch_size=8)

# Kiểm tra mẫu dữ liệu
sample_batch = next(iter(gpt_train_loader))
print(f"Input IDs shape: {sample_batch['input_ids'].shape}")
print(f"Attention Mask shape: {sample_batch['attention_mask'].shape}")
print(f"Labels shape: {sample_batch['labels'].shape}")
print(f"Mẫu câu: EN='{sample_batch['en_text'][0]}', VI='{sample_batch['vi_text'][0]}'")


# In[30]:


# Huấn luyện
optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=500, num_training_steps=len(gpt_train_loader) * 5
)
scaler = GradScaler("cuda")

train_losses, val_losses, val_bleus, train_times = [], [], [], []
best_bleu = -float("inf")
patience = 5
patience_counter = 0

for epoch in range(EPOCH_NUM):
    start_time = time.time()
    gpt_model.train()
    train_loss = 0
    for batch in tqdm(gpt_train_loader, desc=f"GPT-2 Epoch {epoch+1}/{EPOCH_NUM}"):
        optimizer.zero_grad()
        with autocast("cuda"):
            outputs = gpt_model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(gpt_train_loader))

    gpt_model.eval()
    val_loss = 0
    preds, refs = [], []
    with torch.no_grad():
        for batch in tqdm(gpt_val_loader, desc="Validation"):
            outputs = gpt_model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            val_loss += outputs.loss.item()
            for en in batch["en_text"]:
                prompt = f"English: {en} Vietnamese:"
                inputs = gpt_tokenizer(
                    prompt, return_tensors="pt", max_length=50, truncation=True
                )
                attention_mask = inputs["attention_mask"].to(device)
                generated_ids = gpt_model.generate(
                    inputs["input_ids"].to(device),
                    attention_mask=attention_mask,
                    max_length=60,
                    num_beams=7,
                    early_stopping=True,
                    pad_token_id=gpt_tokenizer.pad_token_id,
                    top_k=50,
                )
                pred = gpt_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                pred = pred.split("Vietnamese:")[-1].strip()
                preds.append(pred)
            refs.extend([reference_dict[en] for en in batch["en_text"]])
    val_losses.append(val_loss / len(gpt_val_loader))
    bleu = BLEU().corpus_score(preds, refs).score
    val_bleus.append(bleu)

    epoch_time = time.time() - start_time
    train_times.append(epoch_time)

    print(
        f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, Val BLEU={bleu:.2f}, Time={epoch_time:.2f}s"
    )

    # Early stopping
    if bleu > best_bleu:
        best_bleu = bleu
        patience_counter = 0
        gpt_model.save_pretrained(os.path.join(output_dir, "gpt2_best"))
        gpt_tokenizer.save_pretrained(os.path.join(output_dir, "gpt2_best"))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

# Lưu metrics
model_metrics["GPT-2"]["train_losses"] = train_losses
model_metrics["GPT-2"]["val_losses"] = val_losses
model_metrics["GPT-2"]["val_bleus"] = val_bleus
model_metrics["GPT-2"]["train_times"] = train_times


# In[31]:


# Đánh giá
gpt_model.eval()
test_preds, test_refs = [], []
with torch.no_grad():
    for batch in tqdm(gpt_test_loader, desc="GPT-2 Testing"):
        for en in batch["en_text"]:
            prompt = f"English: {en} Vietnamese:"
            inputs = gpt_tokenizer(
                prompt, return_tensors="pt", max_length=50, truncation=True
            )
            attention_mask = inputs["attention_mask"].to(device)
            generated_ids = gpt_model.generate(
                inputs["input_ids"].to(device),
                attention_mask=attention_mask,
                max_length=60,
                num_beams=7,
                early_stopping=True,
                pad_token_id=gpt_tokenizer.pad_token_id,
                top_k=50,
            )
            pred = gpt_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            pred = pred.split("Vietnamese:")[-1].strip()
            test_preds.append(pred)
        test_refs.extend([reference_dict[en] for en in batch["en_text"]])

gpt_bleu = BLEU().corpus_score(test_preds, test_refs).score
gpt_rouges = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
for pred, refs in zip(test_preds, test_refs):
    scores = [rouge_scorer_obj.score(ref, pred) for ref in refs]
    best_scores = max(scores, key=lambda x: x["rougeL"].fmeasure)
    for key in gpt_rouges:
        gpt_rouges[key] += best_scores[key].fmeasure
gpt_rouges = {k: v / len(test_preds) for k, v in gpt_rouges.items()}

print(f"\nGPT-2 Test BLEU: {gpt_bleu:.2f}")
print(
    f"GPT-2 Test ROUGE: R1={gpt_rouges['rouge1']:.4f}, R2={gpt_rouges['rouge2']:.4f}, RL={gpt_rouges['rougeL']:.4f}"
)


# In[32]:


# Trực quan hóa
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("GPT-2 Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(val_bleus, label="Val BLEU", color="blue")
plt.title("GPT-2 BLEU")
plt.legend()
plt.tight_layout()
plt.show()


# In[33]:


# Biểu đồ thời gian huấn luyện
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(train_times) + 1), train_times, label="Training Time", color="orange"
)
plt.title("GPT-2 Training Time per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Time (seconds)")
plt.legend()
plt.show()


# In[34]:


for i in range(5):
    if i < len(test_preds):
        print(f"EN: {test_df.iloc[i]['en']}")
        print(f"VI (ref): {test_refs[i]}")
        print(f"VI (pred): {test_preds[i]}\n")

# Lưu mô hình cuối cùng
gpt_model.save_pretrained(os.path.join(output_dir, "gpt2_finetuned"))
gpt_tokenizer.save_pretrained(os.path.join(output_dir, "gpt2_finetuned"))


# # MÔ HÌNH GPT TỪ ĐẦU

# In[35]:


# Dataset
class ScratchGPTDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=60):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.cls_id = tokenizer.token_to_id("[CLS]")
        self.sep_id = tokenizer.token_to_id("[SEP]")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        en = str(self.df.iloc[idx]["en"])
        vi = str(self.df.iloc[idx]["vi"])
        src = self.tokenizer.encode(en).ids
        tgt = self.tokenizer.encode(vi).ids
        input_ids = [self.cls_id] + src + [self.sep_id] + tgt
        input_ids = input_ids[: self.max_len] + [self.pad_id] * (
            self.max_len - len(input_ids)
        )
        labels = src + [self.sep_id] + tgt + [self.sep_id]
        labels = labels[: self.max_len] + [self.pad_id] * (self.max_len - len(labels))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "en_text": en,
            "vi_text": vi,
        }


scratch_gpt_train_dataset = ScratchGPTDataset(train_df, bpe_tokenizer)
scratch_gpt_val_dataset = ScratchGPTDataset(val_df, bpe_tokenizer)
scratch_gpt_test_dataset = ScratchGPTDataset(test_df, bpe_tokenizer)

scratch_gpt_train_loader = DataLoader(
    scratch_gpt_train_dataset, batch_size=16, shuffle=True
)
scratch_gpt_val_loader = DataLoader(scratch_gpt_val_dataset, batch_size=16)
scratch_gpt_test_loader = DataLoader(scratch_gpt_test_dataset, batch_size=16)

# Kiểm tra mẫu dữ liệu
sample_batch = next(iter(scratch_gpt_train_loader))
print(f"Input IDs shape: {sample_batch['input_ids'].shape}")
print(f"Labels shape: {sample_batch['labels'].shape}")
print(f"Mẫu câu: EN='{sample_batch['en_text'][0]}', VI='{sample_batch['vi_text'][0]}'")


# In[36]:


# Mô hình
class ScratchGPT(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x):
        x = self.pos_encoder(self.embedding(x) * math.sqrt(self.d_model))
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(device)
        return self.fc(self.transformer(x, mask))

    def generate(self, prompt, max_len, end_id):
        generated = prompt
        for _ in range(max_len - prompt.size(1)):
            output = self.forward(generated)
            next_token = output[:, -1].argmax(dim=-1).unsqueeze(1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == end_id).all():
                break
        return generated


scratch_gpt_model = ScratchGPT(bpe_tokenizer.get_vocab_size()).to(device)


# In[37]:


# Huấn luyện
optimizer = torch.optim.AdamW(scratch_gpt_model.parameters(), lr=3e-4)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=4000,
    num_training_steps=len(scratch_gpt_train_loader) * 20,
)
criterion = nn.CrossEntropyLoss(
    ignore_index=bpe_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1
)
scaler = GradScaler("cuda")

train_losses, val_losses, val_bleus, train_times = [], [], [], []
best_bleu = -float("inf")
patience = 7
patience_counter = 0

for epoch in range(EPOCH_NUM):
    start_time = time.time()
    scratch_gpt_model.train()
    train_loss = 0
    for batch in tqdm(
        scratch_gpt_train_loader, desc=f"Scratch GPT Epoch {epoch+1}/{EPOCH_NUM}"
    ):
        optimizer.zero_grad()
        with autocast("cuda"):
            output = scratch_gpt_model(batch["input_ids"].to(device))
            loss = criterion(
                output.view(-1, output.size(-1)), batch["labels"].to(device).view(-1)
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(scratch_gpt_train_loader))

    scratch_gpt_model.eval()
    val_loss = 0
    preds, refs = [], []
    with torch.no_grad():
        for batch in tqdm(scratch_gpt_val_loader, desc="Validation"):
            output = scratch_gpt_model(batch["input_ids"].to(device))
            loss = criterion(
                output.view(-1, output.size(-1)), batch["labels"].to(device).view(-1)
            )
            val_loss += loss.item()
            for en in batch["en_text"]:
                prompt = torch.tensor(
                    [bpe_tokenizer.token_to_id("[CLS]")]
                    + bpe_tokenizer.encode(en).ids
                    + [bpe_tokenizer.token_to_id("[SEP]")],
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                generated_ids = scratch_gpt_model.generate(
                    prompt, max_len=60, end_id=bpe_tokenizer.token_to_id("[SEP]")
                )
                pred = bpe_tokenizer.decode(
                    generated_ids[0, prompt.size(1) :].cpu().tolist(),
                    skip_special_tokens=True,
                )
                preds.append(pred)
            refs.extend([reference_dict[en] for en in batch["en_text"]])
    val_losses.append(val_loss / len(scratch_gpt_val_loader))
    bleu = BLEU().corpus_score(preds, refs).score
    val_bleus.append(bleu)

    epoch_time = time.time() - start_time
    train_times.append(epoch_time)

    print(
        f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, Val BLEU={bleu:.2f}, Time={epoch_time:.2f}s"
    )

    # Early stopping
    if bleu > best_bleu:
        best_bleu = bleu
        patience_counter = 0
        torch.save(
            scratch_gpt_model.state_dict(),
            os.path.join(output_dir, "scratch_gpt_best.pt"),
        )
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break

# Lưu metrics
model_metrics["Scratch GPT"]["train_losses"] = train_losses
model_metrics["Scratch GPT"]["val_losses"] = val_losses
model_metrics["Scratch GPT"]["val_bleus"] = val_bleus
model_metrics["Scratch GPT"]["train_times"] = train_times


# In[38]:


# Đánh giá
scratch_gpt_model.eval()
test_preds, test_refs = [], []
with torch.no_grad():
    for batch in tqdm(scratch_gpt_test_loader, desc="Scratch GPT Testing"):
        for en in batch["en_text"]:
            prompt = torch.tensor(
                [bpe_tokenizer.token_to_id("[CLS]")]
                + bpe_tokenizer.encode(en).ids
                + [bpe_tokenizer.token_to_id("[SEP]")],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            generated_ids = scratch_gpt_model.generate(
                prompt, max_len=60, end_id=bpe_tokenizer.token_to_id("[SEP]")
            )
            pred = bpe_tokenizer.decode(
                generated_ids[0, prompt.size(1) :].cpu().tolist(),
                skip_special_tokens=True,
            )
            test_preds.append(pred)
        test_refs.extend([reference_dict[en] for en in batch["en_text"]])

scratch_gpt_bleu = BLEU().corpus_score(test_preds, test_refs).score
scratch_gpt_rouges = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
for pred, refs in zip(test_preds, test_refs):
    scores = [rouge_scorer_obj.score(ref, pred) for ref in refs]
    best_scores = max(scores, key=lambda x: x["rougeL"].fmeasure)
    for key in scratch_gpt_rouges:
        scratch_gpt_rouges[key] += best_scores[key].fmeasure
scratch_gpt_rouges = {k: v / len(test_preds) for k, v in scratch_gpt_rouges.items()}

print(f"\nScratch GPT Test BLEU: {scratch_gpt_bleu:.2f}")
print(
    f"Scratch GPT Test ROUGE: R1={scratch_gpt_rouges['rouge1']:.4f}, R2={scratch_gpt_rouges['rouge2']:.4f}, RL={scratch_gpt_rouges['rougeL']:.4f}"
)


# In[39]:


# Trực quan hóa
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Scratch GPT Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(val_bleus, label="Val BLEU", color="blue")
plt.title("Scratch GPT BLEU")
plt.legend()
plt.tight_layout()
plt.show()


# In[40]:


# Biểu đồ thời gian huấn luyện
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(train_times) + 1), train_times, label="Training Time", color="orange"
)
plt.title("Scratch GPT Training Time per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Time (seconds)")
plt.legend()
plt.show()


# In[41]:


for i in range(5):
    if i < len(test_preds):
        print(f"EN: {test_df.iloc[i]['en']}")
        print(f"VI (ref): {test_refs[i]}")
        print(f"VI (pred): {test_preds[i]}\n")

# Lưu mô hình cuối cùng
torch.save(scratch_gpt_model.state_dict(), os.path.join(output_dir, "scratch_gpt.pt"))


# # SO SÁNH MÔ HÌNH

# In[42]:


# Tổng hợp kết quả
avg_train_times = [
    (
        np.mean(model_metrics["MarianMT"]["train_times"])
        if model_metrics["MarianMT"]["train_times"]
        else 0
    ),
    (
        np.mean(model_metrics["Scratch Transformer"]["train_times"])
        if model_metrics["Scratch Transformer"]["train_times"]
        else 0
    ),
    (
        np.mean(model_metrics["GPT-2"]["train_times"])
        if model_metrics["GPT-2"]["train_times"]
        else 0
    ),
    (
        np.mean(model_metrics["Scratch GPT"]["train_times"])
        if model_metrics["Scratch GPT"]["train_times"]
        else 0
    ),
]

results = {
    "Model": ["MarianMT", "Scratch Transformer", "GPT-2", "Scratch GPT"],
    "BLEU": [marian_bleu, scratch_bleu, gpt_bleu, scratch_gpt_bleu],
    "ROUGE-1": [
        marian_rouges["rouge1"],
        scratch_rouges["rouge1"],
        gpt_rouges["rouge1"],
        scratch_gpt_rouges["rouge1"],
    ],
    "ROUGE-2": [
        marian_rouges["rouge2"],
        scratch_rouges["rouge2"],
        gpt_rouges["rouge2"],
        scratch_gpt_rouges["rouge2"],
    ],
    "ROUGE-L": [
        marian_rouges["rougeL"],
        scratch_rouges["rougeL"],
        gpt_rouges["rougeL"],
        scratch_gpt_rouges["rougeL"],
    ],
    "Avg Train Time (s)": [round(t, 2) for t in avg_train_times],
}
results_df = pd.DataFrame(results)
print("\nBảng so sánh:")
results_df


# In[43]:


# Trực quan hóa so sánh hiệu suất
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x="Model", y="BLEU", data=results_df)
plt.title("So sánh BLEU")
plt.subplot(1, 2, 2)
sns.barplot(x="Model", y="ROUGE-L", data=results_df)
plt.title("So sánh ROUGE-L")
plt.tight_layout()
plt.show()


# In[44]:


# Trực quan hóa quá trình huấn luyện
plt.figure(figsize=(15, 10))

# Training Loss
plt.subplot(2, 2, 1)
for model_name in model_metrics:
    epochs = range(1, len(model_metrics[model_name]["train_losses"]) + 1)
    plt.plot(epochs, model_metrics[model_name]["train_losses"], label=model_name)
plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Validation Loss
plt.subplot(2, 2, 2)
for model_name in model_metrics:
    epochs = range(1, len(model_metrics[model_name]["val_losses"]) + 1)
    plt.plot(epochs, model_metrics[model_name]["val_losses"], label=model_name)
plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Validation BLEU
plt.subplot(2, 2, 3)
for model_name in model_metrics:
    epochs = range(1, len(model_metrics[model_name]["val_bleus"]) + 1)
    plt.plot(epochs, model_metrics[model_name]["val_bleus"], label=model_name)
plt.title("Validation BLEU Comparison")
plt.xlabel("Epoch")
plt.ylabel("BLEU Score")
plt.legend()
plt.grid(True)

# Training Time
plt.subplot(2, 2, 4)
for model_name in model_metrics:
    epochs = range(1, len(model_metrics[model_name]["train_times"]) + 1)
    plt.plot(epochs, model_metrics[model_name]["train_times"], label=model_name)
plt.title("Training Time per Epoch Comparison")
plt.xlabel("Epoch")
plt.ylabel("Time (seconds)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
