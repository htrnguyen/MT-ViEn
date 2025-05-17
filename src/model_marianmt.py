import os
import time

import numpy as np
import torch
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from transformers import MarianMTModel, MarianTokenizer, get_linear_schedule_with_warmup


class MarianDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=80):
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


def get_dataloaders(train_df, val_df, test_df, tokenizer, max_len=80, batch_size=16):
    train_set = MarianDataset(train_df, tokenizer, max_len)
    val_set = MarianDataset(val_df, tokenizer, max_len)
    test_set = MarianDataset(test_df, tokenizer, max_len)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size),
        DataLoader(test_set, batch_size=batch_size),
    )


def train_marianmt(
    model,
    tokenizer,
    train_loader,
    val_loader,
    reference_dict,
    device,
    output_dir,
    EPOCHS=30,
    lr=3e-5,
    patience=6,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=300, num_training_steps=len(train_loader) * EPOCHS
    )
    scaler = GradScaler(device_type="cuda")
    train_losses, val_losses, val_bleus, train_times = [], [], [], []
    best_bleu, best_loss = -float("inf"), float("inf")
    bleu_counter = loss_counter = 0

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = model(
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
        train_losses.append(train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        preds, refs = [], []
        with autocast(device_type="cuda"):
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                val_loss += outputs.loss.item()
                generated_ids = model.generate(
                    batch["input_ids"].to(device),
                    max_length=80,
                    num_beams=5,
                    early_stopping=True,
                )
                preds.extend(
                    tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                )
                refs.extend([reference_dict[en] for en in batch["en_text"]])
        val_losses.append(val_loss / len(val_loader))
        bleu = BLEU().corpus_score(preds, refs).score
        val_bleus.append(bleu)
        train_times.append(time.time() - start_time)
        print(
            f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}, BLEU={bleu:.2f}"
        )

        # Early stopping và save model tốt nhất
        improved = False
        if bleu > best_bleu:
            best_bleu = bleu
            bleu_counter = 0
            improved = True
        else:
            bleu_counter += 1
        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            loss_counter = 0
            improved = True
        else:
            loss_counter += 1
        if improved:
            model.save_pretrained(os.path.join(output_dir, "marian_best"))
            tokenizer.save_pretrained(os.path.join(output_dir, "marian_best"))
        if bleu_counter >= patience and loss_counter >= patience:
            print("Early stopping triggered!")
            break
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_bleus": val_bleus,
        "train_times": train_times,
        "best_bleu": best_bleu,
        "best_loss": best_loss,
    }
    return metrics


def evaluate_marianmt(model, tokenizer, test_loader, reference_dict, device):
    model.eval()
    preds, refs = [], []
    with autocast(device_type="cuda"):
        for batch in test_loader:
            generated_ids = model.generate(
                batch["input_ids"].to(device),
                max_length=80,
                num_beams=5,
                early_stopping=True,
            )
            preds.extend(
                tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            )
            refs.extend([reference_dict[en] for en in batch["en_text"]])
    bleu = BLEU().corpus_score(preds, refs).score
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouges = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    for pred, tgt_list in zip(preds, refs):
        scores = [scorer.score(ref, pred) for ref in tgt_list]
        best = max(scores, key=lambda x: x["rougeL"].fmeasure)
        for key in rouges:
            rouges[key] += best[key].fmeasure
    rouges = {k: v / len(preds) for k, v in rouges.items()}
    return bleu, rouges, preds


def load_marian_best(model_dir, device):
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir).to(device)
    return model, tokenizer


def sample_inference(model, tokenizer, input_texts, device, max_len=80):
    model.eval()
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    with autocast(device_type="cuda"):
        generated_ids = model.generate(
            inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_length=max_len,
            num_beams=5,
            early_stopping=True,
        )
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return outputs


# ---- Ví dụ sử dụng (nên gọi từ main.ipynb) ----
if __name__ == "__main__":
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = pd.read_csv("../output/train.csv")
    val_df = pd.read_csv("../output/val.csv")
    test_df = pd.read_csv("../output/test.csv")
    ref_dict = (
        pd.concat([train_df, val_df, test_df]).groupby("en")["vi"].apply(list).to_dict()
    )

    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-vi").to(device)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_df, val_df, test_df, tokenizer
    )

    metrics = train_marianmt(
        model,
        tokenizer,
        train_loader,
        val_loader,
        ref_dict,
        device,
        "../output",
        EPOCHS=10,
    )
    bleu, rouges, preds = evaluate_marianmt(
        model, tokenizer, test_loader, ref_dict, device
    )
    print(f"Test BLEU: {bleu:.2f}")
    print(f"Test ROUGE: {rouges}")
    # Test inference
    print(
        sample_inference(
            model, tokenizer, ["How are you?", "I want to go home."], device
        )
    )
