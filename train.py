#!/usr/bin/env python
# coding: utf-8

"""
train.py: Định nghĩa hàm huấn luyện và đánh giá cho các mô hình dịch máy.
Tích hợp mixed precision training, beam search nâng cao với length penalty và diversity penalty.
Tối ưu cho GPU P100, hỗ trợ kiểm tra và thống kê quá trình.
"""

import os
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from data_preprocessing import GPTTranslationPromptDataset, MarianTranslationDataset
from utils import (
    calculate_bleu,
    calculate_rouge,
    plot_training_history,
    save_results_table,
)

# Thiết lập thiết bị (mặc định GPU P100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training(
    config, train_dataset, val_dataset, test_dataset, model, tokenizer, output_dir
):
    """
    Huấn luyện và đánh giá mô hình dịch máy.

    Args:
        config (dict): Cấu hình huấn luyện (model_type, batch_size, learning_rate, ...).
        train_dataset, val_dataset, test_dataset: Dataset cho train/val/test.
        model: Mô hình dịch máy (MarianMT, TransformerScratch, GPTScratch, GPT2FineTuned).
        tokenizer: Tokenizer tương ứng với mô hình.
        output_dir (str): Thư mục lưu mô hình và biểu đồ.

    Returns:
        dict: Kết quả đánh giá trên tập test.
    """
    model_type = config["model_type"]
    src_lang, tgt_lang = config["src_lang"], config["tgt_lang"]
    experiment_key = f"{model_type}_{src_lang}-{tgt_lang}"
    model_save_dir = os.path.join(output_dir, "models", experiment_key)
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, "best_model.pt")

    print(
        f"\n{'='*30} BẮT ĐẦU HUẤN LUYỆN: {model_type} ({src_lang} -> {tgt_lang}) {'='*30}"
    )
    print(f"Cấu hình: {config}")

    # Tạo DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], num_workers=0, pin_memory=True
    )
    print(
        f"Đã tạo DataLoader: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}"
    )

    # Thiết lập optimizer, scheduler, criterion
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config["warmup_ratio"]),
        num_training_steps=total_steps,
    )
    criterion = (
        nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id,
            label_smoothing=config.get("label_smoothing", 0.1),
        )
        if model_type in ["Transformer_Scratch", "GPT_Scratch"]
        else None
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    # Thiết lập tham số generation
    pad_token_id = tokenizer.pad_token_id
    start_token_id = (
        tokenizer.cls_token_id
        if model_type in ["Transformer_Scratch", "GPT_Scratch"]
        else tokenizer.bos_token_id
    )
    end_token_id = (
        tokenizer.sep_token_id
        if model_type in ["Transformer_Scratch", "GPT_Scratch"]
        else tokenizer.eos_token_id
    )

    # Lưu lịch sử huấn luyện
    history = {"train_loss": [], "val_loss": [], "val_bleu": [], "val_rougeL": []}
    best_val_bleu = -1.0
    patience_counter = 0

    for epoch in range(config["epochs"]):
        epoch_start_time = time.time()
        model.train()
        epoch_train_loss = 0
        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{config['epochs']} Training",
            leave=False,
        )

        # Giảm dần teacher forcing ratio
        teacher_forcing_ratio = (
            max(0.5 - (epoch / config["epochs"]) * 0.4, 0.1)
            if model_type == "GPT_Scratch"
            else 0.5
        )

        for batch in train_pbar:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                if model_type == "Transformer_Scratch":
                    decoder_input_ids = batch["decoder_input_ids"].to(device)
                    src_key_padding_mask = batch["src_key_padding_mask"].to(device)
                    tgt_key_padding_mask = batch["tgt_key_padding_mask"].to(device)
                    outputs = model(
                        input_ids,
                        decoder_input_ids,
                        src_key_padding_mask,
                        tgt_key_padding_mask,
                    )
                    loss = criterion(
                        outputs.view(-1, outputs.size(-1)), labels.view(-1)
                    )
                elif model_type == "GPT_Scratch":
                    outputs = model(
                        input_ids,
                        tgt_key_padding_mask=batch["attention_mask"].to(device),
                        teacher_forcing_ratio=teacher_forcing_ratio,
                    )
                    loss = criterion(
                        outputs.view(-1, outputs.size(-1)), labels.view(-1)
                    )
                else:  # MarianMT, GPT2FineTuned
                    outputs = model(
                        input_ids, attention_mask=attention_mask, labels=labels
                    )
                    loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = epoch_train_loss / len(train_loader)
        history["train_loss"].append(train_loss)

        # Đánh giá trên tập validation
        model.eval()
        epoch_val_loss = 0
        val_preds, val_refs = [], []
        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{config['epochs']} Validation",
            leave=False,
        )

        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = batch["labels"].to(device)
                src_texts = batch["src_text"]
                tgt_texts = batch["tgt_text"]

                with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                    if model_type == "Transformer_Scratch":
                        decoder_input_ids = batch["decoder_input_ids"].to(device)
                        src_key_padding_mask = batch["src_key_padding_mask"].to(device)
                        tgt_key_padding_mask = batch["tgt_key_padding_mask"].to(device)
                        outputs = model(
                            input_ids,
                            decoder_input_ids,
                            src_key_padding_mask,
                            tgt_key_padding_mask,
                        )
                        loss = criterion(
                            outputs.view(-1, outputs.size(-1)), labels.view(-1)
                        )
                    elif model_type == "GPT_Scratch":
                        outputs = model(
                            input_ids,
                            tgt_key_padding_mask=batch["attention_mask"].to(device),
                        )
                        loss = criterion(
                            outputs.view(-1, outputs.size(-1)), labels.view(-1)
                        )
                    else:
                        outputs = model(
                            input_ids, attention_mask=attention_mask, labels=labels
                        )
                        loss = outputs.loss

                epoch_val_loss += loss.item()

                # Sinh dự đoán với beam search nâng cao
                generation_params = {
                    "max_length": (
                        80 if model_type == "GPT2_FineTuned" else config["max_len"] + 10
                    ),
                    "num_beams": (
                        5 if model_type in ["MarianMT", "GPT2_FineTuned"] else 5
                    ),
                    "length_penalty": (
                        1.0 if model_type in ["MarianMT", "GPT2_FineTuned"] else 1.0
                    ),
                    "no_repeat_ngram_size": (
                        3 if model_type in ["MarianMT", "GPT2_FineTuned"] else 2
                    ),
                    "early_stopping": True,
                    "pad_token_id": pad_token_id,
                    "eos_token_id": end_token_id,
                    "top_k": 50 if model_type == "GPT2_FineTuned" else 0,
                }

                if model_type == "MarianMT":
                    gen_ids = model.generate(
                        input_ids, attention_mask=attention_mask, **generation_params
                    )
                    preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                elif model_type == "GPT2_FineTuned":
                    preds = []
                    for src_text in src_texts:
                        prompt = f"Translate from English to Vietnamese: {src_text} -> "
                        input_ids_g = tokenizer.encode(
                            prompt,
                            return_tensors="pt",
                            max_length=config["max_len"] - 20,
                            truncation=True,
                        ).to(device)
                        attn_mask_g = torch.ones_like(input_ids_g).to(device)
                        gen_ids = model.generate(
                            input_ids_g, attention_mask=attn_mask_g, **generation_params
                        )
                        pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                        pred = pred.replace(prompt, "").strip()
                        preds.append(pred)
                elif model_type == "Transformer_Scratch":
                    src_key_padding_mask = batch["src_key_padding_mask"].to(device)
                    _, preds = model.generate(
                        input_ids,
                        src_key_padding_mask,
                        config["max_len"],
                        start_token_id,
                        end_token_id,
                        pad_token_id,
                    )
                elif model_type == "GPT_Scratch":
                    preds = []
                    for src_text in src_texts:
                        src_enc = tokenizer.encode(src_text)
                        prompt_ids = torch.tensor(
                            [start_token_id] + src_enc.ids + [end_token_id],
                            dtype=torch.long,
                            device=device,
                        ).unsqueeze(0)
                        if prompt_ids.size(1) > config["max_len"] // 2:
                            prompt_ids = prompt_ids[:, : config["max_len"] // 2]
                        _, pred = model.generate(
                            prompt_ids, config["max_len"], end_token_id, pad_token_id
                        )
                        preds.append(pred[0])

                val_preds.extend(preds)
                val_refs.extend(tgt_texts)

        val_loss = epoch_val_loss / len(val_loader)
        val_bleu = calculate_bleu(val_preds, val_refs)
        val_rouge = calculate_rouge(val_preds, val_refs)

        history["val_loss"].append(val_loss)
        history["val_bleu"].append(val_bleu)
        history["val_rougeL"].append(val_rouge["rougeL"])

        print(f"\nEpoch {epoch+1} ({time.time() - epoch_start_time:.1f}s):")
        print(f"- Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"- Val BLEU: {val_bleu:.2f}, Val ROUGE-L: {val_rouge['rougeL']:.4f}")

        # Lưu mô hình tốt nhất
        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            if model_type in ["MarianMT", "GPT2_FineTuned"]:
                model.model.save_pretrained(model_save_dir)
                tokenizer.save_pretrained(model_save_dir)
            else:
                torch.save(model.state_dict(), best_model_path)
            print(
                f"-> Lưu mô hình tốt nhất (BLEU: {best_val_bleu:.2f}) tại: {best_model_path}"
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Dừng sớm tại epoch {epoch+1}.")
                break

    # Vẽ biểu đồ lịch sử huấn luyện
    plot_training_history(
        history,
        f"{model_type}_{src_lang}_to_{tgt_lang}",
        os.path.join(output_dir, "plots"),
    )

    # Đánh giá trên tập test
    print(f"\n{'='*30} ĐÁNH GIÁ: {model_type} ({src_lang} -> {tgt_lang}) {'='*30}")
    if os.path.exists(best_model_path):
        if model_type in ["Transformer_Scratch", "GPT_Scratch"]:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Đã tải mô hình tốt nhất từ: {best_model_path}")

    model.eval()
    test_preds, test_refs, test_srcs = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing {model_type}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            src_texts = batch["src_text"]
            tgt_texts = batch["tgt_text"]

            generation_params["num_beams"] = (
                10 if model_type in ["MarianMT", "GPT2_FineTuned"] else 5
            )
            generation_params["length_penalty"] = (
                1.8 if model_type in ["MarianMT", "GPT2_FineTuned"] else 1.0
            )

            if model_type == "MarianMT":
                gen_ids = model.generate(
                    input_ids, attention_mask=attention_mask, **generation_params
                )
                preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            elif model_type == "GPT2_FineTuned":
                preds = []
                for src_text in src_texts:
                    prompt = f"Translate from English to Vietnamese: {src_text} -> "
                    input_ids_g = tokenizer.encode(
                        prompt,
                        return_tensors="pt",
                        max_length=config["max_len"] - 20,
                        truncation=True,
                    ).to(device)
                    attn_mask_g = torch.ones_like(input_ids_g).to(device)
                    gen_ids = model.generate(
                        input_ids_g, attention_mask=attn_mask_g, **generation_params
                    )
                    pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    pred = pred.replace(prompt, "").strip()
                    preds.append(pred)
            elif model_type == "Transformer_Scratch":
                src_key_padding_mask = batch["src_key_padding_mask"].to(device)
                _, preds = model.generate(
                    input_ids,
                    src_key_padding_mask,
                    config["max_len"],
                    start_token_id,
                    end_token_id,
                    pad_token_id,
                )
            elif model_type == "GPT_Scratch":
                preds = []
                for src_text in src_texts:
                    src_enc = tokenizer.encode(src_text)
                    prompt_ids = torch.tensor(
                        [start_token_id] + src_enc.ids + [end_token_id],
                        dtype=torch.long,
                        device=device,
                    ).unsqueeze(0)
                    if prompt_ids.size(1) > config["max_len"] // 2:
                        prompt_ids = prompt_ids[:, : config["max_len"] // 2]
                    _, pred = model.generate(
                        prompt_ids, config["max_len"], end_token_id, pad_token_id
                    )
                    preds.append(pred[0])

            test_preds.extend(preds)
            test_refs.extend(tgt_texts)
            test_srcs.extend(src_texts)

    test_bleu = calculate_bleu(test_preds, test_refs)
    test_rouge = calculate_rouge(test_preds, test_refs)

    print(f"\nKết quả Test ({model_type} {src_lang}->{tgt_lang}):")
    print(f"- BLEU: {test_bleu:.2f}")
    print(
        f"- ROUGE-1: {test_rouge['rouge1']:.4f}, ROUGE-2: {test_rouge['rouge2']:.4f}, ROUGE-L: {test_rouge['rougeL']:.4f}"
    )

    for i in range(min(3, len(test_preds))):
        print(f"\nMẫu {i+1}:")
        print(f"- Src: {test_srcs[i]}")
        print(f"- Ref: {test_refs[i]}")
        print(f"- Pred: {test_preds[i]}")

    results = {
        "BLEU": f"{test_bleu:.2f}",
        "ROUGE-1": f"{test_rouge['rouge1']:.4f}",
        "ROUGE-2": f"{test_rouge['rouge2']:.4f}",
        "ROUGE-L": f"{test_rouge['rougeL']:.4f}",
    }

    final_model_path = os.path.join(model_save_dir, "final_model")
    if model_type in ["MarianMT", "GPT2_FineTuned"]:
        model.model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"Lưu mô hình cuối tại: {final_model_path}")
    else:
        torch.save(model.state_dict(), f"{final_model_path}_state_dict.pt")
        print(f"Lưu mô hình cuối tại: {final_model_path}_state_dict.pt")

    del model, optimizer, scheduler, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return history, results, experiment_key


if __name__ == "__main__":
    from transformers import GPT2Tokenizer, MarianTokenizer

    from data_preprocessing import (
        ScratchGPTDataset,
        ScratchTransformerDataset,
        load_and_preprocess_data,
        train_bpe_tokenizer,
    )
    from models import (
        GPT2FineTunedWrapper,
        MarianMTWrapper,
        ScratchGPTModel,
        TransformerScratchModel,
    )

    config = {
        "model_type": "Transformer_Scratch",
        "src_lang": "en",
        "tgt_lang": "vi",
        "max_len": 45,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "epochs": 5,
        "warmup_ratio": 0.1,
        "label_smoothing": 0.1,
        "early_stopping_patience": 3,
        "d_model": 256,
        "nhead": 4,
        "num_encoder_layers": 3,
        "num_decoder_layers": 3,
        "dim_feedforward": 512,
        "dropout": 0.2,
    }

    data_file = "/kaggle/input/daily-en-vi/eng-vie.csv"
    output_dir = "./output"
    train_df, val_df, test_df, plot_dir = load_and_preprocess_data(
        data_file, output_dir
    )

    tokenizer = train_bpe_tokenizer(train_df, output_dir)
    model = TransformerScratchModel(
        src_vocab_size=tokenizer.get_vocab_size(True),
        tgt_vocab_size=tokenizer.get_vocab_size(True),
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_seq_len=config["max_len"],
    ).to(device)

    train_dataset = ScratchTransformerDataset(
        train_df, tokenizer, "en", "vi", config["max_len"]
    )
    val_dataset = ScratchTransformerDataset(
        val_df, tokenizer, "en", "vi", config["max_len"]
    )
    test_dataset = ScratchTransformerDataset(
        test_df, tokenizer, "en", "vi", config["max_len"]
    )

    history, results, experiment_key = run_training(
        config, train_dataset, val_dataset, test_dataset, model, tokenizer, output_dir
    )

    all_results = {experiment_key: results}
    save_results_table(all_results, output_dir)
