#!/usr/bin/env python
# coding: utf-8

"""
utils.py: Chứa các hàm hỗ trợ cho dự án dịch máy, bao gồm vẽ biểu đồ, tính toán metric,
và các tiện ích khác. Được tối ưu cho GPU P100 và hiển thị kết quả trực quan trong notebook.
"""

import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
from tqdm.auto import tqdm

# Thiết lập kiểu dáng cho biểu đồ
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Thiết lập thiết bị (mặc định GPU P100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_directories(output_dir):
    """
    Tạo các thư mục đầu ra cho mô hình và biểu đồ.

    Args:
        output_dir (str): Thư mục gốc để lưu kết quả.

    Returns:
        tuple: Đường dẫn tới thư mục mô hình và biểu đồ.
    """
    model_output_dir = os.path.join(output_dir, "models")
    plot_output_dir = os.path.join(output_dir, "plots")

    for dir_path in [output_dir, model_output_dir, plot_output_dir]:
        os.makedirs(dir_path, exist_ok=True)

    print(f"Đã thiết lập thư mục đầu ra:")
    print(f"- Thư mục gốc: {output_dir}")
    print(f"- Thư mục mô hình: {model_output_dir}")
    print(f"- Thư mục biểu đồ: {plot_output_dir}")

    return model_output_dir, plot_output_dir


def calculate_rouge(predictions, references):
    """
    Tính điểm ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) cho danh sách dự đoán và tham chiếu.

    Args:
        predictions (list): Danh sách câu dự đoán.
        references (list): Danh sách câu tham chiếu.

    Returns:
        dict: Điểm ROUGE trung bình cho ROUGE-1, ROUGE-2, ROUGE-L.
    """
    if not predictions or not references or len(predictions) != len(references):
        print("Cảnh báo: Không có dữ liệu hợp lệ để tính ROUGE.")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    rouge_calc = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    r1_sum, r2_sum, rl_sum, count = 0.0, 0.0, 0.0, 0

    for pred, ref in zip(predictions, references):
        pred_str = str(pred).strip()
        ref_str = str(ref).strip()
        if not pred_str or not ref_str:
            continue
        scores = rouge_calc.score(ref_str, pred_str)
        r1_sum += scores["rouge1"].fmeasure
        r2_sum += scores["rouge2"].fmeasure
        rl_sum += scores["rougeL"].fmeasure
        count += 1

    if count == 0:
        print("Cảnh báo: Không có cặp câu hợp lệ để tính ROUGE.")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    rouge_scores = {
        "rouge1": r1_sum / count,
        "rouge2": r2_sum / count,
        "rougeL": rl_sum / count,
    }

    print(f"Đã tính ROUGE cho {count} cặp câu:")
    print(f"- ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"- ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"- ROUGE-L: {rouge_scores['rougeL']:.4f}")

    return rouge_scores


def calculate_bleu(predictions, references):
    """
    Tính điểm BLEU cho danh sách dự đoán và tham chiếu.

    Args:
        predictions (list): Danh sách câu dự đoán.
        references (list): Danh sách câu tham chiếu.

    Returns:
        float: Điểm BLEU.
    """
    if not predictions or not references:
        print("Cảnh báo: Không có dữ liệu để tính BLEU.")
        return 0.0

    bleu_score = corpus_bleu(predictions, [references]).score
    print(f"Đã tính BLEU: {bleu_score:.2f}")
    return bleu_score


def plot_training_history(history, title_prefix, plot_output_dir):
    """
    Vẽ và lưu biểu đồ lịch sử huấn luyện (loss và metric).

    Args:
        history (dict): Lịch sử huấn luyện với train_loss, val_loss, val_bleu, val_rougeL.
        title_prefix (str): Tiêu đề biểu đồ.
        plot_output_dir (str): Thư mục lưu biểu đồ.
    """
    epochs_range = range(1, len(history.get("train_loss", [])) + 1)
    if not epochs_range:
        print(f"Biểu đồ {title_prefix}: Không có dữ liệu lịch sử để vẽ.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Biểu đồ Loss
    if "train_loss" in history and history["train_loss"]:
        ax1.plot(
            epochs_range, history["train_loss"], "o-", label="Train Loss", color="blue"
        )
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(epochs_range, history["val_loss"], "x-", label="Val Loss", color="red")
    ax1.set_title(f"{title_prefix} - Loss Huấn luyện và Xác thực")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Biểu đồ Metric
    if "val_bleu" in history and history["val_bleu"]:
        ax2.plot(
            epochs_range, history["val_bleu"], "o-", label="Val BLEU", color="blue"
        )
    if "val_rougeL" in history and history["val_rougeL"]:
        ax2.plot(
            epochs_range, history["val_rougeL"], "x-", label="Val ROUGE-L", color="red"
        )
    ax2.set_title(f"{title_prefix} - Metric Xác thực")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_filename = f"{title_prefix.replace(' ', '_').lower()}_training_history.png"
    plot_path = os.path.join(plot_output_dir, plot_filename)
    plt.savefig(plot_path)
    plt.show()  # Hiển thị biểu đồ trong notebook
    print(f"Lưu biểu đồ lịch sử huấn luyện tại: {plot_path}")
    plt.close(fig)


def plot_comparison_histories(all_histories, plot_output_dir):
    """
    Vẽ biểu đồ so sánh lịch sử huấn luyện của tất cả mô hình.

    Args:
        all_histories (dict): Từ điển chứa lịch sử huấn luyện của các mô hình.
        plot_output_dir (str): Thư mục lưu biểu đồ.
    """
    if not all_histories:
        print("Không có lịch sử huấn luyện để so sánh.")
        return

    num_models = len(all_histories)
    fig, axs = plt.subplots(num_models, 2, figsize=(15, 5 * num_models), squeeze=False)

    for idx, (model_key, history) in enumerate(all_histories.items()):
        epochs_range = range(1, len(history.get("train_loss", [])) + 1)
        if not epochs_range:
            continue

        # Biểu đồ Loss
        ax1 = axs[idx, 0]
        if "train_loss" in history and history["train_loss"]:
            ax1.plot(
                epochs_range,
                history["train_loss"],
                "o-",
                label="Train Loss",
                color="blue",
            )
        if "val_loss" in history and history["val_loss"]:
            ax1.plot(
                epochs_range, history["val_loss"], "x-", label="Val Loss", color="red"
            )
        ax1.set_title(f"{model_key} - Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # Biểu đồ Metric
        ax2 = axs[idx, 1]
        if "val_bleu" in history and history["val_bleu"]:
            ax2.plot(
                epochs_range, history["val_bleu"], "o-", label="Val BLEU", color="blue"
            )
        if "val_rougeL" in history and history["val_rougeL"]:
            ax2.plot(
                epochs_range,
                history["val_rougeL"],
                "x-",
                label="Val ROUGE-L",
                color="red",
            )
        ax2.set_title(f"{model_key} - Metric")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score")
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    comparison_plot_path = os.path.join(
        plot_output_dir, "models_comparison_history.png"
    )
    plt.savefig(comparison_plot_path)
    plt.show()  # Hiển thị biểu đồ trong notebook
    print(f"Lưu biểu đồ so sánh tại: {comparison_plot_path}")
    plt.close(fig)


def save_results_table(all_results, output_dir):
    """
    Lưu bảng so sánh kết quả của các mô hình dưới dạng CSV và Markdown.

    Args:
        all_results (dict): Từ điển chứa kết quả của các mô hình.
        output_dir (str): Thư mục lưu bảng.
    """
    if not all_results:
        print("Không có kết quả để lưu.")
        return

    results_df = pd.DataFrame(all_results).T.fillna("N/A")
    print("\nBảng so sánh hiệu suất trên tập kiểm tra:")
    print(results_df.to_markdown())

    results_csv_path = os.path.join(output_dir, "model_comparison_results.csv")
    results_md_path = os.path.join(output_dir, "model_comparison_results.md")
    results_df.to_csv(results_csv_path)
    with open(results_md_path, "w", encoding="utf-8") as f:
        f.write(results_df.to_markdown())

    print(f"Lưu bảng so sánh tại:")
    print(f"- CSV: {results_csv_path}")
    print(f"- Markdown: {results_md_path}")


def print_device_info():
    """
    In thông tin về thiết bị đang sử dụng.
    """
    print(f"Thiết bị: {device}")
    if torch.cuda.is_available():
        print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Bộ nhớ GPU khả dụng: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )


if __name__ == "__main__":
    # Kiểm tra thiết bị
    print_device_info()

    # Thử nghiệm tạo thư mục
    output_dir = "./output"
    model_dir, plot_dir = setup_directories(output_dir)

    # Dữ liệu mẫu để kiểm tra metric
    sample_preds = ["This is a test sentence.", "Another example here."]
    sample_refs = ["This is a test sentence.", "Another example."]
    print("\nKiểm tra tính toán metric với dữ liệu mẫu:")
    bleu = calculate_bleu(sample_preds, sample_refs)
    rouge = calculate_rouge(sample_preds, sample_refs)
