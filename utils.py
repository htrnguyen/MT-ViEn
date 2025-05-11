# utils.py
import os
import time

import matplotlib.pyplot as plt
import torch


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        time.sleep(1)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved to {path}")


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f"[INFO] Model loaded from {path}")
    return model


def plot_training_history(history, title, plot_path=None):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if plot_path:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path)
        print(f"[INFO] Plot saved to {plot_path}")
    plt.show()
