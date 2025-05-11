# train_utils.py
import os
import time

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def train_model(
    model,
    train_loader,
    val_loader,
    tokenizer,
    device,
    optimizer,
    scheduler,
    epochs,
    model_dir,
    direction,
):
    model.to(device)
    model.train()
    scaler = GradScaler()
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0
        model.train()

        with tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", unit="batch"
        ) as pbar:
            for batch in pbar:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                optimizer.zero_grad()

                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f}")
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                os.path.join(model_dir, f"best_model_{direction}.pt"),
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 3:
            print("Early stopping triggered.")
            break

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_duration:.2f}s")

    return model, history
