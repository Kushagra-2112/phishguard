"""
Train PhishGuardModel on data/processed/combined.csv.

Run from the project root:
    python -m scripts.train              # full run using configs/default.yaml
    python -m scripts.train --smoke      # quick 600-row sanity check
    python -m scripts.train --limit 3000 # cap total rows used (train+val+test)
"""

from __future__ import annotations
import time
from pathlib import Path
import argparse

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import load_config, resolve, set_seed, get_device
from src.data.dataset import PhishDataset, build_tokenizers
from src.models.fusion_model import PhishGuardModel


def load_splits(cfg, smoke: bool = False, limit: int | None = None):
    path = resolve(cfg.paths.processed_dir) / "combined.csv"
    df = pd.read_csv(path)
    df["text"] = df["text"].fillna("")
    df["url"] = df["url"].fillna("")

    if smoke:
        df = df.sample(n=min(600, len(df)), random_state=cfg.seed).reset_index(drop=True)
    elif limit:
        df = df.sample(n=min(limit, len(df)), random_state=cfg.seed).reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df, test_size=cfg.data.val_size + cfg.data.test_size,
        random_state=cfg.seed, stratify=df["label"],
    )
    rel_test = cfg.data.test_size / (cfg.data.val_size + cfg.data.test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, random_state=cfg.seed, stratify=temp_df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = move_batch(batch, device)
        logits, _ = model(
            text_input_ids=batch["text_input_ids"],
            text_attention_mask=batch["text_attention_mask"],
            url_input_ids=batch["url_input_ids"],
            url_attention_mask=batch["url_attention_mask"],
            has_text=batch["has_text"],
            has_url=batch["has_url"],
        )
        loss = loss_fn(logits, batch["label"])
        total_loss += loss.item() * len(batch["label"])
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["label"].cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision, recall, _, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    avg_loss = total_loss / len(all_labels)
    return {"loss": avg_loss, "accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Quick sanity run on a tiny subset.")
    parser.add_argument("--limit", type=int, default=None, help="Cap total rows used for this run.")
    args = parser.parse_args()

    cfg = load_config()
    if args.smoke:
        cfg["train"]["epochs"] = 1
        cfg["paths"]["run_name"] = "smoke-test"
        print("=== SMOKE TEST MODE: 600 rows, 1 epoch ===")

    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_df, val_df, test_df = load_splits(cfg, smoke=args.smoke, limit=args.limit)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    text_tok, url_tok = build_tokenizers(cfg)

    train_ds = PhishDataset(train_df, text_tok, url_tok, cfg, training=True)
    val_ds = PhishDataset(val_df, text_tok, url_tok, cfg, training=False)
    test_ds = PhishDataset(test_df, text_tok, url_tok, cfg, training=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.eval_batch_size, shuffle=False,
        num_workers=cfg.train.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.train.eval_batch_size, shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    model = PhishGuardModel(cfg).to(device)

    # Separate learning rates: encoders (pretrained, fine-tune slowly)
    # vs the fusion/classifier head (randomly initialized, learn faster).
    encoder_params = list(model.text_branch.encoder.parameters()) + list(model.url_branch.encoder.parameters())
    head_params = (
        list(model.text_branch.proj.parameters()) + [model.text_branch.absent_embedding]
        + list(model.url_branch.proj.parameters()) + [model.url_branch.absent_embedding]
        + list(model.fusion.parameters()) + list(model.classifier.parameters())
    )
    optimizer = AdamW([
        {"params": [p for p in encoder_params if p.requires_grad], "lr": cfg.train.encoder_lr},
        {"params": head_params, "lr": cfg.train.head_lr},
    ], weight_decay=cfg.train.weight_decay)

    total_steps = len(train_loader) * cfg.train.epochs
    warmup_steps = int(total_steps * cfg.train.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = nn.CrossEntropyLoss()

    ckpt_dir = resolve(cfg.paths.checkpoint_dir) / cfg.paths.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    patience_left = cfg.train.early_stopping_patience

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.train.epochs}")
        for step, batch in enumerate(pbar, start=1):
            batch = move_batch(batch, device)
            optimizer.zero_grad()
            logits, _ = model(
                text_input_ids=batch["text_input_ids"],
                text_attention_mask=batch["text_attention_mask"],
                url_input_ids=batch["url_input_ids"],
                url_attention_mask=batch["url_attention_mask"],
                has_text=batch["has_text"],
                has_url=batch["has_url"],
            )
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss / step:.4f}")

        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch} done in {elapsed/60:.1f}min | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_prec={val_metrics['precision']:.4f} "
            f"val_rec={val_metrics['recall']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_left = cfg.train.early_stopping_patience
            torch.save(
                {"model_state": model.state_dict(), "cfg": dict(cfg), "epoch": epoch, "val_f1": best_f1},
                ckpt_dir / "best.pt",
            )
            print(f"  -> New best model saved (val_f1={best_f1:.4f})")
        else:
            patience_left -= 1
            print(f"  -> No improvement. Patience left: {patience_left}")
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    print("\n=== Final test set evaluation (best checkpoint) ===")
    best = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    test_metrics = evaluate(model, test_loader, device)
    print(test_metrics)


if __name__ == "__main__":
    main()