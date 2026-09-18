"""
Build dataset1_email.csv and dataset2_url.csv from raw sources:
  data/raw/ceas_phishing_email.csv   (sender, receiver, date, subject, body, label, urls[flag])
  data/raw/malicious_urls.csv        (url, type)

Run from the project root:
    python -m scripts.build_datasets
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.config import load_config, resolve
from src.data.url_utils import extract_urls, pick_primary_url, clean_email_text


def build_email_dataset(raw_dir: Path, out_dir: Path):
    path = raw_dir / "CEAS_08.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"body", "label"}
    assert required.issubset(df.columns), f"Missing columns, found: {df.columns.tolist()}"

    df["subject"] = df.get("subject", "").fillna("")
    df["body"] = df["body"].fillna("")

    raw_combined = (df["subject"].astype(str) + "\n\n" + df["body"].astype(str)).str.strip()

    df["text"] = raw_combined.apply(clean_email_text)
    df["url"] = raw_combined.apply(lambda b: pick_primary_url(extract_urls(b)))
    df["has_text"] = df["text"].str.strip().ne("")
    df["has_url"] = df["url"].str.strip().ne("")
    df["label"] = df["label"].astype(int)

    out = df[["text", "url", "has_text", "has_url", "label"]]
    out = out[out["has_text"] | out["has_url"]]
    out = out.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = out_dir / "dataset1_email.csv"
    out.to_csv(out_path, index=False)
    print(f"[dataset1_email] wrote {len(out)} rows -> {out_path}")
    print(out["label"].value_counts())
    print(f"  has_url=True: {out['has_url'].mean():.1%}  (real URLs extracted from body text)")
    return out


def build_url_dataset(raw_dir: Path, out_dir: Path, max_benign: int = 120_000):
    path = raw_dir / "malicious_phish.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    assert "url" in df.columns and "type" in df.columns, f"Unexpected columns: {df.columns.tolist()}"

    df["type"] = df["type"].str.strip().str.lower()
    phishing = df[df["type"] == "phishing"].copy()
    benign = df[df["type"] == "benign"].copy()

    if len(benign) > max_benign:
        benign = benign.sample(n=max_benign, random_state=42)

    phishing["label"] = 1
    benign["label"] = 0

    out = pd.concat([phishing[["url", "label"]], benign[["url", "label"]]], ignore_index=True)
    out = out.dropna(subset=["url"])
    out = out.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = out_dir / "dataset2_url.csv"
    out.to_csv(out_path, index=False)
    print(f"[dataset2_url] wrote {len(out)} rows -> {out_path}")
    print(out["label"].value_counts())
    return out


def main():
    cfg = load_config()
    raw_dir = resolve(cfg.paths.raw_dir)
    out_dir = raw_dir

    print("=== Building email dataset ===")
    build_email_dataset(raw_dir, out_dir)

    print("\n=== Building URL dataset ===")
    build_url_dataset(raw_dir, out_dir)


if __name__ == "__main__":
    main()