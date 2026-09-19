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
import hashlib
import re
from langdetect import detect, LangDetectException

from src.config import load_config, resolve
from src.data.url_utils import extract_urls, pick_primary_url, clean_email_text

def _normalize_for_dedup(text: str) -> str:
    """Collapse whitespace/case and strip common variable tokens (numbers,
    emails) so near-identical spam templates hash the same even when the
    recipient name or a tracking ID differs between copies."""
    t = text.lower()
    t = re.sub(r"\S+@\S+", " ", t)          # strip email addresses
    t = re.sub(r"\d+", " ", t)              # strip numbers/IDs
    t = re.sub(r"\s+", " ", t).strip()
    return t


def dedup_near_duplicates(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Drop rows whose normalized text hashes identically to an earlier row.
    Cheap and effective against templated spam campaigns; won't catch every
    paraphrase, but removes the common case of bulk-identical sends."""
    before = len(df)
    hashes = df[text_col].fillna("").apply(
        lambda t: hashlib.md5(_normalize_for_dedup(t).encode()).hexdigest()
    )
    df = df.loc[~hashes.duplicated()].copy()
    print(f"  near-duplicate removal: {before} -> {len(df)} rows ({before - len(df)} removed)")
    return df


def filter_english(df: pd.DataFrame, text_col: str = "text", min_chars: int = 20) -> pd.DataFrame:
    """Keep only rows whose body text is detected as English. Very short
    text is left as-is (language detection is unreliable below ~20 chars,
    and short/empty bodies are common+legitimate in this dataset)."""
    before = len(df)

    def is_english_or_short(t: str) -> bool:
        t = str(t)
        if len(t.strip()) < min_chars:
            return True
        try:
            return detect(t) == "en"
        except LangDetectException:
            return True  # ambiguous/undetectable -> keep rather than drop

    mask = df[text_col].fillna("").apply(is_english_or_short)
    df = df.loc[mask].copy()
    print(f"  language filter: {before} -> {len(df)} rows ({before - len(df)} removed)")
    return df

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

    print("Cleaning email dataset:")
    out = dedup_near_duplicates(out, text_col="text")
    out = filter_english(out, text_col="text")

    out = out.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = out_dir / "dataset1_email.csv"
    out.to_csv(out_path, index=False)
    print(f"[dataset1_email] wrote {len(out)} rows -> {out_path}")
    print(out["label"].value_counts())
    print(f"  has_url=True: {out['has_url'].mean():.1%}  (real URLs extracted from body text)")
    return out


def build_url_dataset(raw_dir: Path, out_dir: Path, max_benign: int = 120_000, n_tranco: int = 40_000):
    path = raw_dir / "malicious_phish.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    assert "url" in df.columns and "type" in df.columns, f"Unexpected columns: {df.columns.tolist()}"

    df["type"] = df["type"].str.strip().str.lower()
    phishing = df[df["type"] == "phishing"].copy()
    benign = df[df["type"] == "benign"].copy()

    # The malicious-urls dataset's benign class is almost entirely deep
    # links with paths (wikipedia articles, blog posts, etc.) and rarely
    # includes bare root domains. Without correction, the model never sees
    # "https://google.com"-style URLs labeled benign, and associates
    # short/clean URLs with phishing instead. Tranco (a ranked list of the
    # world's most popular domains) patches this gap directly.
    tranco_path = raw_dir / "tranco.csv"
    tranco_rows = []
    if tranco_path.exists():
        tranco = pd.read_csv(tranco_path, header=None, names=["rank", "domain"])
        tranco = tranco.head(n_tranco)
        tranco_urls = "https://" + tranco["domain"].astype(str)
        tranco_rows = pd.DataFrame({"url": tranco_urls, "label": 0})
        print(f"[dataset2_url] adding {len(tranco_rows)} Tranco root-domain benign examples")
    else:
        print("[dataset2_url] WARNING: tranco.csv not found, skipping root-domain augmentation")

    if len(benign) > max_benign:
        benign = benign.sample(n=max_benign, random_state=42)

    phishing["label"] = 1
    benign["label"] = 0

    parts = [phishing[["url", "label"]], benign[["url", "label"]]]
    if len(tranco_rows):
        parts.append(tranco_rows[["url", "label"]])

    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=["url"])
    out = out.drop_duplicates(subset=["url"])
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