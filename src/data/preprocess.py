"""
Merge dataset1_email.csv (has text, sometimes has_url too) and
dataset2_url.csv (url-only) into a single combined.csv with columns:
    text, url, has_text, has_url, label

Run from the project root:
    python -m src.data.preprocess
"""
from __future__ import annotations
import pandas as pd
from src.config import load_config, resolve


def main():
    cfg = load_config()
    raw_dir = resolve(cfg.paths.raw_dir)
    out_dir = resolve(cfg.paths.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d1 = pd.read_csv(raw_dir / "dataset1_email.csv")
    d2 = pd.read_csv(raw_dir / "dataset2_url.csv")

    # dataset1_email.csv already has: text, url, has_text, has_url, label
    d1["text"] = d1["text"].fillna("")
    d1["url"] = d1["url"].fillna("")

    # dataset2_url.csv only has: url, label -- add the missing columns
    d2["text"] = ""
    d2["has_text"] = False
    d2["url"] = d2["url"].fillna("")
    d2["has_url"] = d2["url"].str.strip().ne("")

    combined = pd.concat(
        [d1[["text", "url", "has_text", "has_url", "label"]],
         d2[["text", "url", "has_text", "has_url", "label"]]],
        ignore_index=True,
    )
    combined = combined[combined["has_text"] | combined["has_url"]]
    combined = combined.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)

    out_path = out_dir / "combined.csv"
    combined.to_csv(out_path, index=False)

    print(f"Wrote {len(combined)} rows -> {out_path}")
    print(combined["label"].value_counts())
    print(f"has_text: {combined['has_text'].mean():.1%}   has_url: {combined['has_url'].mean():.1%}")
    both = (combined["has_text"] & combined["has_url"]).mean()
    print(f"both modalities present: {both:.1%}")


if __name__ == "__main__":
    main()