from __future__ import annotations
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PhishDataset(Dataset):
    def __init__(self, df: pd.DataFrame, text_tok, url_tok, cfg, training: bool = False):
        self.df = df.reset_index(drop=True)
        self.text_tok = text_tok
        self.url_tok = url_tok
        self.text_max_len = cfg.model.text_max_len
        self.url_max_len = cfg.model.url_max_len
        self.modality_dropout = cfg.data.modality_dropout if training else 0.0
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["text"] if row["has_text"] else ""
        url = row["url"] if row["has_url"] else ""
        has_text, has_url = bool(row["has_text"]), bool(row["has_url"])

        # Modality dropout: only makes sense when BOTH are present for this
        # row (the ~10.4% of paired examples) -- randomly blank one so the
        # fusion gate doesn't learn to always lean on whichever is present.
        if self.training and has_text and has_url:
            if random.random() < self.modality_dropout:
                if random.random() < 0.5:
                    text, has_text = "", False
                else:
                    url, has_url = "", False

        text_enc = self.text_tok(
            text or " ", truncation=True, padding="max_length",
            max_length=self.text_max_len, return_tensors="pt",
        )
        url_enc = self.url_tok(
            url or " ", truncation=True, padding="max_length",
            max_length=self.url_max_len, return_tensors="pt",
        )
        return {
            "text_input_ids": text_enc["input_ids"].squeeze(0),
            "text_attention_mask": text_enc["attention_mask"].squeeze(0),
            "url_input_ids": url_enc["input_ids"].squeeze(0),
            "url_attention_mask": url_enc["attention_mask"].squeeze(0),
            "has_text": torch.tensor(float(has_text)),
            "has_url": torch.tensor(float(has_url)),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def build_tokenizers(cfg):
    text_tok = AutoTokenizer.from_pretrained(cfg.model.text_model_name)
    url_tok = AutoTokenizer.from_pretrained(cfg.model.url_model_name)
    return text_tok, url_tok