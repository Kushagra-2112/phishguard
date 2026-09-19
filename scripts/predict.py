"""
Load a trained PhishGuardModel checkpoint and run inference on a single
email body and/or URL.

Usage (from project root):
    python -m scripts.predict --text "your email body here" --url "http://example.com"
    python -m scripts.predict --text "your email body here"      # text only
    python -m scripts.predict --url "http://example.com"          # url only
"""
from __future__ import annotations
import argparse

import torch
import torch.nn.functional as F

from src.config import load_config, resolve, get_device
from src.data.dataset import build_tokenizers
from src.data.url_utils import clean_email_text, normalize_url, url_surface_features
from src.models.fusion_model import PhishGuardModel

_CACHE = {}


def load_model_and_tokenizers(checkpoint_path=None):
    """Loads once, caches in-process -- Streamlit will call this every rerun
    otherwise, and reloading two transformer models each time is slow."""
    if "model" in _CACHE:
        return _CACHE["model"], _CACHE["text_tok"], _CACHE["url_tok"], _CACHE["cfg"], _CACHE["device"]

    cfg = load_config()
    device = get_device()
    ckpt_path = resolve(checkpoint_path or cfg.inference.checkpoint)

    text_tok, url_tok = build_tokenizers(cfg)
    model = PhishGuardModel(cfg)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    _CACHE.update(model=model, text_tok=text_tok, url_tok=url_tok, cfg=cfg, device=device)
    return model, text_tok, url_tok, cfg, device


@torch.no_grad()
def predict(text: str = "", url: str = "", checkpoint_path=None):
    model, text_tok, url_tok, cfg, device = load_model_and_tokenizers(checkpoint_path)

    text = clean_email_text(text) if text else ""
    url = normalize_url(url) if url else ""
    has_text = bool(text.strip())
    has_url = bool(url.strip())

    if not has_text and not has_url:
        raise ValueError("Provide at least one of --text or --url.")

    text_enc = text_tok(
        text or " ", truncation=True, padding="max_length",
        max_length=cfg.model.text_max_len, return_tensors="pt",
    )
    url_enc = url_tok(
        url or " ", truncation=True, padding="max_length",
        max_length=cfg.model.url_max_len, return_tensors="pt",
    )

    logits, gate_weights = model(
        text_input_ids=text_enc["input_ids"].to(device),
        text_attention_mask=text_enc["attention_mask"].to(device),
        url_input_ids=url_enc["input_ids"].to(device),
        url_attention_mask=url_enc["attention_mask"].to(device),
        has_text=torch.tensor([float(has_text)]).to(device),
        has_url=torch.tensor([float(has_url)]).to(device),
    )
    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_label = int(probs.argmax())
    weights = gate_weights.cpu().numpy()[0]

    result = {
        "label": "Phishing" if pred_label == 1 else "Legitimate",
        "phishing_probability": float(probs[1]),
        "legitimate_probability": float(probs[0]),
        "gate_weight_text": float(weights[0]),
        "gate_weight_url": float(weights[1]),
        "has_text": has_text,
        "has_url": has_url,
    }
    if has_url:
        result["url_features"] = url_surface_features(url)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="", help="Email body text.")
    parser.add_argument("--url", type=str, default="", help="URL to check.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .pt checkpoint.")
    args = parser.parse_args()

    result = predict(text=args.text, url=args.url, checkpoint_path=args.checkpoint)

    print(f"\nVerdict: {result['label']}")
    print(f"  Phishing probability:   {result['phishing_probability']:.1%}")
    print(f"  Legitimate probability: {result['legitimate_probability']:.1%}")
    print(f"  Gate weights -> text: {result['gate_weight_text']:.2f}  url: {result['gate_weight_url']:.2f}")
    if "url_features" in result:
        print(f"  URL features: {result['url_features']}")


if __name__ == "__main__":
    main()