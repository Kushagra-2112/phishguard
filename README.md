# 🛡️ PhishGuard

**A dual-encoder fusion architecture for multimodal phishing detection**, combining a transformer-based email-text encoder (ModernBERT) with a character-level URL encoder (CANINE), joined by a learned gating mechanism.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](YOUR_STREAMLIT_URL_HERE)
[![Landing Page](https://img.shields.io/badge/landing%20page-visit-blue)](YOUR_NETLIFY_URL_HERE)
[![Model](https://img.shields.io/badge/🤗%20model-Hugging%20Face-yellow)](https://huggingface.co/Kushagrasharma2112/phishguard-checkpoint)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## Overview

Phishing attacks rarely fail in only one place. A message can be written convincingly while the embedded link points to fraudulent infrastructure — or a well-obfuscated link can be delivered with generic, low-effort text. Most existing detectors specialize in a single modality (either the email body *or* the URL) and are blind to whichever one they weren't trained on.

**PhishGuard** encodes both signals independently and combines them through a **gated fusion layer** that learns, per message, how much to trust each modality — including handling the common case where only one modality is available at all.

```
Email Body → ModernBERT ─┐
                          ├─→ Gated Fusion → Classifier → Phishing / Legitimate
URL        → CANINE     ─┘
```

## Results

Trained on a combined corpus of **288,665** examples (CEAS phishing-email dataset + a large-scale malicious-URL dataset + Tranco top-domains augmentation), evaluated on a held-out test split:

| Metric | Score |
|---|---|
| Accuracy | **97.52%** |
| F1-score | **96.79%** |
| Precision | 97.40% |
| Recall | 96.19% |

See [`docs/`](docs/) or the project thesis for the full training methodology, dataset construction details, and a discussion of diagnosed failure modes (root-domain misclassification fix, trailing-slash normalization bug, calibration drift analysis).

## Architecture

- **Text branch:** [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) encodes the email subject + body, capturing tone, urgency, and social-engineering phrasing.
- **URL branch:** [CANINE-s](https://huggingface.co/google/canine-s) encodes the URL character-by-character — chosen specifically because typosquatting, homoglyphs, and punycode tricks operate below the granularity a subword tokenizer preserves.
- **Fusion:** a small gated network takes both branch embeddings plus explicit presence flags and produces a softmax-weighted combination. When a modality is absent, a **learned "absent" embedding** is substituted rather than a zero vector.
- **Output:** a binary verdict (Phishing / Legitimate) along with the exact fusion gate weights, so the reasoning behind a prediction is inspectable rather than a black box.

## Project Structure

```
phishguard/
├── app.py                     # Streamlit inference UI
├── configs/
│   └── default.yaml           # All hyperparameters, paths, model config
├── data/
│   ├── raw/                   # Source datasets (not tracked in git)
│   └── processed/             # Combined, cleaned training data
├── src/
│   ├── config.py               # Config loader
│   ├── data/
│   │   ├── url_utils.py        # URL extraction/normalization
│   │   ├── dataset.py          # PyTorch Dataset with modality dropout
│   │   └── preprocess.py       # Merges raw sources into combined.csv
│   └── models/
│       └── fusion_model.py     # ModernBERT + CANINE + gated fusion
├── scripts/
│   ├── build_datasets.py       # Builds dataset1_email.csv / dataset2_url.csv
│   ├── train.py                # Training loop with early stopping
│   ├── predict.py              # Inference (CLI + used by app.py)
│   └── upload_checkpoint.py    # Pushes trained checkpoint to Hugging Face Hub
├── checkpoints/                # Trained model weights (not tracked in git)
└── requirements.txt
```

## Getting Started

### 1. Clone and install
```bash
git clone https://github.com/Kushagra-2112/phishguard.git
cd phishguard
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the app
The trained checkpoint auto-downloads from Hugging Face Hub on first run — no manual setup needed.
```bash
streamlit run app.py
```

### 3. Or use the CLI directly
```bash
python -m scripts.predict --text "Your account has been suspended..." --url "http://paypa1-secure-verify.tk/login"
```

### 4. Train from scratch (optional)
```bash
python -m scripts.build_datasets      # requires raw datasets in data/raw/ — see below
python -m src.data.preprocess
python -m scripts.train --limit 100000
```

**Raw datasets** (place in `data/raw/`, not included in this repo due to size):
- [CEAS_08 Phishing Email Dataset](https://huggingface.co/datasets/naserabdullahalam/phishing-email-dataset)
- [Malicious URLs Dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)
- [Tranco Top Sites List](https://tranco-list.eu)

## Tech Stack

| Component | Technology |
|---|---|
| Text encoder | ModernBERT-base |
| URL encoder | CANINE-s |
| Training framework | PyTorch, Hugging Face Transformers |
| Training compute | Kaggle Notebooks (NVIDIA T4 GPU) |
| Checkpoint storage | Hugging Face Hub |
| Frontend | Streamlit |
| Landing page | Static HTML/CSS/JS, deployed on Netlify |

## Known Limitations

- Trained on a 100,000-row stratified subsample due to compute constraints, not the full 288K-row corpus.
- Reduced sensitivity to single-character brand-impersonation typosquats lacking other structural red flags (e.g. IP host, unusual TLD).
- Some confidence-calibration drift observed in later training epochs (validation loss rises while F1 plateaus) — see the thesis discussion for details.

These are documented in depth, along with proposed fixes, in the project report.

## Live Demo

- **App:** [https://catchphishing.netlify.app/](https://catchphishing.netlify.app/)
- **Model checkpoint:** [huggingface.co/Kushagrasharma2112/phishguard-checkpoint](https://huggingface.co/Kushagrasharma2112/phishguard-checkpoint)

## Disclaimer

PhishGuard is a research/educational project and should not be relied upon as a sole safeguard against phishing. Always verify suspicious emails through official channels.

## License

MIT — see [LICENSE](LICENSE) for details.
