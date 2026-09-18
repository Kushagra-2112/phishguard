from src.config import load_config
from src.models.fusion_model import PhishGuardModel
import torch

cfg = load_config()
model = PhishGuardModel(cfg)
B = 2
out = model(
    text_input_ids=torch.randint(0, 1000, (B, cfg.model.text_max_len)),
    text_attention_mask=torch.ones(B, cfg.model.text_max_len, dtype=torch.long),
    url_input_ids=torch.randint(0, 1000, (B, cfg.model.url_max_len)),
    url_attention_mask=torch.ones(B, cfg.model.url_max_len, dtype=torch.long),
    has_text=torch.tensor([1.0, 0.0]),
    has_url=torch.tensor([1.0, 1.0]),
)
logits, gate_weights = out
print("logits shape:", logits.shape)
print("gate_weights shape:", gate_weights.shape)
print("gate_weights sample:", gate_weights)