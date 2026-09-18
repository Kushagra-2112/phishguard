from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel


def _freeze_bottom_layers(encoder, n_layers: int):
    """Freeze embeddings + the first n_layers transformer blocks.
    Cuts training time and memory a lot on CPU, with minimal accuracy cost
    since bottom layers mostly learn generic token-level features anyway.

    Different model families (BERT-style vs CANINE) expose their internals
    differently, so this is defensive: it freezes what it can find and
    silently skips what it can't, rather than crashing.
    """
    if n_layers <= 0:
        return

    # Try to freeze an embeddings module if one exists at the top level.
    if hasattr(encoder, "embeddings"):
        for p in encoder.embeddings.parameters():
            p.requires_grad = False

    # Try to find a list of transformer layers under a few common names.
    layers = None
    for attr in ("encoder", "transformer"):
        module = getattr(encoder, attr, None)
        if module is not None and hasattr(module, "layer"):
            layers = module.layer
            break

    if layers is None:
        # CANINE and some other architectures don't expose a simple
        # `.encoder.layer` list -- skip layer freezing for these rather
        # than erroring out. Embeddings-level freezing above (if any)
        # still applies, and the rest of the model just trains fully.
        return

    for layer in layers[:n_layers]:
        for p in layer.parameters():
            p.requires_grad = False


class Branch(nn.Module):
    """One encoder -> pooled embedding -> projection to shared d_model.
    Maps to 'modernBERT'/'CANINE' + 'TEXT-EMBEDDING'/'URL-EMBEDDING' boxes."""
    def __init__(self, model_name: str, d_model: int, dropout: float, freeze_layers: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        _freeze_bottom_layers(self.encoder, freeze_layers)
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden, d_model), nn.GELU(), nn.Dropout(dropout),
        )
        # Learned vector substituted in when this modality is absent for a sample.
        self.absent_embedding = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.absent_embedding, std=0.02)

    def forward(self, input_ids, attention_mask, present_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]           # [CLS]-style pooled representation
        embed = self.proj(cls)
        present_mask = present_mask.unsqueeze(-1)    # (B,1)
        absent = self.absent_embedding.unsqueeze(0).expand_as(embed)
        return present_mask * embed + (1 - present_mask) * absent


class GatedFusion(nn.Module):
    """FUSION OF BOTH: a learned gate weighs text vs url embeddings,
    conditioned on which modalities are actually present for this sample."""
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2 + 2, d_model), nn.GELU(),
            nn.Linear(d_model, 2), nn.Softmax(dim=-1),
        )
        self.combine = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, text_emb, url_emb, has_text, has_url):
        flags = torch.stack([has_text, has_url], dim=-1)          # (B,2)
        gate_in = torch.cat([text_emb, url_emb, flags], dim=-1)
        weights = self.gate(gate_in)                              # (B,2) -- softmax'd
        weighted = torch.cat(
            [text_emb * weights[:, :1], url_emb * weights[:, 1:]], dim=-1
        )
        fused = self.combine(weighted)
        return fused, weights


class PhishGuardModel(nn.Module):
    """Dataset-1(Email)->ModernBERT ; Dataset-2(URL)->CANINE ;
    gated fusion ; classifier -> Phishing/Legitimate."""
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.text_branch = Branch(m.text_model_name, m.d_model, m.dropout, m.freeze_text_layers)
        self.url_branch = Branch(m.url_model_name, m.d_model, m.dropout, m.freeze_url_layers)
        self.fusion = GatedFusion(m.d_model, m.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(m.d_model, m.d_model // 2), nn.GELU(), nn.Dropout(m.dropout),
            nn.Linear(m.d_model // 2, m.num_labels),
        )

    def forward(self, text_input_ids, text_attention_mask, url_input_ids,
                url_attention_mask, has_text, has_url):
        text_emb = self.text_branch(text_input_ids, text_attention_mask, has_text)
        url_emb = self.url_branch(url_input_ids, url_attention_mask, has_url)
        fused, gate_weights = self.fusion(text_emb, url_emb, has_text, has_url)
        logits = self.classifier(fused)
        return logits, gate_weights