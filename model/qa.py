"""QA head + pretrained-encoder loading helpers used by finetune / evaluate."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .model import BERT, DeBERTa


class QAModel(nn.Module):
    """Pre-trained encoder + a linear(hidden -> 2) split into start/end logits."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.qa_outputs = nn.Linear(encoder.hidden, 2)
        nn.init.xavier_uniform_(self.qa_outputs.weight)
        nn.init.zeros_(self.qa_outputs.bias)

    def forward(self, input_ids, segment_ids):
        hidden = self.encoder(input_ids, segment_ids)  # [B, S, H]
        logits = self.qa_outputs(hidden)               # [B, S, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


def detect_architecture(state: Dict[str, torch.Tensor],
                        config: Optional[Dict]) -> str:
    """Infer 'bert' or 'deberta' from either the saved config or the keys."""
    if config and "max_relative_positions" in config:
        return "deberta"
    for k in state.keys():
        if "relative_pos_emb" in k or "enhanced_mask_decoder" in k:
            return "deberta"
    return "bert"


def build_encoder(model_type: str, vocab_size: int, encoder_config: Dict) -> nn.Module:
    """Instantiate a fresh encoder from a saved config dict."""
    hidden = encoder_config.get("hidden", 768)
    n_layers = encoder_config.get("layers", encoder_config.get("n_layers", 12))
    attn_heads = encoder_config.get("attn_heads", 12)
    dropout = encoder_config.get("dropout", 0.1)

    if model_type == "bert":
        return BERT(vocab_size=vocab_size, hidden=hidden,
                    n_layers=n_layers, attn_heads=attn_heads, dropout=dropout)
    if model_type == "deberta":
        max_rel = encoder_config.get("max_relative_positions", 512)
        return DeBERTa(vocab_size=vocab_size, hidden=hidden,
                       n_layers=n_layers, attn_heads=attn_heads, dropout=dropout,
                       max_relative_positions=max_rel)
    raise ValueError(f"Unknown model_type: {model_type!r}")


def load_pretrained_encoder(checkpoint: dict, vocab_size: int,
                            override_model_type: Optional[str] = None,
                            logger=None) -> Tuple[nn.Module, str]:
    """Rebuild the pre-trained encoder from a pretraining checkpoint.

    Strips the ``bert.`` / ``deberta.`` prefix added by BERTLM / DeBERTaLM so
    what's left is exactly the encoder sub-module's state dict.
    """
    state = checkpoint["model_state"]
    config = checkpoint.get("config", {}) or {}
    model_type = override_model_type or detect_architecture(state, config)

    encoder = build_encoder(model_type, vocab_size, config)

    # DeBERTaLM registers the encoder under both self.bert and self.deberta,
    # but PyTorch state_dict() deduplication keeps only the first-seen path
    # ("bert."), so we must try both prefixes and use whichever matches.
    candidates = ["deberta.", "bert."] if model_type == "deberta" else ["bert."]
    encoder_state = {}
    for prefix in candidates:
        encoder_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if encoder_state:
            break
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)

    if logger is not None:
        logger.info(
            "Rebuilt %s encoder from pretraining ckpt: hidden=%d, layers=%d, heads=%d. "
            "Loaded %d tensors (missing=%d, unexpected=%d).",
            model_type, encoder.hidden, encoder.n_layers, encoder.attn_heads,
            len(encoder_state), len(missing), len(unexpected),
        )
        if missing:
            logger.warning("Missing keys on encoder load: %s", missing[:8])
        if unexpected:
            logger.warning("Unexpected keys on encoder load: %s", unexpected[:8])

    return encoder, model_type


def load_qa_model_from_checkpoint(checkpoint: dict, vocab_size: int,
                                  logger=None) -> Tuple[QAModel, str]:
    """Rebuild a QAModel (encoder + QA head) from a finetuning checkpoint.

    The saved ``config`` must include ``model_type`` and encoder hyperparams.
    """
    config = checkpoint.get("config", {}) or {}
    model_type = config.get("model_type")
    if model_type is None:
        model_type = detect_architecture(checkpoint["model_state"], config)

    encoder = build_encoder(model_type, vocab_size, config)
    model = QAModel(encoder)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if logger is not None:
        logger.info(
            "Loaded QA %s model: hidden=%d, layers=%d, heads=%d "
            "(missing=%d, unexpected=%d).",
            model_type, encoder.hidden, encoder.n_layers, encoder.attn_heads,
            len(missing), len(unexpected),
        )
        if missing:
            logger.warning("Missing keys on QA load: %s", missing[:8])
        if unexpected:
            logger.warning("Unexpected keys on QA load: %s", unexpected[:8])
    return model, model_type
