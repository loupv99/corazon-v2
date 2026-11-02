"""Pérdida de segmentación multietiqueta (pulmón der., pulmón izq., corazón).

Adaptada del parcial (BCE + Dice). Cambia una cosa: aquí hay 3 canales independientes, así que el Dice se calcula por
canal y se promedia. Si se calculara sobre los 3 juntos, los pulmones (más grandes) dominarían al corazón, que es la
estructura que más importa para el ICT.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits, y, eps: float = 1.0):
    p = torch.sigmoid(logits)
    dims = (0, 2, 3)                                  # suma sobre batch y píxeles, separado por canal
    inter = (p * y).sum(dims)
    denom = p.sum(dims) + y.sum(dims)
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


def combo_loss(logits, y):
    """BCE da gradientes estables píxel a píxel; Dice optimiza directamente el solapamiento."""
    return F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
