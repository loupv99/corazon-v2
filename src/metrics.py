"""Métricas de segmentación: Dice por estructura, promedio por imagen (cada placa pesa lo mismo)."""
from __future__ import annotations

import numpy as np

NOMBRES = ("pulmon_der", "pulmon_izq", "corazon")


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else 2.0 * float((pred & gt).sum()) / float(s)


class DiceAccumulator:
    def __init__(self):
        self.vals = {n: [] for n in NOMBRES}

    def update(self, pred: np.ndarray, gt: np.ndarray):
        """pred, gt: (3, H, W) binarias."""
        for k, n in enumerate(NOMBRES):
            self.vals[n].append(dice(pred[k].astype(bool), gt[k].astype(bool)))

    def summary(self) -> dict:
        out = {f"dice_{n}": float(np.mean(v)) for n, v in self.vals.items() if v}
        out["dice_medio"] = float(np.mean(list(out.values()))) if out else float("nan")
        out["n"] = len(self.vals[NOMBRES[0]])
        return out
