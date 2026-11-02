"""Dataset de segmentación sobre los arrays de prepare_data (memmap: no carga todo en RAM).

Aumentación (solo en train), cada decisión con su motivo:
  - rotación ±10°, escala 0,9–1,1 y traslación ±5 %: la placa nunca está perfectamente centrada;
  - brillo y contraste ±15 %: equipos y exposiciones distintos;
  - SIN volteo horizontal: el corazón está a la izquierda del paciente; voltear crearía anatomía falsa
    (dextrocardia) y confundiría los canales de pulmón derecho e izquierdo.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .prepare_data import unpack

MEAN, STD = 0.5, 0.25   # normalización fija de la radiografía en [0, 1]


class SegDataset(Dataset):
    """images/masks: arrays o rutas .npy. Con rutas, cada worker abre su propio memmap: pasar un memmap ya abierto a
    los workers del DataLoader lo copiaría entero a RAM al serializarlo."""

    def __init__(self, images, masks, idx, augment: bool = False, seed: int = 0):
        self._src = (images, masks)
        self.images = self.masks = None
        self.idx = np.asarray(idx)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def _abrir(self):
        if self.images is None:
            self.images, self.masks = (np.load(s, mmap_mode="r") if isinstance(s, str) else s for s in self._src)
            # cada worker recibe una semilla distinta (y reproducible) de PyTorch: sin esto, todos los workers
            # repetirían la misma secuencia de aumentaciones
            self.rng = np.random.default_rng(torch.initial_seed() % 2**32)

    def __len__(self):
        return len(self.idx)

    def _aug(self, img, msk):
        s = img.shape[0]
        ang = self.rng.uniform(-10, 10)
        esc = self.rng.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((s / 2, s / 2), ang, esc)
        M[:, 2] += self.rng.uniform(-0.05, 0.05, 2) * s
        img = cv2.warpAffine(img, M, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        msk = cv2.warpAffine(msk, M, (s, s), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        img = img.astype(np.float32) * self.rng.uniform(0.85, 1.15) + self.rng.uniform(-0.15, 0.15) * 255
        return np.clip(img, 0, 255), msk

    def __getitem__(self, i):
        self._abrir()
        k = self.idx[i]
        img, msk = np.asarray(self.images[k]), np.asarray(self.masks[k])
        if self.augment:
            img, msk = self._aug(img, msk)
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        y = unpack(msk).astype(np.float32)
        return torch.from_numpy(x)[None], torch.from_numpy(y)
