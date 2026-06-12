"""
repro.py
========
Utilidad de reproducibilidad. Fija TODAS las semillas (random, numpy, torch) y
fuerza rutas deterministas en PyTorch para que el pipeline completo
(entrenamiento -> evaluacion -> identificacion de espacio de estados) produzca
siempre el mismo checkpoint, las mismas metricas y las mismas graficas.

Uso: llamar una sola vez al inicio de cualquier script ejecutable:

    import repro
    repro.fijar_semilla()
"""

import os
import random

import numpy as np
import torch

import config


def fijar_semilla(seed=None):
    """Fija las semillas globales y activa el modo determinista de PyTorch."""
    if seed is None:
        seed = config.SEED

    os.environ["PYTHONHASHSEED"] = str(seed)
    # Requerido para que las operaciones de cuBLAS sean deterministas en GPU.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN: desactivar el autotuning (no determinista) y forzar rutas fijas.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed
