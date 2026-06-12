"""
dataset.py
==========
Módulo compartido de datos. Lo importan train.py, evaluate.py y
nn_to_state_space.py. No ejecuta nada por sí mismo.

Responsabilidades:
  - Leer el CSV limpio.
  - Escalar entradas (6 variables de proceso) y salida con StandardScaler
    AJUSTADO SOLO CON TRAIN (sin leakage del val/test).
  - Construir ventanas deslizantes (Dataset de PyTorch).
  - Particionar temporalmente train / val / test (sin barajar).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import config


# --------------------------------------------------------------------------- #
# Carga
# --------------------------------------------------------------------------- #
def cargar_datos_limpios(ruta=config.RUTA_DATOS_LIMPIOS):
    """Lee el CSV limpio. Si no existe, lo genera con preprocessing.py."""
    if not ruta.exists():
        import preprocessing
        df = preprocessing.cargar_y_limpiar()
        ruta.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta)
        return df
    df = pd.read_csv(ruta, parse_dates=["timestamp"]).set_index("timestamp")
    return df.sort_index()


def cortes(n):
    """Índices de corte temporal train/val."""
    return int(n * config.FRAC_TRAIN), int(n * config.FRAC_VAL)


# --------------------------------------------------------------------------- #
# Escalado
# --------------------------------------------------------------------------- #
def ajustar_y_escalar(df):
    """
    Para ENTRENAMIENTO: ajusta StandardScaler SOLO con train y transforma todo.
    Devuelve (X_scaled, y_scaled, scaler_X, scaler_y, train_end, val_end).
    """
    X_raw = df[config.INPUT_COLS].values     # 6 variables de proceso
    y_raw = df[[config.OUTPUT_COL]].values
    train_end, val_end = cortes(len(X_raw))

    scaler_X = StandardScaler().fit(X_raw[:train_end])
    scaler_y = StandardScaler().fit(y_raw[:train_end])

    X_scaled = scaler_X.transform(X_raw).astype(np.float32)
    y_scaled = scaler_y.transform(y_raw).astype(np.float32)

    assert not np.isnan(X_scaled).any() and not np.isnan(y_scaled).any()
    return X_scaled, y_scaled, scaler_X, scaler_y, train_end, val_end


def escalar_con(df, scaler_X, scaler_y):
    """
    Para EVALUACIÓN / ESPACIO DE ESTADOS: transforma con escaladores ya
    entrenados (los del checkpoint), sin re-ajustar.
    Devuelve (X_scaled, y_scaled, train_end, val_end).
    """
    X_scaled = scaler_X.transform(df[config.INPUT_COLS].values).astype(np.float32)
    y_scaled = scaler_y.transform(df[[config.OUTPUT_COL]].values).astype(np.float32)
    train_end, val_end = cortes(len(X_scaled))
    return X_scaled, y_scaled, train_end, val_end


# --------------------------------------------------------------------------- #
# Dataset de ventanas
# --------------------------------------------------------------------------- #
class FurnaceDataset(Dataset):
    """Convierte una serie (N, features) en ventanas (window, features) -> (horizon, 1)."""

    def __init__(self, X, y, window, horizon):
        self.X, self.y = [], []
        for i in range(len(X) - window - horizon + 1):
            self.X.append(X[i:i + window])
            self.y.append(y[i + window:i + window + horizon])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def construir_dataloaders(X_scaled, y_scaled, train_end, val_end,
                          window=config.WINDOW, horizon=config.HORIZON,
                          batch_size=config.BATCH_SIZE):
    """Particiona temporalmente y devuelve (dl_train, dl_val, dl_test)."""
    pin = torch.cuda.is_available()
    ds_train = FurnaceDataset(X_scaled[:train_end], y_scaled[:train_end], window, horizon)
    ds_val = FurnaceDataset(X_scaled[train_end:val_end], y_scaled[train_end:val_end], window, horizon)
    ds_test = FurnaceDataset(X_scaled[val_end:], y_scaled[val_end:], window, horizon)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=False, pin_memory=pin)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, pin_memory=pin)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, pin_memory=pin)

    print(f"Ventanas -> train: {len(ds_train)}  val: {len(ds_val)}  test: {len(ds_test)}")
    return dl_train, dl_val, dl_test
