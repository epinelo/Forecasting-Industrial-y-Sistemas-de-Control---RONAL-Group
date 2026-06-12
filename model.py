"""
model.py
========
Arquitectura CNN + LSTM (predicción de temperatura absoluta) y utilidades de
guardado/carga robusta.

Este módulo NO ejecuta nada por sí solo: solo define clases y funciones que
importan train.py, evaluate.py y nn_to_state_space.py.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# Arquitectura
# --------------------------------------------------------------------------- #
class CNN_LSTM(nn.Module):
    """CNN 1D + LSTM para predicción de temperatura del horno fusor."""

    def __init__(self, n_features, n_filters=64, kernel_size=3,
                 lstm_hidden=64, lstm_layers=2, horizon=1, dropout=0.2):
        super().__init__()

        # Bloque CNN (convolución sobre la dimensión temporal).
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=n_filters,
                      kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bloque LSTM.
        self.lstm = nn.LSTM(
            input_size=n_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Cabeza de predicción.
        self.fc = nn.Linear(lstm_hidden, horizon)

    def forward(self, x):
        # x: (batch, window, features)
        x = x.permute(0, 2, 1)           # -> (batch, features, window) para Conv1d
        x = self.cnn(x)                  # -> (batch, n_filters, window)
        x = x.permute(0, 2, 1)           # -> (batch, window, n_filters) para LSTM
        out, _ = self.lstm(x)            # out: (batch, window, hidden)
        last = out[:, -1, :]             # último paso temporal
        return self.fc(last)             # (batch, horizon)

    def get_hidden_state(self, x):
        """Secuencia completa de estados ocultos (útil para análisis latente)."""
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return out


# --------------------------------------------------------------------------- #
# Inferir arquitectura desde los pesos (robustez)
# --------------------------------------------------------------------------- #
def inferir_arquitectura(state_dict):
    """
    Deduce los hiperparámetros estructurales desde las formas reales de los
    tensores entrenados. A prueba de errores: los pesos nunca mienten (en el
    checkpoint, el diccionario 'hyperparameters' quedó desincronizado).
    """
    # cnn.0.weight: (n_filters, n_features, kernel_size)
    n_filters, n_features, kernel_size = state_dict["cnn.0.weight"].shape
    # lstm.weight_ih_l0: (4 * hidden, n_filters)
    lstm_hidden = state_dict["lstm.weight_ih_l0"].shape[0] // 4
    lstm_layers = len([k for k in state_dict if k.startswith("lstm.weight_ih_l")])
    horizon = state_dict["fc.weight"].shape[0]
    return {
        "n_features": int(n_features),
        "n_filters": int(n_filters),
        "kernel_size": int(kernel_size),
        "lstm_hidden": int(lstm_hidden),
        "lstm_layers": int(lstm_layers),
        "horizon": int(horizon),
    }


# --------------------------------------------------------------------------- #
# Reconstrucción de los escaladores (StandardScaler) desde el checkpoint
# --------------------------------------------------------------------------- #
def _rebuild_standard(mean, scale):
    """Reconstruye un StandardScaler desde sus parámetros (mean_, scale_)."""
    mean = np.asarray(mean, dtype=float)
    scale = np.asarray(scale, dtype=float)
    s = StandardScaler()
    s.mean_ = mean
    s.scale_ = scale
    s.var_ = scale ** 2
    s.n_features_in_ = len(scale)
    s.n_samples_seen_ = 1
    return s


def cargar_escaladores(checkpoint):
    """Devuelve (scaler_X, scaler_y) reconstruidos desde el checkpoint."""
    scaler_X = _rebuild_standard(checkpoint["scaler_X_mean"],
                                 checkpoint["scaler_X_scale"])
    scaler_y = _rebuild_standard(checkpoint["scaler_y_mean"],
                                 checkpoint["scaler_y_scale"])
    return scaler_X, scaler_y


# --------------------------------------------------------------------------- #
# Cargar / guardar el modelo entrenado
# --------------------------------------------------------------------------- #
def cargar_modelo(ruta_checkpoint, device="cpu", dropout=0.2):
    """
    Carga el modelo entrenado de forma robusta:
      1. Lee el checkpoint.
      2. Infiere la arquitectura desde los pesos (no desde 'hyperparameters').
      3. Construye el modelo, carga los pesos y lo deja en eval().

    Devuelve (model, checkpoint).
    """
    checkpoint = torch.load(ruta_checkpoint, map_location=device,
                            weights_only=False)
    arch = inferir_arquitectura(checkpoint["model_state_dict"])
    # Usa el dropout realmente guardado si está disponible (no afecta a eval(),
    # pero mantiene coherencia si se reanuda el entrenamiento).
    dropout = checkpoint.get("hyperparameters", {}).get("dropout", dropout)
    model = CNN_LSTM(dropout=dropout, **arch).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def guardar_checkpoint(ruta, model, optimizer, epoch, val_loss, window, dropout,
                       input_cols, output_col, scaler_X, scaler_y, training=None):
    """
    Guarda el checkpoint con hiperparámetros SIEMPRE coherentes con los pesos.

    La arquitectura NO se hardcodea ni se confía a un dict externo: se infiere
    directamente del state_dict del modelo que se está guardando. Así es
    imposible que 'hyperparameters' se desincronice de los pesos reales.

    - dropout no afecta las formas de los tensores, así que no es inferible;
      se registra el valor realmente usado al construir el modelo.
    - 'training' guarda los hiperparámetros de entrenamiento que de verdad se
      usaron (lr, epochs, batch_size, etc.) para trazabilidad y reproducibilidad.
    """
    arch = inferir_arquitectura(model.state_dict())   # desde los pesos reales

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "hyperparameters": {**arch, "dropout": dropout, "window": window},
        "training": training or {},
        "input_cols": input_cols,
        "output_col": output_col,
        "scaler_X_mean": np.asarray(scaler_X.mean_).tolist(),
        "scaler_X_scale": np.asarray(scaler_X.scale_).tolist(),
        "scaler_y_mean": np.asarray(scaler_y.mean_).tolist(),
        "scaler_y_scale": np.asarray(scaler_y.scale_).tolist(),
    }
    torch.save(checkpoint, ruta)
