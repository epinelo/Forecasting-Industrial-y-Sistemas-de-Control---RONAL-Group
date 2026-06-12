"""
train.py
========
Entrenamiento del modelo CNN + LSTM (Entregable: "scripts de entrenamiento").

Flujo independiente:
    CSV limpio -> ventanas -> entrenamiento -> checkpoint .pth

Lee los datos de disco y, al terminar, escribe models/cnn_lstm_checkpoint.pth
con hiperparámetros COHERENTES con los pesos reales.

Ejecutar:
    python train.py
"""

import torch
import torch.nn as nn

import config
import dataset
import repro
from model import CNN_LSTM, guardar_checkpoint


def entrenar():
    repro.fijar_semilla()   # reproducibilidad: pesos, dropout y orden identicos
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)

    # 1) Datos (escaladores ajustados solo con train)
    df = dataset.cargar_datos_limpios()
    X_scaled, y_scaled, scaler_X, scaler_y, tr_end, va_end = dataset.ajustar_y_escalar(df)
    dl_train, dl_val, _ = dataset.construir_dataloaders(X_scaled, y_scaled, tr_end, va_end)

    # 2) Modelo
    arch = {
        "n_features": X_scaled.shape[1],
        "n_filters": config.N_FILTERS,
        "kernel_size": config.KERNEL_SIZE,
        "lstm_hidden": config.LSTM_HIDDEN,
        "lstm_layers": config.LSTM_LAYERS,
        "horizon": config.HORIZON,
    }
    model = CNN_LSTM(dropout=config.DROPOUT, **arch).to(device)

    # 3) Optimización
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.SmoothL1Loss(beta=config.SMOOTHL1_BETA)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=config.SCHED_PATIENCE)

    avg_val = float("nan")
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb.squeeze(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb.squeeze(-1)).item()

        avg_train = train_loss / len(dl_train)
        avg_val = val_loss / len(dl_val)
        scheduler.step(avg_val)
        print(f"Epoch {epoch:3d} | Train: {avg_train:.5f} | Val: {avg_val:.5f}")

    # 4) Guardar checkpoint coherente
    #    La arquitectura se infiere de los pesos dentro de guardar_checkpoint
    #    (no se hardcodea). Aquí registramos los hiperparámetros de entrenamiento
    #    realmente usados, para trazabilidad y reproducibilidad.
    training = {
        "optimizer": "Adam",
        "lr_inicial": config.LEARNING_RATE,
        "lr_final": optimizer.param_groups[0]["lr"],
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "loss": "SmoothL1Loss",
        "smoothl1_beta": config.SMOOTHL1_BETA,
        "grad_clip": config.GRAD_CLIP,
        "scheduler": "ReduceLROnPlateau",
        "sched_patience": config.SCHED_PATIENCE,
        "frac_train": config.FRAC_TRAIN,
        "frac_val": config.FRAC_VAL,
    }

    config.RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
    guardar_checkpoint(
        config.RUTA_CHECKPOINT, model, optimizer,
        epoch=epoch, val_loss=avg_val, window=config.WINDOW,
        dropout=config.DROPOUT,
        input_cols=config.INPUT_COLS, output_col=config.OUTPUT_COL,
        scaler_X=scaler_X, scaler_y=scaler_y, training=training,
    )
    print(f"\nCheckpoint guardado en {config.RUTA_CHECKPOINT}")


if __name__ == "__main__":
    entrenar()
