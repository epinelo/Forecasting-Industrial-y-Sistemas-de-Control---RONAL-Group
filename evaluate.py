"""
evaluate.py
===========
Evaluación del predictor neuronal (Entregable: "scripts de evaluación" y
"generación de gráficas").

Flujo independiente:
    checkpoint .pth + CSV limpio -> métricas (MAE, RMSE, R²) + figuras

NO reentrena: carga el modelo y los escaladores del checkpoint. Corre en CPU en
segundos.

Ejecutar:
    python evaluate.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config
import dataset
import repro
from model import cargar_modelo, cargar_escaladores


def evaluar():
    repro.fijar_semilla()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)

    # 1) Modelo + escaladores desde el checkpoint.
    model, ckpt = cargar_modelo(config.RUTA_CHECKPOINT, device=device,
                                dropout=config.DROPOUT)
    scaler_X, scaler_y = cargar_escaladores(ckpt)
    window = ckpt["hyperparameters"]["window"]
    horizon = ckpt["hyperparameters"]["horizon"]

    # 2) Datos escalados con los escaladores del checkpoint.
    df = dataset.cargar_datos_limpios()
    X_scaled, y_scaled, tr_end, va_end = dataset.escalar_con(df, scaler_X, scaler_y)
    _, _, dl_test = dataset.construir_dataloaders(
        X_scaled, y_scaled, tr_end, va_end, window=window, horizon=horizon)

    # 3) Predicción sobre test.
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            out = model(xb.to(device))
            y_true.append(yb.cpu().numpy().reshape(-1))
            y_pred.append(out.cpu().numpy().reshape(-1))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # 4) Desnormalizar a °C.
    y_true_C = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_C = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_true_C, y_pred_C))
    mae = mean_absolute_error(y_true_C, y_pred_C)
    r2 = r2_score(y_true_C, y_pred_C)

    print("=" * 50)
    print("EVALUACIÓN RED NEURONAL (TEST)")
    print("=" * 50)
    print(f"RMSE: {rmse:.3f} °C")
    print(f"MAE:  {mae:.3f} °C")
    print(f"R²:   {r2:.6f}")

    # 5) Gráfica predicción vs real.
    plt.figure(figsize=(10, 5))
    plt.plot(y_true_C, label="Real")
    plt.plot(y_pred_C, "--", label="Predicción NN")
    plt.title("Predicción vs Real (Test)")
    plt.xlabel("Paso temporal"); plt.ylabel("Temperatura [°C]")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    f1 = config.RUTA_FIGURAS / "pred_vs_real.png"
    plt.savefig(f1, dpi=150); plt.close()

    # 6) Gráfica de error absoluto.
    plt.figure(figsize=(10, 4))
    plt.plot(np.abs(y_true_C - y_pred_C), color="red")
    plt.title("Error absoluto (Test)")
    plt.xlabel("Paso temporal"); plt.ylabel("Error [°C]")
    plt.grid(alpha=0.3); plt.tight_layout()
    f2 = config.RUTA_FIGURAS / "error_absoluto.png"
    plt.savefig(f2, dpi=150); plt.close()

    print(f"\nFiguras guardadas:\n  {f1}\n  {f2}")
    return rmse, mae, r2


if __name__ == "__main__":
    evaluar()
