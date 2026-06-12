"""
nn_to_state_space.py
====================
Identificación del modelo en espacio de estados a partir del predictor neuronal
(Entregable: implementación del modelo de control moderno).

El CNN+LSTM entrenado se usa como SIMULADOR: dada una ventana de las 6 entradas
de proceso, predice la temperatura. Deslizando la ventana sobre entradas reales
se generan pares (U, Y), se ajusta un ARX por mínimos cuadrados y se convierte a
una realización en espacio de estados discreto:

        x(k+1) = Ad x(k) + Bd u(k)
        y(k)   = Cd x(k) + Dd u(k)

A diferencia de versiones previas, sensor_temp NO es entrada del modelo, así que
el buffer del simulador solo desliza las entradas de proceso (sin realimentar la
temperatura). U e Y van en el dominio ESCALADO (StandardScaler); la verificación
desnormaliza a °C con scaler_y. El orden ARX se elige por AIC y se verifica
estabilidad (|λ|<1) y ajuste contra el NN.

Ejecutar:
    python nn_to_state_space.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import eigvals

import config
import dataset
import repro
from model import cargar_modelo, cargar_escaladores


# --------------------------------------------------------------------------- #
# 1) Generar datos (u, y) usando el CNN+LSTM como simulador
# --------------------------------------------------------------------------- #
def generar_datos_io(model, X_scaled, window, n_traj, traj_len, device, seed):
    """Genera pares (U, Y) normalizados. Sin realimentación de sensor_temp."""
    model.eval()
    rng = np.random.RandomState(seed)
    indices = rng.choice(range(window, len(X_scaled) - traj_len),
                         n_traj, replace=False)

    U_list, Y_list = [], []
    for idx in indices:
        x_buf = X_scaled[idx - window:idx].copy()      # (window, 6)
        u_seq = X_scaled[idx:idx + traj_len]           # (traj_len, 6)
        y_seq = []
        for k in range(traj_len):
            xb = torch.tensor(x_buf[np.newaxis], dtype=torch.float32).to(device)
            with torch.no_grad():
                y_seq.append(model(xb).item())
            x_buf = np.vstack([x_buf[1:], u_seq[k:k + 1]])   # solo entradas
        U_list.append(u_seq)
        Y_list.append(np.array(y_seq))

    return np.vstack(U_list), np.concatenate(Y_list)


# --------------------------------------------------------------------------- #
# 2) Ajuste ARX por mínimos cuadrados
# --------------------------------------------------------------------------- #
def _regresores(U, Y, n):
    Y = np.asarray(Y).flatten()
    N, m = U.shape
    rows, targets = [], []
    for k in range(n, N):
        y_pasadas = Y[k - n:k][::-1]
        u_pasadas = U[k - n:k][::-1].reshape(-1)
        rows.append(np.concatenate([y_pasadas, u_pasadas]))
        targets.append(Y[k])
    return np.vstack(rows), np.array(targets)


def fit_arx(U, Y, n_order=4, verbose=True):
    m = U.shape[1]
    Phi, y_vec = _regresores(U, Y, n_order)
    theta, _, rank, _ = np.linalg.lstsq(Phi, y_vec, rcond=None)
    a = theta[:n_order]
    B = theta[n_order:].reshape(n_order, m)
    if verbose:
        r2 = 1 - np.sum((y_vec - Phi @ theta) ** 2) / np.sum((y_vec - y_vec.mean()) ** 2)
        print(f"  ARX n={n_order}: R²={r2:.6f}, rango Φ={rank}/{Phi.shape[1]}")
    return a, B


def seleccionar_orden(U, Y, ordenes=config.SS_ORDENES):
    resultados = []
    for n in ordenes:
        Phi, y_vec = _regresores(U, Y, n)
        theta, _, _, _ = np.linalg.lstsq(Phi, y_vec, rcond=None)
        residuo = y_vec - Phi @ theta
        sigma2 = np.sum(residuo ** 2) / len(y_vec)
        aic = len(y_vec) * np.log(sigma2) + 2 * len(theta)
        r2 = 1 - np.sum(residuo ** 2) / np.sum((y_vec - y_vec.mean()) ** 2)
        resultados.append({"n": n, "AIC": aic, "R2": r2})
        print(f"  n={n:2d}  AIC={aic:10.2f}  R²={r2:.6f}")
    mejor = min(resultados, key=lambda r: r["AIC"])
    print(f"  -> Orden recomendado por AIC: n={mejor['n']}")
    return mejor["n"]


# --------------------------------------------------------------------------- #
# 3) ARX -> espacio de estados discreto (forma compañera)
# --------------------------------------------------------------------------- #
def arx_a_espacio_estados(a, B, m):
    n = len(a)
    Ad = np.zeros((n, n))
    Ad[0, :] = a
    Ad[1:, :-1] = np.eye(n - 1)
    Bd = np.zeros((n, m)); Bd[0, :] = B[0, :]
    Cd = np.zeros((1, n)); Cd[0, 0] = 1.0
    Dd = np.zeros((1, m))
    return Ad, Bd, Cd, Dd


# --------------------------------------------------------------------------- #
# 4) Verificación: estabilidad + simulación vs CNN+LSTM
# --------------------------------------------------------------------------- #
def verificar(Ad, Bd, Cd, Dd, U_test, Y_test_nn, scaler_y, n_order, ts=config.TS):
    print("\nVerificación de estabilidad (|λ| < 1):")
    eigs = eigvals(Ad)
    estable = all(abs(e) < 1.0 for e in eigs)
    for i, e in enumerate(eigs):
        print(f"  λ{i+1} = {e.real:+.4f}{e.imag:+.4f}j  |λ|={abs(e):.4f}")
    print("  ->", "ESTABLE" if estable else "INESTABLE (prueba reducir el orden)")

    nx = Ad.shape[0]
    x = Y_test_nn[0] * np.ones(nx)
    y_ss = []
    for k in range(len(U_test)):
        y_ss.append((Cd @ x + Dd @ U_test[k]).item())
        x = Ad @ x + Bd @ U_test[k]
    y_ss = np.array(y_ss)

    y_ss_C = scaler_y.inverse_transform(y_ss.reshape(-1, 1)).flatten()
    y_nn_C = scaler_y.inverse_transform(Y_test_nn.reshape(-1, 1)).flatten()
    rmse = np.sqrt(np.mean((y_ss_C - y_nn_C) ** 2))
    mae = np.mean(np.abs(y_ss_C - y_nn_C))
    print(f"\n  SS vs CNN+LSTM -> RMSE={rmse:.3f} °C, MAE={mae:.3f} °C")

    t = np.arange(len(y_ss_C)) * ts
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"ARX n={n_order} -> espacio de estados | RMSE={rmse:.2f} °C")
    ax[0].plot(t, y_nn_C, label="CNN+LSTM (referencia)", color="#7B66CC", lw=1.5)
    ax[0].plot(t, y_ss_C, "--", label="Espacio de estados (ARX)", color="#C8921A", lw=1.5)
    ax[0].set_ylabel("Temperatura [°C]"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(t, y_nn_C - y_ss_C, color="#CC6666", lw=1.0)
    ax[1].axhline(0, color="gray", lw=0.5)
    ax[1].set_ylabel("Error [°C]"); ax[1].set_xlabel("Tiempo [s]"); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    ruta = config.RUTA_FIGURAS / "verificacion_arx_ss.png"
    plt.savefig(ruta, dpi=150); plt.close()
    print(f"  Figura guardada: {ruta}")
    return rmse


# --------------------------------------------------------------------------- #
# Orquestador
# --------------------------------------------------------------------------- #
def main():
    repro.fijar_semilla()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)

    model, ckpt = cargar_modelo(config.RUTA_CHECKPOINT, device=device,
                                dropout=config.DROPOUT)
    scaler_X, scaler_y = cargar_escaladores(ckpt)
    window = ckpt["hyperparameters"]["window"]

    df = dataset.cargar_datos_limpios()
    X_scaled, _, _, _ = dataset.escalar_con(df, scaler_X, scaler_y)

    print("1) Generando datos (u, y) con el CNN+LSTM...")
    U, Y = generar_datos_io(model, X_scaled, window,
                            config.SS_N_TRAJ, config.SS_TRAJ_LEN,
                            device, config.SS_SEED)
    print(f"   U={U.shape}, Y={Y.shape}")

    print("\n2) Selección de orden (AIC):")
    n_opt = seleccionar_orden(U, Y)

    print("\n3) Ajuste ARX final:")
    a, B = fit_arx(U, Y, n_order=n_opt)
    print(f"   coef. AR: {a.round(4)}  (suma={a.sum():.4f})")

    print("\n4) Conversión a espacio de estados:")
    m = U.shape[1]
    Ad, Bd, Cd, Dd = arx_a_espacio_estados(a, B, m)
    print(f"   Ad{Ad.shape}  Bd{Bd.shape}  Cd{Cd.shape}  Dd{Dd.shape}")

    print("\n5) Verificación en trayectoria nueva:")
    U_t, Y_t = generar_datos_io(model, X_scaled, window, 1, 400,
                                device, config.SS_SEED + 1)
    verificar(Ad, Bd, Cd, Dd, U_t, Y_t, scaler_y, n_opt)

    ruta = config.RUTA_MODELOS / "espacio_estados.npz"
    np.savez(ruta, Ad=Ad, Bd=Bd, Cd=Cd, Dd=Dd, Ts=config.TS, n_order=n_opt,
             scaler_X_mean=np.asarray(ckpt["scaler_X_mean"]),
             scaler_X_scale=np.asarray(ckpt["scaler_X_scale"]),
             scaler_y_mean=np.asarray(ckpt["scaler_y_mean"]),
             scaler_y_scale=np.asarray(ckpt["scaler_y_scale"]))
    print(f"\nMatrices guardadas en {ruta}  (para diseño de controlador)")


if __name__ == "__main__":
    main()
