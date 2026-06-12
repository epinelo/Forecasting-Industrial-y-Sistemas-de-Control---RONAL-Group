# Forecasting Industrial y Sistemas de Control — RONAL Group

Modelado y control de la temperatura del **horno fusor de aluminio** de RONAL Group.
El proyecto combina dos enfoques complementarios:

1. **Predictor de la dinámica térmica** basado en *Deep Learning* (CNN + LSTM), que
   aprende a estimar la temperatura del horno a partir de las variables de proceso.
2. **Modelo de control moderno** en espacio de estados, identificado a partir del
   predictor neuronal mediante un ajuste ARX, listo para diseñar un controlador.

Reto **MA2008B** — Equipo 2 — ITESM Campus Querétaro.

---

## El problema

El horno se monitorea con un termopar (`sensor_temp`) muestreado cada **20 s**
(3 muestras/min). El objetivo es modelar cómo evoluciona la temperatura en función
de **6 variables de proceso controlables/medibles** y obtener un modelo de planta
útil para control.

| Variable | Rol |
|---|---|
| `gas_flow`, `air_flow`, `furnace_load`, `ambient_temp`, `gas_pressure`, `energy_consumption` | **Entradas** del sistema (las *u*) |
| `sensor_temp` | **Salida** a predecir / controlar (la *y*) |

> El modelo predice la temperatura **absoluta** usando *únicamente* las 6 variables
> de proceso. `sensor_temp` **no** es entrada del predictor, lo que evita *data
> leakage*. El `StandardScaler` se ajusta **solo con la partición de entrenamiento**.

---

## Arquitectura del pipeline

```
CSV crudo
   │  preprocessing.py   (limpieza: timestamps, códigos de error, interpolación)
   ▼
CSV limpio
   │  train.py           (ventanas deslizantes → CNN+LSTM → checkpoint .pth)
   ▼
models/cnn_lstm_checkpoint.pth
   │
   ├─ evaluate.py            → métricas (MAE, RMSE, R²) + figuras
   │
   └─ nn_to_state_space.py   (NN como simulador → ARX por AIC → espacio de estados)
          ▼
   models/espacio_estados.npz   → matrices A, B, C, D para diseño de controlador
```

Todos los scripts comparten una **única fuente de verdad** ([config.py](config.py))
para rutas e hiperparámetros, y se inicializan con [repro.py](repro.py) para que el
pipeline completo sea **reproducible** (semilla global, modo determinista de PyTorch).

---

## Estructura del repositorio

| Archivo | Descripción |
|---|---|
| [config.py](config.py) | Rutas, variables de proceso e hiperparámetros (única fuente de verdad). |
| [repro.py](repro.py) | Fija semillas y fuerza determinismo en PyTorch. |
| [preprocessing.py](preprocessing.py) | Limpieza del CSV crudo → CSV limpio. |
| [dataset.py](dataset.py) | Carga, escalado sin *leakage*, ventanas deslizantes y particiones temporales. |
| [model.py](model.py) | Arquitectura `CNN_LSTM` y carga/guardado robusto del checkpoint. |
| [train.py](train.py) | Entrenamiento del predictor CNN+LSTM. |
| [evaluate.py](evaluate.py) | Evaluación en *test* + generación de gráficas. |
| [nn_to_state_space.py](nn_to_state_space.py) | Identificación del modelo en espacio de estados (ARX). |
| [EDA_Horno_Fusor.ipynb](EDA_Horno_Fusor.ipynb) | Análisis exploratorio de datos. |
| [Modelo_final.slx](Modelo_final.slx) | Modelo de Simulink del sistema de control. |
| `dataset_horno_fusor1.csv` | Datos crudos del horno (21 600 muestras). |
| `dataset_horno_fusor_cleaned.csv` | Datos limpios (generados por `preprocessing.py`). |
| `models/cnn_lstm_checkpoint.pth` | Pesos y escaladores del modelo entrenado. |
| `models/espacio_estados.npz` | Matrices A, B, C, D del modelo en espacio de estados. |
| `figures/` | Gráficas generadas por los scripts de evaluación. |

---

## Instalación

Requiere **Python 3.10+**.

```bash
pip install -r requirements.txt
```

Dependencias principales: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`,
`torch`. El pipeline corre en **CPU**; usa GPU automáticamente si hay CUDA disponible.

---

## Uso

Ejecuta los scripts desde la raíz del repositorio, en orden:

```bash
# 1) Limpieza de datos  →  dataset_horno_fusor_cleaned.csv
python preprocessing.py

# 2) Entrenamiento del predictor  →  models/cnn_lstm_checkpoint.pth
python train.py

# 3) Evaluación + gráficas  →  figures/pred_vs_real.png, figures/error_absoluto.png
python evaluate.py

# 4) Identificación en espacio de estados  →  models/espacio_estados.npz
python nn_to_state_space.py
```

El repositorio ya incluye el checkpoint y las matrices entrenadas, así que puedes
ejecutar directamente `evaluate.py` o `nn_to_state_space.py` sin reentrenar.
Si el CSV limpio no existe, `dataset.py` lo genera automáticamente.

---

## Detalles técnicos

### Limpieza de datos ([preprocessing.py](preprocessing.py))
- Reconstrucción del *timestamp* real (una muestra cada 20 s).
- Reemplazo de los códigos de error del termopar (`544.823`, `1003.413`) por `NaN`.
- Interpolación lineal de los huecos resultantes en `sensor_temp`.

### Modelo predictor ([model.py](model.py))
- **CNN 1D** (2 capas convolucionales + ReLU + *dropout*) sobre la dimensión temporal,
  seguida de un **LSTM** apilado (2 capas) y una cabeza lineal.
- Ventana de **200 pasos**, horizonte de **1 paso**.
- Pérdida `SmoothL1Loss`, optimizador `Adam`, *scheduler* `ReduceLROnPlateau`,
  *gradient clipping*.
- Particiones **temporales** (no aleatorias): train 70 % / val 15 % / test 15 %.
- El *loader* **infiere la arquitectura desde los pesos**, no de metadatos externos,
  garantizando que el checkpoint siempre sea coherente.

### Identificación en espacio de estados ([nn_to_state_space.py](nn_to_state_space.py))
El CNN+LSTM se usa como **simulador**: deslizando ventanas de entradas reales se
generan pares (U, Y), se ajusta un **ARX por mínimos cuadrados** (orden elegido por
**AIC**) y se convierte a una realización discreta en forma compañera:

```
x(k+1) = Ad·x(k) + Bd·u(k)
y(k)   = Cd·x(k) + Dd·u(k)
```

Se verifica **estabilidad** (|λ| < 1) y el ajuste contra el predictor neuronal.
Las matrices se guardan en `models/espacio_estados.npz` junto con los escaladores,
listas para el diseño del controlador y para [Modelo_final.slx](Modelo_final.slx).

---

## Resultados

Las gráficas se regeneran al ejecutar los scripts y se guardan en [figures/](figures/):

- `pred_vs_real.png` — predicción del CNN+LSTM vs. temperatura real (*test*).
- `error_absoluto.png` — error absoluto a lo largo del conjunto de *test*.
- `verificacion_arx_ss.png` — modelo en espacio de estados vs. predictor neuronal.

`evaluate.py` imprime las métricas **RMSE**, **MAE** y **R²** en °C sobre el conjunto
de prueba.

---

*Proyecto académico — Reto MA2008B, ITESM Campus Querétaro · RONAL Group.*
