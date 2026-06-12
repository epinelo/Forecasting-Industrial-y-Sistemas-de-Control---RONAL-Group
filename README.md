# Modelado y Control del Horno Fusor — RONAL Group

Este proyecto modela y controla la **temperatura de un horno fusor de aluminio**
de RONAL Group, usado en la fabricación de rines por colada a baja presión (LPDC).

El objetivo es sencillo: lograr que la temperatura del baño de aluminio se mantenga
estable en su valor deseado (**780 °C**), incluso cuando el proceso cambia.

Para lograrlo combinamos **dos enfoques** que se complementan:

1. **Modelo de control (caja blanca):** un modelo físico del horno, hecho en Simulink,
   con un controlador que garantiza estabilidad.
2. **Modelo de Deep Learning (caja negra):** una red neuronal (CNN + LSTM) que aprende
   la dinámica real del horno a partir de los datos medidos.

Al final, la red neuronal se convierte en un modelo lineal (ARX → espacio de estados)
para poder analizarlo y diseñar controladores con herramientas clásicas.

Reto **MA2008B** — Equipo 2 — ITESM Campus Querétaro.

---

## El problema en pocas palabras

El horno tiene un sensor de temperatura (`sensor_temp`) que mide cada **20 segundos**.
Queremos predecir y controlar esa temperatura a partir de **6 variables de proceso**
que sí podemos medir o ajustar:

| Variable | Qué es | Rol |
|---|---|---|
| `gas_flow` | Flujo de gas natural | Entrada |
| `air_flow` | Flujo de aire | Entrada |
| `furnace_load` | Carga del horno | Entrada |
| `ambient_temp` | Temperatura ambiente | Entrada |
| `gas_pressure` | Presión del gas | Entrada |
| `energy_consumption` | Consumo de energía | Entrada |
| `sensor_temp` | **Temperatura del baño** | Salida (lo que predecimos) |

> **Importante:** la red predice la temperatura usando *solo* esas 6 variables.
> Nunca usa la temperatura pasada como entrada, así que no hay "trampa" (*data leakage*).
> Además, el escalado de datos se ajusta **solo con los datos de entrenamiento**.

---

## ¿Cómo funciona? (el flujo de trabajo)

```
  dataset crudo (CSV)
        │
        │   1. preprocessing.py   → limpia los datos
        ▼
  dataset limpio (CSV)
        │
        │   2. train.py          → entrena la red CNN+LSTM
        ▼
  modelo entrenado (.pth)
        │
        ├── 3. evaluate.py          → mide qué tan bien predice (MAE, RMSE, R²)
        │
        └── 4. nn_to_state_space.py → convierte la red en matrices A, B, C, D
                                       (modelo de espacio de estados para control)
```

Todos los scripts leen su configuración de un solo archivo, [config.py](config.py),
y usan [repro.py](repro.py) para fijar las semillas aleatorias. Así, **cualquiera que
ejecute el proyecto obtiene los mismos resultados**.

---

## Cómo usarlo

**1. Instalar dependencias** (necesitas Python 3.10 o superior):

```bash
pip install -r requirements.txt
```

**2. Ejecutar los scripts en orden** desde la carpeta del proyecto:

```bash
python preprocessing.py      # limpia los datos
python train.py              # entrena la red (opcional: ya viene entrenada)
python evaluate.py           # evalúa y genera gráficas
python nn_to_state_space.py  # genera el modelo de espacio de estados
```

> El repositorio **ya incluye** el modelo entrenado y las matrices del espacio de
> estados. Si solo quieres ver resultados, puedes correr directamente `evaluate.py`
> o `nn_to_state_space.py` sin reentrenar nada.

---

## Resultados

**Predicción de la red (CNN + LSTM)**, sobre datos que nunca vio:

| Métrica | Valor | Qué significa |
|---|---|---|
| MAE | **1.83 °C** | error promedio (~0.23 % del setpoint) |
| RMSE | 5.20 °C | error que penaliza más los picos grandes |

> En operación normal el error es de menos de 2 °C. El RMSE sube porque hay unos
> pocos picos bruscos (eventos de carga) que la red, a propósito, no persigue:
> actúa como un simulador suave de la planta.

**Conversión a espacio de estados (ARX):**

| Resultado | Valor |
|---|---|
| Orden elegido (por criterio AIC) | n = 7 |
| Ajuste del ARX a la red (R²) | 0.986 |
| Diferencia ARX vs red | RMSE = 2.13 °C |
| Estabilidad | Estable (todos los polos con \|λ\| < 1) |

Las gráficas se guardan en [figures/](figures/):

- `pred_vs_real.png` — predicción de la red vs. temperatura real.
- `error_absoluto.png` — error a lo largo del conjunto de prueba.
- `verificacion_arx_ss.png` — modelo de espacio de estados vs. red neuronal.
- `Comparación control vs deep.jpeg` — comparación de ambos enfoques.

---

## Modelo de control (Simulink)

En la carpeta [Modelo control/](Modelo%20control/) está el modelo de control moderno:

- `Simulink_ronal.slx` — el modelo de control en Simulink.
- `ARX_PREDICCION.mat`, `sim1.mat` — datos de simulación.

Usa el modelo de espacio de estados (`models/espacio_estados.npz`) para diseñar un
controlador por **realimentación de estados** con un **observador**, sintonizado con
la **fórmula de Ackermann** (ubicación de polos).

---

## Qué hay en cada archivo

| Archivo | Para qué sirve |
|---|---|
| [config.py](config.py) | Toda la configuración: rutas, variables e hiperparámetros. |
| [repro.py](repro.py) | Fija las semillas para que todo sea reproducible. |
| [preprocessing.py](preprocessing.py) | Limpia el CSV crudo (errores del sensor, huecos, fechas). |
| [dataset.py](dataset.py) | Arma las ventanas de datos y las particiones de tiempo. |
| [model.py](model.py) | Define la arquitectura de la red CNN + LSTM. |
| [train.py](train.py) | Entrena la red. |
| [evaluate.py](evaluate.py) | Evalúa la red y genera gráficas. |
| [nn_to_state_space.py](nn_to_state_space.py) | Convierte la red en un modelo de espacio de estados. |
| [EDA_Horno_Fusor.ipynb](EDA_Horno_Fusor.ipynb) | Análisis exploratorio de los datos. |
| `dataset_horno_fusor1.csv` | Datos crudos del horno (21 600 muestras). |
| `dataset_horno_fusor_cleaned.csv` | Datos ya limpios. |
| `models/` | Modelo entrenado (`.pth`) y matrices del espacio de estados (`.npz`). |
| `figures/` | Gráficas de resultados. |
| `Modelo control/` | Modelo de Simulink y datos de simulación. |

---

## Detalles técnicos (para quien quiera profundizar)

**Limpieza de datos** ([preprocessing.py](preprocessing.py)):
reconstruye la fecha real (una muestra cada 20 s), reemplaza los códigos de error del
sensor (544.823 y 1003.413) por valores faltantes, y rellena los huecos por
interpolación lineal.

**La red CNN + LSTM** ([model.py](model.py)):
dos capas convolucionales 1D (que detectan patrones entre las variables) seguidas de
un LSTM de 2 capas (que recuerda la historia térmica). Usa una ventana de **200 pasos**
(≈ 67 minutos) para predecir el siguiente paso. Se entrena con `SmoothL1Loss`, optimizador
`Adam` y *gradient clipping*. Las particiones son **temporales** (no aleatorias):
70 % entrenamiento, 15 % validación, 15 % prueba.

**Espacio de estados** ([nn_to_state_space.py](nn_to_state_space.py)):
la red se usa como simulador para generar pares entrada-salida limpios. Sobre ellos se
ajusta un modelo ARX por mínimos cuadrados (el orden se elige con el criterio AIC) y se
convierte a una realización discreta:

```
x(k+1) = Ad·x(k) + Bd·u(k)
y(k)   = Cd·x(k) + Dd·u(k)
```

Se verifica que sea estable y que reproduzca bien a la red. Las matrices se guardan en
`models/espacio_estados.npz`, listas para diseñar el controlador.

---

*Proyecto académico — Reto MA2008B, ITESM Campus Querétaro · RONAL Group.*
