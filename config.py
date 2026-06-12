"""
config.py
=========
Única fuente de verdad para rutas e hiperparámetros del proyecto.

Equipo 2 — Horno fusor — Control moderno (MA2008B, RONAL GROUP).

Versión "corregida": el modelo predice la temperatura ABSOLUTA usando únicamente
las 6 variables de proceso como entrada (sin sensor_temp -> sin leakage), con
StandardScaler ajustado SOLO con train.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas (relativas a la raíz del repositorio)
# --------------------------------------------------------------------------- #
RAIZ = Path(__file__).resolve().parent

# El CSV crudo vive en la raíz para que el notebook de EDA ya existente
# (EDA_Horno_Fusor.ipynb), que lo lee con ruta relativa, siga funcionando.
RUTA_DATOS_CRUDOS   = RAIZ / "dataset_horno_fusor1.csv"
RUTA_DATOS_LIMPIOS  = RAIZ / "dataset_horno_fusor_cleaned.csv"
RUTA_MODELOS        = RAIZ / "models"
RUTA_CHECKPOINT     = RUTA_MODELOS / "cnn_lstm_checkpoint.pth"
RUTA_FIGURAS        = RAIZ / "figures"

# --------------------------------------------------------------------------- #
# Definición de variables del proceso
# --------------------------------------------------------------------------- #
# 6 entradas físicas (las "u" del sistema). sensor_temp NO es entrada.
INPUT_COLS = ["gas_flow", "air_flow", "furnace_load",
              "ambient_temp", "gas_pressure", "energy_consumption"]

# Variable a predecir / controlar.
OUTPUT_COL = "sensor_temp"

# Periodo de muestreo (s). 3 muestras por minuto.
TS = 20

# Semilla global de reproducibilidad (usada por repro.fijar_semilla()).
SEED = 42

# Códigos de error del termopar detectados en el EDA (se reemplazan por NaN).
CODIGOS_ERROR_SENSOR = [544.823, 1003.413]

# --------------------------------------------------------------------------- #
# Hiperparámetros del modelo CNN + LSTM
# --------------------------------------------------------------------------- #
# Reflejan la arquitectura realmente entrenada. Para CARGAR el checkpoint, el
# loader infiere la arquitectura de los pesos (model.py), así que no dependemos
# de que estos números coincidan con un archivo viejo.
WINDOW       = 200    # ventana temporal (pasos)
HORIZON      = 1      # horizonte de predicción (pasos hacia adelante)
N_FILTERS    = 64     # filtros de las capas convolucionales
KERNEL_SIZE  = 3      # tamaño de kernel de la primera convolución
LSTM_HIDDEN  = 64     # unidades ocultas del LSTM
LSTM_LAYERS  = 2      # capas LSTM apiladas
DROPOUT      = 0.2

# --------------------------------------------------------------------------- #
# Hiperparámetros de entrenamiento
# --------------------------------------------------------------------------- #
EPOCHS         = 50
BATCH_SIZE     = 64
LEARNING_RATE  = 1e-3
SMOOTHL1_BETA  = 1.0
GRAD_CLIP      = 1.0
SCHED_PATIENCE = 5

# Particiones temporales (no aleatorias: serie de tiempo).
FRAC_TRAIN = 0.70
FRAC_VAL   = 0.85   # train: [0,0.70)  val: [0.70,0.85)  test: [0.85,1.0]

# --------------------------------------------------------------------------- #
# Identificación de espacio de estados (control moderno)
# --------------------------------------------------------------------------- #
SS_N_TRAJ      = 60          # trayectorias para ajustar el ARX
SS_TRAJ_LEN    = 500         # largo de cada trayectoria
SS_ORDENES     = range(1, 8) # órdenes ARX a probar (selección por AIC)
SS_SEED        = SEED        # semilla para reproducibilidad (semilla global)
