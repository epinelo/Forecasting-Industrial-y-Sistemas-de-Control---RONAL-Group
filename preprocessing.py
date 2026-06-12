"""
preprocessing.py
================
Preprocesamiento y limpieza de datos (Entregable: "preprocesamiento y limpieza").

Toma el CSV CRUDO y produce el CSV LIMPIO que consumen el resto de los scripts.
Reproduce la limpieza decidida en el análisis exploratorio:

  1. Reconstrucción del timestamp real (una muestra cada 20 s, 3 por minuto).
  2. Reemplazo de códigos de error del termopar (544.823 y 1003.413) por NaN.
  3. Interpolación lineal de los huecos resultantes en sensor_temp.

Ejecutar:
    python preprocessing.py
"""

import numpy as np
import pandas as pd

import config


def cargar_y_limpiar(ruta_crudos=config.RUTA_DATOS_CRUDOS):
    """Devuelve un DataFrame limpio indexado por timestamp."""
    df = pd.read_csv(ruta_crudos)

    # 1) Reconstruir el timestamp.
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
    df["intra"] = df.groupby("timestamp").cumcount()
    df["timestamp"] = df["timestamp"] + pd.to_timedelta(df["intra"] * config.TS,
                                                         unit="s")
    df = df.drop(columns="intra").set_index("timestamp").sort_index()

    # 2) Reemplazar códigos de error del sensor por NaN.
    df[config.OUTPUT_COL] = df[config.OUTPUT_COL].replace(
        config.CODIGOS_ERROR_SENSOR, np.nan)

    # 3) Interpolación lineal de los huecos del sensor.
    n_nan = int(df[config.OUTPUT_COL].isna().sum())
    df[config.OUTPUT_COL] = df[config.OUTPUT_COL].interpolate(method="linear")

    assert df.isna().sum().sum() == 0, "Quedan NaN tras la limpieza"
    assert not np.isinf(df.values).any(), "Hay valores Inf en el dataframe"

    print(f"Muestras: {len(df)}  |  variables: {df.shape[1]}")
    print(f"Valores de sensor corregidos (error/NaN): {n_nan}")
    print(f"sensor_temp -> rango [{df[config.OUTPUT_COL].min():.2f}, "
          f"{df[config.OUTPUT_COL].max():.2f}] °C")
    return df


def main():
    df = cargar_y_limpiar()
    config.RUTA_DATOS_LIMPIOS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RUTA_DATOS_LIMPIOS)
    print(f"\nGuardado: {config.RUTA_DATOS_LIMPIOS}")


if __name__ == "__main__":
    main()
