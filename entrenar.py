import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RUTA_BASE = Path(__file__).resolve().parent
RUTA_CSV = RUTA_BASE / "sensores_planta.csv"
RUTA_MODELO = RUTA_BASE / "modelo_mantenimiento.pkl"
COLUMNAS_REQUERIDAS = [
    "temperatura_c",
    "vibracion_mms",
    "horas_operacion",
    "presion_psi",
    "estado",
]


def main():
    print("1. Cargando conjunto de datos desde 'sensores_planta.csv'...")
    try:
        df = pd.read_csv(RUTA_CSV)
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'sensores_planta.csv'.")
        sys.exit(1)

    df.columns = df.columns.str.strip().str.lower()
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        print(f"Error: Faltan columnas requeridas: {faltantes}")
        sys.exit(1)

    df = df.dropna(subset=COLUMNAS_REQUERIDAS)
    print(f"Archivo cargado correctamente. Registros totales: {len(df)}")
    print(f"Columnas detectadas: {df.columns.tolist()}")

    X = df[["temperatura_c", "vibracion_mms", "horas_operacion", "presion_psi"]]
    y = df["estado"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("2. Entrenando el algoritmo RandomForestClassifier...")
    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo (Accuracy): {precision * 100:.2f}%")

    if precision < 0.80:
        print("Advertencia: La precisión es menor a 80%. Revisa el dataset o el modelo.")
        sys.exit(1)

    joblib.dump(modelo, RUTA_MODELO)
    print(f"3. Éxito! Modelo exportado como '{RUTA_MODELO.name}'.")


if __name__ == "__main__":
    main()
