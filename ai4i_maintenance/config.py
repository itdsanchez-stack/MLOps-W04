"""Rutas, columnas y constantes del proyecto. Ningún otro módulo escribe una ruta."""

from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "ai4i2020.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "ai4i2020_clean.csv"

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Columnas documentadas por el autor del dataset (Matzka, 2020)
EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]
TARGET = "Machine failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
ID_COLUMNS = ["UDI", "Product ID"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]

# Si el servidor no responde, train.py guarda las corridas en mlflow.db local
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_FALLBACK_URI = f"sqlite:///{PROJ_ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "AI4I_PredictiveMaintenance_DecisionTree"

# Paleta apta para daltonismo
BLUE = "#2a78d6"
ORANGE = "#eb6834"
YELLOW = "#eda100"
