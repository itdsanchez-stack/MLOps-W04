"""Carga, valida y limpia el dataset AI4I 2020: data/raw/ -> data/processed/.

Uso (desde la raíz del repositorio):
    python -m ai4i_maintenance.dataset
"""

import argparse
from pathlib import Path

import pandas as pd

from ai4i_maintenance.config import (
    CATEGORICAL_FEATURES,
    EXPECTED_COLUMNS,
    FAILURE_MODES,
    ID_COLUMNS,
    NUMERIC_FEATURES,
    PROCESSED_DATA_FILE,
    RAW_DATA_FILE,
    TARGET,
)


def load_raw_data(path: Path) -> pd.DataFrame:
    """Lee el CSV original y valida que tenga exactamente las columnas documentadas."""
    df = pd.read_csv(path)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Schema validation: detiene el proceso si falta una columna o aparece una no esperada."""
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    extra_columns = set(df.columns) - set(EXPECTED_COLUMNS)
    if missing_columns or extra_columns:
        raise ValueError(
            f"Schema validation falló. Faltan: {missing_columns or '-'} | "
            f"No esperadas: {extra_columns or '-'}"
        )


def find_inconsistent_labels(df: pd.DataFrame) -> pd.Series:
    """Marca los registros que rompen la regla documentada del dataset.

    Regla: Machine failure = 1 si y solo si ocurre al menos un modo de falla.
    """
    n_active_modes = df[FAILURE_MODES].sum(axis=1)
    return ((df[TARGET] == 1) & (n_active_modes == 0)) | ((df[TARGET] == 0) & (n_active_modes > 0))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina etiquetas inconsistentes, identificadores y modos de falla (data leakage)."""
    df_clean = df[~find_inconsistent_labels(df)].copy()

    if find_inconsistent_labels(df_clean).any():
        raise ValueError("Persisten etiquetas inconsistentes después de la limpieza")

    df_clean = df_clean.drop(columns=ID_COLUMNS + FAILURE_MODES)

    expected_output = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET]
    if df_clean.columns.tolist() != expected_output:
        raise ValueError(f"Columnas inesperadas en el dataset limpio: {df_clean.columns.tolist()}")
    return df_clean


def save_processed_data(df: pd.DataFrame, path: Path) -> None:
    """Guarda el dataset limpio, creando la carpeta de destino si no existe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Limpia el dataset AI4I 2020 y lo guarda en data/processed/."
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=RAW_DATA_FILE,
        help=f"CSV original de entrada (default: {RAW_DATA_FILE.name})",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROCESSED_DATA_FILE,
        help=f"CSV limpio de salida (default: {PROCESSED_DATA_FILE.name})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df_raw = load_raw_data(args.raw_path)
    print(f"Schema validation OK: {len(EXPECTED_COLUMNS)} columnas esperadas")

    df_clean = clean_data(df_raw)
    save_processed_data(df_clean, args.output_path)

    print(
        f"Registros       : {len(df_raw)} ---> {len(df_clean)} "
        f"({find_inconsistent_labels(df_raw).sum()} inconsistentes eliminados)"
    )
    print(f"Fallas          : {df_raw[TARGET].sum()} ---> {df_clean[TARGET].sum()}")
    print(f"Tasa de falla   : {df_raw[TARGET].mean():.2%} ---> {df_clean[TARGET].mean():.2%}")
    print(f"Columnas finales: {df_clean.columns.tolist()}")
    print(f"Dataset limpio guardado en: {args.output_path}")


if __name__ == "__main__":
    main()
