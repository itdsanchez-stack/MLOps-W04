"""Separación de features/target y preprocesamiento de las features."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from ai4i_maintenance.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa el dataset limpio en features (X) y target (y)."""
    return df.drop(columns=[TARGET]), df[TARGET]


def build_preprocessor() -> ColumnTransformer:
    """One-hot encoding para Type; las numéricas no se escalan porque un árbol no lo necesita."""
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
