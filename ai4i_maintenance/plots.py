"""Gráficas de evaluación que se registran como artefactos en MLflow."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: las gráficas se guardan como archivo, no se muestran

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

from ai4i_maintenance.config import BLUE

CLASS_LABELS = ["Sin falla (0)", "Con falla (1)"]


def plot_confusion_matrix(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, path: Path) -> None:
    """Matriz de confusión con el nombre de cada celda (TN, FP, FN, TP) junto al conteo."""
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay.from_estimator(
        pipeline, X, y, display_labels=CLASS_LABELS, cmap="viridis", colorbar=False, ax=ax
    )
    cell_names = [["TN", "FP"], ["FN", "TP"]]
    for row in range(2):
        for col in range(2):
            text = display.text_[row, col]
            text.set_text(f"{cell_names[row][col]}\n{text.get_text()}")

    ax.set_title("Matriz de confusión (conjunto de prueba)")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_feature_importance(importances: pd.Series, path: Path) -> None:
    """Barras horizontales con la importancia de cada feature, de menor a mayor."""
    importances = importances.sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importances.index, importances.values, color=BLUE)
    for i, value in enumerate(importances.values):
        ax.text(value, i, f" {value:.3f}", va="center")

    ax.set_xlim(0, importances.max() * 1.15)  # espacio para la etiqueta de la barra más larga
    ax.set_title("Importancia de las features")
    ax.set_xlabel("Importancia")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
