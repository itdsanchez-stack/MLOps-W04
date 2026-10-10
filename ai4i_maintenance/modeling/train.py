"""Entrena un Decision Tree para predecir fallas de máquina y registra la corrida en MLflow.

Uso (desde la raíz del repositorio):
    python -m ai4i_maintenance.modeling.train
    python -m ai4i_maintenance.modeling.train --max-depth 5 --min-samples-leaf 10
    python -m ai4i_maintenance.modeling.train --help
"""

import argparse
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.request

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ai4i_maintenance.config import (
    CV_FOLDS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_FALLBACK_URI,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_FILE,
    SEED,
    TEST_SIZE,
)
from ai4i_maintenance.features import build_preprocessor, split_features_target
from ai4i_maintenance.plots import CLASS_LABELS, plot_confusion_matrix, plot_feature_importance


def optional_int(value: str) -> int | None:
    """Convierte el texto a entero; 'None' significa sin límite (default de scikit-learn)."""
    return None if value.lower() == "none" else int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un DecisionTreeClassifier sobre el dataset AI4I 2020 "
        "limpio y registra parámetros, métricas, artefactos y modelo "
        "en MLflow. Sin argumentos reproduce el modelo baseline."
    )
    parser.add_argument(
        "--max-depth",
        type=optional_int,
        default=None,
        help="Profundidad máxima del árbol; 'None' = sin límite (default: None)",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Mínimo de registros en cada hoja; valores altos simplifican el árbol (default: 1)",
    )
    parser.add_argument(
        "--criterion",
        choices=["gini", "entropy", "log_loss"],
        default="gini",
        help="Medida de impureza para elegir cada corte (default: gini)",
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="'balanced' da más peso a la clase minoritaria (fallas) (default: none)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed del split, el árbol y la cross-validation (default: {SEED})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help=f"Proporción del conjunto de prueba (default: {TEST_SIZE})",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROCESSED_DATA_FILE,
        help="CSV limpio generado por ai4i_maintenance.dataset "
        f"(default: data/processed/{PROCESSED_DATA_FILE.name})",
    )
    parser.add_argument(
        "--experiment-name",
        default=MLFLOW_EXPERIMENT_NAME,
        help=f"Experimento de MLflow que agrupa las corridas (default: {MLFLOW_EXPERIMENT_NAME})",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nombre de la corrida; si se omite se arma con los hiperparámetros",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI),
        help=f"Servidor de MLflow (default: {MLFLOW_TRACKING_URI}); si no responde, se usa "
        "mlflow.db local",
    )
    return parser.parse_args()


def load_processed_data(path: Path) -> pd.DataFrame:
    """Lee el dataset limpio; si no existe, indica cómo generarlo."""
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Genera primero el dataset limpio con: "
            "python -m ai4i_maintenance.dataset"
        )
    return pd.read_csv(path)


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, seed: int) -> tuple:
    """Train/test split estratificado por el target."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    """Pipeline de preprocesamiento + Decision Tree."""
    model = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        criterion=args.criterion,
        class_weight=None if args.class_weight == "none" else args.class_weight,
        random_state=args.seed,
    )
    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Accuracy, precision, recall y F1 sobre la clase falla (1)."""
    y_pred = pipeline.predict(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
    }


def cross_validate_f1(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, seed: int
) -> tuple[float, float]:
    """F1 con stratified K-fold sobre train.

    Compara configuraciones sin usar el test set, que se reserva para la evaluación final.
    """
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    scores = cross_val_score(pipeline, X_train, y_train, cv=folds, scoring="f1")
    return scores.mean(), scores.std()


def resolve_tracking_uri(uri: str) -> str:
    """Usa el servidor de MLflow si responde; si no, cae al archivo local mlflow.db."""
    if not uri.startswith("http"):
        return uri
    try:
        urllib.request.urlopen(f"{uri}/health", timeout=3)
        return uri
    except (urllib.error.URLError, OSError):
        print(
            f"AVISO: no se encontró el servidor de MLflow en {uri}.\n"
            f"       Las corridas se guardarán en {MLFLOW_FALLBACK_URI}"
        )
        return MLFLOW_FALLBACK_URI


EXPERIMENT_DESCRIPTION = (
    "Clasificación binaria de fallas de máquina (AI4I 2020) con DecisionTreeClassifier. "
    "Cada corrida varía hiperparámetros del árbol; se comparan con F1 de stratified "
    "cross-validation sobre train (cv_f1_mean) y se reportan métricas en test."
)


def default_run_name(args: argparse.Namespace) -> str:
    return (
        f"dt_depth-{args.max_depth}_leaf-{args.min_samples_leaf}_"
        f"{args.criterion}_cw-{args.class_weight}"
    )


def log_run(args, pipeline, X_train, X_test, y_test, params, metrics, data_path) -> str:
    """Registra parámetros, métricas, tags, artefactos y modelo en una corrida de MLflow."""
    tree = pipeline.named_steps["model"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    with mlflow.start_run(run_name=args.run_name or default_run_name(args)) as run:
        for name, value in params.items():
            mlflow.log_param(name, value)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        mlflow.set_tags(
            {
                "model_type": "DecisionTreeClassifier",
                "dataset": "AI4I 2020 Predictive Maintenance (limpio)",
                "author": "Daniel Sánchez Hernández",
            }
        )

        # Carpeta temporal: MLflow guarda su copia y no quedan archivos sueltos en el repo
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plot_confusion_matrix(pipeline, X_test, y_test, tmp / "confusion_matrix.png")
            plot_feature_importance(
                pd.Series(tree.feature_importances_, index=feature_names),
                tmp / "feature_importance.png",
            )
            (tmp / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
            (tmp / "classification_report.txt").write_text(
                classification_report(
                    y_test, pipeline.predict(X_test), target_names=CLASS_LABELS, digits=4
                )
            )

            mlflow.log_artifact(str(tmp / "confusion_matrix.png"), artifact_path="plots")
            mlflow.log_artifact(str(tmp / "feature_importance.png"), artifact_path="plots")
            mlflow.log_artifact(str(tmp / "metrics.json"), artifact_path="metrics")
            mlflow.log_artifact(str(tmp / "classification_report.txt"), artifact_path="metrics")
        mlflow.log_artifact(str(data_path), artifact_path="data")

        # skops, el formato de MLflow 3, rechaza por seguridad los tipos que no se declaran
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=infer_signature(X_train, pipeline.predict(X_train)),
            input_example=X_train.head(5),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
    return run.info.run_id


def main() -> None:
    args = parse_args()

    df = load_processed_data(args.data_path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, args.test_size, args.seed)

    pipeline = build_pipeline(args)
    cv_f1_mean, cv_f1_std = cross_validate_f1(pipeline, X_train, y_train, args.seed)
    pipeline.fit(X_train, y_train)
    tree = pipeline.named_steps["model"]

    params = {
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "criterion": args.criterion,
        "class_weight": args.class_weight,
        "seed": args.seed,
        "test_size": args.test_size,
        "cv_folds": CV_FOLDS,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    test_metrics = evaluate(pipeline, X_test, y_test)
    train_metrics = evaluate(pipeline, X_train, y_train)
    metrics = {
        **{f"test_{name}": value for name, value in test_metrics.items()},
        **{f"train_{name}": value for name, value in train_metrics.items()},
        "cv_f1_mean": cv_f1_mean,
        "cv_f1_std": cv_f1_std,
        "tree_depth": tree.get_depth(),
        "tree_n_leaves": tree.get_n_leaves(),
    }

    mlflow.set_tracking_uri(resolve_tracking_uri(args.tracking_uri))
    # Con `mlflow run` el experimento ya viene definido por la línea de comandos
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment(args.experiment_name)
        mlflow.set_experiment_tags(
            {
                "mlflow.note.content": EXPERIMENT_DESCRIPTION,
                "project": "TC5061 MLOps",
                "dataset": "AI4I 2020",
            }
        )
    run_id = log_run(args, pipeline, X_train, X_test, y_test, params, metrics, args.data_path)

    print(f"\nRun ID      : {run_id}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Árbol       : profundidad {metrics['tree_depth']}, {metrics['tree_n_leaves']} hojas")
    print(f"CV F1 (train): {cv_f1_mean:.4f} +/- {cv_f1_std:.4f}")
    print(pd.DataFrame({"test": test_metrics, "train": train_metrics}).round(4).to_string())


if __name__ == "__main__":
    main()
