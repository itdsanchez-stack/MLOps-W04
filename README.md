# Mantenimiento predictivo AI4I 2020 con MLflow

Predice si una máquina falla (`Machine failure`) a partir de sus condiciones de operación,
con un `DecisionTreeClassifier`. Cada entrenamiento se registra en MLflow.

TC5061, Semana 4 · Daniel Sánchez Hernández

## 1. Preparar el ambiente (solo la primera vez)

```bash
git clone https://github.com/itdsanchez-stack/MLOps-W04.git
cd MLOps-W04
conda create -n mlops python=3.11 -y
conda activate mlops
pip install -r requirements.txt
```

Todos los comandos se corren desde la raíz del repositorio con el ambiente `mlops` activo.

## 2. Levantar MLflow

En una terminal aparte:

```bash
make mlflow-server
```

La UI queda en http://127.0.0.1:5000. Si el servidor no está activo, el script lo avisa y guarda
las corridas en `mlflow.db`; se pueden ver después levantando el servidor.

## 3. Correr el proyecto

```bash
python -m ai4i_maintenance.dataset          # limpia data/raw/ y guarda en data/processed/
python -m ai4i_maintenance.modeling.train   # entrena con los valores por defecto (baseline)
```

Los hiperparámetros se cambian sin editar el código:

```bash
python -m ai4i_maintenance.modeling.train --max-depth 8 --min-samples-leaf 5 --criterion entropy
python -m ai4i_maintenance.modeling.train --help   # todos los argumentos y sus valores por defecto
```

`make experiments` corre las 10 configuraciones usadas en el análisis.

## 4. Correr como MLflow Project

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000              # si no, usa mlflow.db local
export MLFLOW_EXPERIMENT_NAME=AI4I_PredictiveMaintenance_DecisionTree
mlflow run . --env-manager local
mlflow run . --env-manager local -P max_depth=8 -P min_samples_leaf=5
```

Al final debe decir `=== Run (ID '...') succeeded ===`.

## 5. Comparar resultados

En la UI, abrir el experimento `AI4I_PredictiveMaintenance_DecisionTree`, seleccionar las corridas
y dar clic en **Compare**. Las configuraciones se comparan con `cv_f1_mean` (cross-validation sobre
entrenamiento); el conjunto de prueba se reserva para la evaluación final.

## Reproducibilidad

- Random seed fija (`SEED = 42` en `config.py`) en el split, el árbol y la cross-validation.
- Versiones fijas en `requirements.txt`.
- `make experiments` repite una configuración (`reproducibility_check`) para comprobar que da lo mismo.

## Estructura

```
├── MLproject                  # entry point de mlflow run
├── python_env.yaml            # ambiente del MLproject
├── Makefile                   # make data, train, experiments, mlflow-server, lint
├── requirements.txt
├── data/raw/ai4i2020.csv      # dataset original
├── notebooks/
│   └── 1.0-dsh-ai4i-eda.ipynb # EDA y justificación de la limpieza de datos
└── ai4i_maintenance/
    ├── config.py              # rutas, columnas y constantes
    ├── dataset.py             # carga, schema validation y limpieza
    ├── features.py            # X/y y preprocesamiento
    ├── plots.py               # matriz de confusión e importancia de features
    └── modeling/train.py      # entrena y registra la corrida en MLflow
```
