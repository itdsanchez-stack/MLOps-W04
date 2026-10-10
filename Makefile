.PHONY: requirements data train experiments mlflow-server lint format

requirements:
	pip install -r requirements.txt

data:
	python -m ai4i_maintenance.dataset

train: data
	python -m ai4i_maintenance.modeling.train

experiments: data
	python -m ai4i_maintenance.modeling.train
	python -m ai4i_maintenance.modeling.train --max-depth 5
	python -m ai4i_maintenance.modeling.train --max-depth 8
	python -m ai4i_maintenance.modeling.train --max-depth 12
	python -m ai4i_maintenance.modeling.train --min-samples-leaf 5
	python -m ai4i_maintenance.modeling.train --min-samples-leaf 20
	python -m ai4i_maintenance.modeling.train --criterion entropy
	python -m ai4i_maintenance.modeling.train --class-weight balanced
	python -m ai4i_maintenance.modeling.train --criterion entropy --max-depth 10 --min-samples-leaf 2
	python -m ai4i_maintenance.modeling.train --criterion entropy --max-depth 10 --min-samples-leaf 2 --run-name reproducibility_check

mlflow-server:
	mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

lint:
	ruff format --check
	ruff check

format:
	ruff check --fix
	ruff format
