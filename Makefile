install:
	#pip install --upgrade pip && pip install -r requirements.txt
	pip install --upgrade pip
	pip install python-dotenv
	pip install azureml-core
	pip install pandas-datareader
	pip install azureml-dataset-runtime
	pip install azureml.pipeline

setup_infra:
	./scripts/setup/create-aml-infra.sh
	python ./scripts/setup/clusters.py
	python ./scripts/setup/build_env.py

run_pipeline:
	python ./scripts/existing_model/train_pipeline.py

lint:
	pylint --disable=C0103,E0110,W1203,W0703 ./scripts/automl/train.py
	pylint --disable=C0103,E0110,W1203,W0703 ./scripts/automl/get_data.py
