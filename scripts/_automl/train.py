"""Full training script"""
import logging
from azureml.core import Dataset#, ScriptRunConfig, Environment
from azureml.core.compute import ComputeTarget
from azureml.core.experiment import Experiment
from azureml.automl.core.forecasting_parameters import ForecastingParameters
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun
import pandas as pd
from authentication import ws
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def pd_dataframe_register(
        df=None,
        def_blob_store=None,
        name=None,
        desc=None):
    """Register pandas dataframe"""
    full_dataset = Dataset.Tabular.register_pandas_dataframe(
            dataframe=df,
            target=def_blob_store,
            name=name,
            description=desc
            )
    full_dataset = full_dataset.with_timestamp_columns('date')


def create_register_datasets(source=None,full_dataset_name=None,
        training_set_name=None,
        test_set_name=None
        ):
    """Register the full dataset, including the training/test set"""
    df = pd.read_csv(source)
    train = df[:-10]
    test = df[-10:]
    def_blob_store = ws.get_default_datastore()
    pd_dataframe_register(df=df, def_blob_store=def_blob_store, name=full_dataset_name)
    pd_dataframe_register(df=train, def_blob_store=def_blob_store,name=training_set_name)
    pd_dataframe_register(df=test, def_blob_store=def_blob_store, name=test_set_name)


def model_train(dataset=None, compute_target=None, experiment_name=None):
    """Model and train with AutoML the dataset"""

    forecasting_parameters = ForecastingParameters(
        time_column_name='date',
        forecast_horizon=12,
        target_rolling_window_size=3, # for simple regression, comment this
        feature_lags='auto',# for simple regression, comment this
        target_lags=12,# for simple regression, comment this
        freq='D',
        validate_parameters=True)

    # Setup the classifier
    automl_settings = {
        "task": 'forecasting',
        "primary_metric":'normalized_root_mean_squared_error',
        "iteration_timeout_minutes": 30,
        "experiment_timeout_hours": 1.0,
        "compute_target":compute_target,
        "max_concurrent_iterations": 4,
        "featurization": "off",
        #"allowed_models":['AutoArima', 'Prophet'],
        #"blocked_models":['XGBoostClassifier'],
        #"verbosity": logging.INFO,
        "training_data":dataset,#.as_named_input('retrain_dataset'),
        "label_column_name":'close',
        "n_cross_validations": 3,
        "enable_voting_ensemble":True,
        "enable_early_stopping": False,
        "model_explainability":True,
        "enable_dnn":True,
        "forecasting_parameters": forecasting_parameters
            }

    automl_config = AutoMLConfig(**automl_settings)
    experiment = Experiment(ws, experiment_name)
    remote_run = experiment.submit(automl_config, show_output=True, wait_post_processing=True)
    remote_run.wait_for_completion()
    logging.info(f'Run details: {remote_run}')

    # Convert to AutoMLRun object
    remote_run = AutoMLRun(experiment, run_id=remote_run.id)
    return remote_run

def register_best_model(
        remote_run=None,
        model_name=None,
        model_path=None,
        description=None
        ):
    """Register the best model from the AutoML Run"""
    best_child = remote_run.get_best_child()
    model = best_child.register_model(
            model_name = model_name,
            model_path = model_path,
            description = description,
            )
    logging.info(f"Registered {model_name}, with {description}")
    return model

def main():
    """Main operational flow"""
    # Declare key objects
    ticker = 'FB'
    full_dataset_name = 'Full dataset for ' + str(ticker)
    training_set_name = 'Training set for ' + str(ticker)
    test_set_name = 'Test set for ' + str(ticker)
    experiment_name = 'stock_price_timeseries'
    compute_target = ComputeTarget(workspace=ws, name='cpu-cluster')

    create_register_datasets(source='./datasets/stock_data.csv',
            full_dataset_name= full_dataset_name,
            training_set_name= training_set_name,
            test_set_name = test_set_name
            )

    # Train the model
    ds = Dataset.get_by_name(workspace=ws, name=training_set_name)
    remote_run = model_train(
            dataset=ds,
            compute_target= compute_target,
            experiment_name=experiment_name)

    # Register the best model
    register_best_model(
            remote_run = remote_run,
            model_name='stock_price_model',
            model_path='outputs/model.pkl',
            description='Stock price model'
            )


if __name__ == "__main__":
    main()
