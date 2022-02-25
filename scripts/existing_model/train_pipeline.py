from azureml.core import Dataset, Environment, Workspace, ScriptRunConfig
from azureml.core.experiment import Experiment
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.pipeline.core.graph import PipelineParameter
from azureml.core.conda_dependencies import CondaDependencies
from azureml.data import OutputFileDatasetConfig
import logging
import sys
import os
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
from scripts.authentication.service_principal import ws
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.core.runconfig import DockerConfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


def main():

    # Create pipeline parameter for the stock symbol
    pipeline_param = PipelineParameter(name='stock_ticker', default_value='FB')
    start_date_param = PipelineParameter(name='start_date_param', default_value="2020-01-01")
    end_date_param = PipelineParameter(name='end_date_param', default_value="2020-03-01")

    def_blob_store = ws.get_default_datastore()
    compute_target = ComputeTarget(workspace=ws, name='cpu-cluster')
    env= Environment.get(workspace=ws, name='model_env')
    experiment = Experiment(ws, 'STOCK_PRICE')
    intermediate_source = OutputFileDatasetConfig(destination=(def_blob_store,'/inter/')).as_mount()
    intermediate_filename = 'stock_data.csv'

    run_config = RunConfiguration()
    run_config.environment = env

    first_step = PythonScriptStep(
            name='Pull data and register dataset',
            source_directory='.',
            script_name='./scripts/existing_model/register_dataset.py',
            compute_target=compute_target,
            arguments=[
                '--stock_symbol', pipeline_param,
                '--start_date_arg', start_date_param,
                '--end_date_arg', end_date_param,
                '--output_filepath', intermediate_source,
                '--output_filename', intermediate_filename,
                ],
            runconfig=run_config,
            allow_reuse=False
            )

    second_step = PythonScriptStep(
            name='Train with model and predict',
            source_directory='.',
            script_name='./scripts/existing_model/model_train.py',
            compute_target=compute_target,
            arguments=[
                #'--input_file_path', ds.as_named_input('baseline_raw_input')#.as_mount(),
                '--stock_symbol', pipeline_param,
                '--input_filepath', intermediate_source.as_input(),
                '--input_filename', intermediate_filename
                ],
            runconfig=run_config,
            allow_reuse=False
            )

    # Pipeline integration
    steps = [ first_step, second_step]
    pipeline = Pipeline(workspace=ws, steps=steps)
    pipeline_run = experiment.submit(pipeline, pipeline_parameters={
        "stock_ticker":"AAPL",
        "start_date_param":"2020-01-01",
        "end_date_param":"2021-02-14"
        })
    pipeline_run.wait_for_completion()

    ## Publish the pipeline
    published_pipeline = pipeline_run.publish_pipeline(
            name='Stock price pipeline (units 500)',
            description='Pipeline to run time series prediction for given stock ticker',
            version='1.0'
            )

if __name__ == "__main__":
    main()
