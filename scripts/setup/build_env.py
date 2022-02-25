"""Build the desired software environment"""
import sys
import os
import os.path
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
from scripts.authentication.service_principal import ws

def build_env():
    # Conda dependencies
    conda_dep = CondaDependencies()
    conda_dep.add_pip_package("tensorflow")
    conda_dep.add_pip_package("pandas-datareader")
    conda_dep.add_pip_package("scikit-learn")
    conda_dep.add_pip_package("python-dotenv")
    conda_dep.add_pip_package("matplotlib")

    # Env specifications
    baseline_env = Environment(name="model_env")#, file_path='./requirements.txt')
    baseline_env.python.conda_dependencies=conda_dep
    #baseline_env = Environment.from_pip_requirements(name="for_some", file_path='./requirements.txt')
    baseline_env.docker.base_image = None
    baseline_env.docker.base_dockerfile = "./scripts/setup/Dockerfile"

    # Register the environment
    baseline_env.register(workspace=ws)

    # Build the env
    build = baseline_env.build(workspace=ws)
    build.wait_for_completion(show_output=True)

if __name__ == "__main__":
    build_env()
