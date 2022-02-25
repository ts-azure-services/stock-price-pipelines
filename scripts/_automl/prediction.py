from azureml.core.model import Model
from azureml.core import Dataset
import joblib
import sys
import os.path
from authentication import ws

def load_best_model(model_name=None, version=None):
    final_model = Model(workspace=ws,name=model_name,version=version)
    final_model.download('./model/', exist_ok=True)
    model = joblib.load('./model/model.pkl')
    return model

def load_test_dataset(dataset_name=None, target_column=None):
    ds = Dataset.get_by_name(workspace=ws, name=dataset_name)
    y_test = ds.to_pandas_dataframe().reset_index(drop=True)
    y_test_values = y_test.pop(target_column).values
    return y_test, y_test_values

# Predict values from test dataset inputs
def predict_values(
        model=None, 
        y_test=None, 
        y_test_values=None, 
        target_column=None
        ):
    quantiles = [0.025, 0.5, 0.95]
    predicted_column_name = 'predicted'
    PI = 'prediction_interval'
    model.quantiles = quantiles
    pred_quantiles = model.forecast_quantiles(y_test)
    pred_quantiles[PI] = pred_quantiles[\
            [min(quantiles), max(quantiles)]].\
            apply(lambda x: '[{}, {}]'.format(x[0],x[1]), axis=1)
    y_test[target_column] = y_test_values
    y_test[PI] = pred_quantiles[PI]
    y_test[predicted_column_name] = pred_quantiles[0.5]
    clean = y_test[y_test[[target_column,predicted_column_name]].notnull().all(axis=1)]
    clean.to_csv('./model/prediction.csv', header=True, index=False)

def main():

    # Specify the specific registered model, and the test data to use
    model_name, version = ('stock_price_model','3')
    dataset_name, target_column = ('Test set for FB', 'close')

    # Download and load the model
    model = load_best_model(model_name=model_name, version=version)

    # Load the test dataset
    y_test, y_test_values = load_test_dataset(dataset_name=dataset_name, target_column=target_column)

    # Predict using the model
    predict_values(model=model, y_test=y_test, y_test_values=y_test_values, target_column=target_column)

if __name__ == "__main__":
    main() 
