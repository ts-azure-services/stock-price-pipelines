import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
import argparse
import os
#import datetime as dt
import sys
import os.path
from azureml.core import Dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
from scripts.authentication.service_principal import ws

def getArgs(argv=None):
    parser = argparse.ArgumentParser(description="filepaths")
    parser.add_argument("--stock_symbol", help='Stock symbol', required=True)
    parser.add_argument("--input_filepath", help='Input file path', required=True)
    parser.add_argument("--input_filename", help='Input file name', required=True)
    return parser.parse_args(argv)


def model_train(source=None, company=None):
    start_time = time.time()
    full_data = pd.read_csv(source)
    #ds = Dataset.get_by_id(ws, id=source)
    #data = ds.to_pandas_dataframe()

    # Set train/test split
    data = full_data[:-20].copy()
    test_data = full_data[-20:].copy()

    # Prepare data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1,1))
    prediction_days = 10 # how many days do I want to base my prediction on, how many to look back on

    x_train = []
    y_train = []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append( scaled_data[x-prediction_days:x,0] )
        y_train.append( scaled_data[x, 0] )

    x_train, y_train = np.array( x_train ), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1) )

    # Build the model
    model = Sequential()
    model.add(LSTM(units=500, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=500, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=500, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=500))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) # prediction of the next closing value
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=100, batch_size=32)

    # Test model accuracy on existing data
    actual_prices = test_data['close'].values
    total_dataset = pd.concat((data['close'], test_data['close']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1,1)
    model_inputs = scaler.transform(model_inputs)

    # Make predictions on test data
    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append( model_inputs[x-prediction_days:x, 0] )
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predict price (scaled)
    predicted_prices = model.predict(x_test)

    # Reverse scale prices
    predicted_prices = scaler.inverse_transform(predicted_prices)
    logging.info(f"Predicted prices: {predicted_prices}")

    # Plot test predictions
    def plotting_code(company=None):
        plt.plot(actual_prices, color='black', label=f'Actual {company} price')
        plt.plot(predicted_prices, color='green', label=f'Predicted {company} price')
        plt.title(f"{company} Share Price")
        plt.xlabel("Time")
        plt.ylabel(f"{company} Share Price")
        plt.legend()
        ts = dt.datetime.now().strftime('%s')
        plt.savefig(f'./outputs/{company}_prediction' + str(ts) + '.png')

    plotting_code(company=company)

    end_time = time.time()
    logging.info("The time of execution of above program is:", end_time-start_time)

if __name__ == "__main__":
    args = getArgs()
    logging.info(f'Stock ticker is: {args.stock_symbol}')
    logging.info(f'Input file path is: {args.input_filepath}')
    logging.info(f'Input file name is: {args.input_filename}')
    model_train(source=args.input_filepath + '/' + args.input_filename, company=args.stock_symbol)
