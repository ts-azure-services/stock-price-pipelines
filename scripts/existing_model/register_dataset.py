"""Retrieve stock price data"""
import logging
import argparse
import datetime as dt
from datetime import datetime
import pandas_datareader as web
from azureml.core import Dataset
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
from scripts.authentication.service_principal import ws
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def getArgs(argv=None):
    parser = argparse.ArgumentParser(description="filepaths")
    parser.add_argument("--stock_symbol", help='Stock symbol', required=True)
    parser.add_argument("--start_date_arg", help='Start date for pulling stock history', required=True)
    parser.add_argument("--end_date_arg", help='End date for pulling stock history', required=True)
    parser.add_argument("--output_filepath", help='Output filepath', required=True)
    parser.add_argument("--output_filename", help='Output filename', required=True)
    return parser.parse_args(argv)

def get_stock_data(symbol=None, start=None, end=None):
    """Return stock price history for given symbol"""
    data = None
    try:
        # Load stock price data from specific companies
        company = [symbol]
        #start = dt.datetime(2019,1,1)
        #end = dt.datetime(2021,2,17)
        #end = datetime.now()
        for i in company:
            data = web.DataReader(i, 'yahoo', start, end) # returns a df
            data = data.reset_index()
            data.columns=data.columns.str.lower()
            data.columns=data.columns.str.replace(' ','')
            #data = data.to_json() # converts to a string
        logging.info('Successfully received stock price data.')
    except Exception as e:
        logging.warning(f'Exception: {e}. Failed to load stock price data.')
    finally:
        if data is None:
            final_value = 'No data returned.'
        else:
            final_value = data
    return final_value

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
    #full_dataset = full_dataset.with_timestamp_columns('date')


def register_datasets(df=None,full_dataset_name=None):
    """Register the full dataset, including the training/test set"""
    #df = pd.read_csv(source)
    #train = df[:-20].copy()
    #test = df[-20:].copy()
    def_blob_store = ws.get_default_datastore()
    pd_dataframe_register(df=df, def_blob_store=def_blob_store, name=full_dataset_name)
    #pd_dataframe_register(df=train, def_blob_store=def_blob_store,name=training_set_name)
    #pd_dataframe_register(df=test, def_blob_store=def_blob_store, name=test_set_name)

if __name__ == "__main__":
    #company = ['FB', 'MSFT', 'AAPL']
    args = getArgs()
    #start = dt.datetime(2019,1,1)
    #end = dt.datetime(2021,2,17)
    start = datetime.strptime(args.start_date_arg, "%Y-%m-%d")
    end = datetime.strptime(args.end_date_arg, "%Y-%m-%d")
    stock_history = get_stock_data(symbol=args.stock_symbol, start=start, end=end)
    logging.info(f"Start date for {args.stock_symbol} is: {args.start_date_arg}")
    logging.info(f"End date for {args.stock_symbol} is: {args.end_date_arg}")
    logging.info(f"First five rows for {args.stock_symbol} is:\n {stock_history.head()}")
    logging.info(f"Last five rows for {args.stock_symbol} is:\n {stock_history.tail()}")
    register_datasets(df=stock_history,full_dataset_name=str(args.stock_symbol) + '-stock-data')
    stock_history.to_csv(args.output_filepath +'/'+ args.output_filename, index=False,
            encoding='utf-8')
