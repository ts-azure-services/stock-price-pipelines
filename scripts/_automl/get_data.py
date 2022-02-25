"""Retrieve stock price data"""
import logging
import datetime as dt
from datetime import datetime
import pandas_datareader as web
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def get_stock_data(symbol=None):
    """Return stock price history for given symbol"""
    data = None
    try:
        # Load stock price data from specific companies
        company = [symbol]
        start = dt.datetime(2019,1,1)
        #end = dt.datetime(2021,2,10)
        end = datetime.now()
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

if __name__ == "__main__":
    #company = ['FB', 'MSFT', 'AAPL']
    stock_history = get_stock_data(symbol='FB')
    stock_history.to_csv('./datasets/stock_data.csv', index=False, encoding='utf-8')
    print(stock_history)
