# -*- coding: utf-8 -*-
from os import path
import numpy as np
import pandas as pd

def load_index_data():
    """Loads an index of prices composed by several major cryptocurrencies.

    The index includes Bitcoin, Ethereum, Litecoin, BNB, Cardano, XRP. Index
    prices are weighted on market capitalization.

    Data are stored in a local CSV file.

    Method returns two arrays with time and log-price values.

    """

    filepath = path.join(path.dirname(__file__), 'cryptoIndex.csv')

    # Load crypto index with labels.
    df = pd.read_csv(filepath, sep = ',', index_col = 'date', parse_dates = True)

    # Time index.
    time = df.index

    # Convert to log the close prices.
    price = np.log(df['close'])

    return time, price
