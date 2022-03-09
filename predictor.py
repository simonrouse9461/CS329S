from pmdarima.arima import auto_arima
import numpy as np
import pandas as pd
from prophet import Prophet


def get_log_return(data):
    data["return"] = data["price"].apply(np.log) - data["price"].shift(1).apply(np.log)
    return data.dropna()


def get_bollinger_bands(prices, window):
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    bollinger_top = sma + std * 2
    bollinger_bottom = sma - std * 2
    return bollinger_top, bollinger_bottom


class Arima:

    def __init__(self, start=0, max_search=10, seasonality=True, m=10):
        self.start = start
        self.max = max_search
        self.seasonality = seasonality
        self.m = m 

    def fit(self, data):
        data = get_log_return(data)
        self.base_date = data['date'].iloc[-1]
        self.base_price = data['price'].iloc[-1]
        self.model = auto_arima(data['return'], 
                                start_p=self.start, d=self.start, start_q=self.start, 
                                max_p=self.max, max_d=self.max, max_q=self.max, 
                                start_P=self.start, start_D=self.start, start_Q=self.start, 
                                max_P=self.max, max_D=self.max, max_Q=self.max, 
                                m=self.m, seasonality=self.seasonality) 
        return self

    def predict(self, period):
        predicted_log_return = pd.DataFrame(self.model.predict(n_periods=period)).rename(columns={0: 'return'})
        predicted_log_return['price'] = np.nan

        predicted_log_return['price'].iloc[0] = np.exp(predicted_log_return['return'].iloc[0]) * self.base_price
        for i in range(1, len(predicted_log_return)):
            predicted_log_return['price'].iloc[i] = np.exp(predicted_log_return['return'].iloc[i]) * predicted_log_return['price'].iloc[i - 1]

        return pd.DataFrame({"date": pd.date_range(self.base_date, periods=period+1), 
                             "price": [self.base_price] + predicted_log_return['price'].to_list()})


class DeepProphet:

    def __init__(self):
        self.model = Prophet(yearly_seasonality='auto', weekly_seasonality='auto', daily_seasonality='auto')

    def fit(self, data):
        self.base_date = data['date'].iloc[-1]
        self.base_price = data['price'].iloc[-1]
        data = get_log_return(data)
        data = data.reset_index().rename(columns={'date' : 'ds', 'return' : 'y'})
        self.model.fit(data)
        return self

    def predict(self, period):
        future = self.model.make_future_dataframe(period)
        forecast = self.model.predict(future)
        predicted_log_return = forecast[['yhat']].iloc[-period:]
        predicted_log_return['price'] = np.nan
        predicted_log_return.rename(columns={'yhat': 'return'}, inplace=True)

        predicted_log_return['price'].iloc[0] = np.exp(predicted_log_return['return'].iloc[0]) * self.base_price
        for i in range(1, len(predicted_log_return)):
            predicted_log_return['price'].iloc[i] = np.exp(predicted_log_return['return'].iloc[i]) * predicted_log_return['price'].iloc[i - 1]

        return pd.DataFrame({"date": pd.date_range(self.base_date, periods=period+1), 
                             "price": [self.base_price] + predicted_log_return['price'].to_list()})
