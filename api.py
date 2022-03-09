import json
import urllib
import requests
import pandas as pd


class CoinAPI:
    
    API_ENDPOINT = "https://rest.coinapi.io/v1"
    
    def __init__(self, api_key):
        self.api_key = api_key if isinstance(api_key, list) else [api_key] 
        
    def fetch(self, ticker, start, end=None):
        params = {"time_start": start.strftime("%Y-%m-%d"),
                  "period_id": "1DAY",
                  "limit": 100000}
        if end is not None:
            params["time_end"] = end.strftime("%Y-%m-%d")
        paramstr = urllib.parse.urlencode(params)
        url = f'{self.API_ENDPOINT}/exchangerate/{ticker}/USD/history?{paramstr}'
        for key in self.api_key:
            if (response := requests.get(url, headers={"X-CoinAPI-Key": key})).ok:
                return (pd.DataFrame(json.loads(response.text))
                        .astype({"time_period_start": "datetime64", "time_period_end": "datetime64"}))
        raise Exception(json.loads(response.text)["error"])


class CryptoNews:
    
    API_ENDPOINT = "https://cryptonews-api.com/api/v1"
    
    def __init__(self, api_key):
        self.api_key = api_key
        
    def _encode_datetime_range(self, start, end=None):
        if end is None:
            end = pd.Timestamp.now()
        start_date, start_time = start.strftime("%m%d%Y %H%M%S").split()
        end_date, end_time = end.strftime("%m%d%Y %H%M%S").split()
        return {
            "date": f"{start_date}-{end_date}",
            "time": f"{start_time}-{end_time}"
        }

    def _fetch_page(self, tickers, start, end, source, items, page):
        params = {"tickers": ",".join(tickers), 
                  "token": self.api_key,
                  "items": items,
                  "page": page}
        if start is not None:
            params["date"] = self._encode_datetime_range(start, end)["date"]
        if source is not None:
            params["source"] = ",".join(source)
        paramstr = urllib.parse.urlencode(params)
        print(paramstr)
        data = urllib.request.urlopen(f"{self.API_ENDPOINT}?{paramstr}").read()
        df = pd.DataFrame(json.loads(data)['data'])
        if len(df) == 0:
            return None
        df["date"] = pd.to_datetime(df["date"])
        return df

    def fetch(self, ticker, start, end=None, source=None):
        results = []
        page = 1
        while (df := self._fetch_page(tickers=[ticker], 
                                      start=start, 
                                      end=None, 
                                      source=source,
                                      items=50,
                                      page=page)) is not None:
            results.append(df)
            page += 1
        if len(results) > 0:
            return pd.concat(results).reset_index(drop=True)[[
                "news_url", "title", "source_name", "date", "topics", "tickers"
            ]]
        return None