import json
import urllib
import pickle
from tqdm.auto import tqdm
import pandas as pd
import plotly.express as px
from pipe import *
from newspaper import Article, ArticleException
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

tqdm.pandas()


class CryptoNewsScraper:
    
    API_ENDPOINT = "https://cryptonews-api.com/api/v1"
    
    def __init__(self, token):
        self.token = token
        
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
        params = ({"tickers": ",".join(tickers), 
                   "token": self.token})
        params["items"] = items
        params["page"] = page
        if start is not None:
            params["date"] = self._encode_datetime_range(start, end)["date"]
        if source is not None:
            params["source"] = ",".join(source)
        paramstr = urllib.parse.urlencode(params)
        data = urllib.request.urlopen(f"{self.API_ENDPOINT}?{paramstr}").read()
        df = pd.DataFrame(json.loads(data)['data'])
        if len(df) == 0:
            return None
        df["date"] = pd.to_datetime(df["date"])
        return df
            
    def _scrape_article(self, url, nlp):
        article = Article(url)
        try:
            article.download()
            article.parse()
            if nlp:
                article.nlp()
        except ArticleException:
            return None
        return article
    
    def fetch(self, tickers, start=None, end=None, source=None, nlp=True, verbose=False):
        if start is None:
            start = pd.Timestamp.now() - pd.DateOffset(days=1)
        results = []
        page = 1
        while (df := self._fetch_page(tickers=tickers, start=start, end=end, source=source, items=50, page=page)) is not None:
            results.append(df)
            page += 1
            end = df["date"].min()
            if verbose:
                print(end)
        data = None if len(results) == 0 else pd.concat(results).reset_index(drop=True)
        tqdm.pandas(desc="Downloading")
        articles = data["news_url"].progress_apply(lambda url: self._scrape_article(url=url, nlp=nlp))
        notnan = ~articles.isna()
        data.loc[notnan, "news_content"] = articles[notnan].apply(lambda art: art.text)
        if nlp:
            data.loc[notnan, "news_summary"] = articles[notnan].apply(lambda art: art.summary)
            data.loc[notnan, "news_keywords"] = articles[notnan].apply(lambda art: art.keywords)
        return data
        

class AnalyticsPipeline:
    def __init__(self, model="oandreae/financial_sentiment_model"):
        self.data = None
        self.classifier = pipeline("text-classification", model=model)

    def analyze(self):
        notnan = ~self.data["news_summary"].isna()
        analysis = pd.DataFrame(tqdm(self.classifier(self.data.loc[notnan, "news_summary"].to_list(), batch_size=8), desc="Analyzing"))
        self.data.loc[notnan, "model_prediction"] = analysis["label"].values
        self.data.loc[notnan, "model_confidence"] = analysis["score"].values
        
    def summary(self):
        return (self.data
                .groupby(by=[self.data["date"].dt.date, "model_prediction"])[["model_confidence"]]
                .sum()
                .reset_index()
                .astype({"date": "datetime64"})
                .pivot(index="date", columns="model_prediction", values="model_confidence")
                .fillna(0)
                .resample("D", kind="timestamp")
                .sum())
        
    def set_data(self, data):
        self.data = data
        
    def save_data(self, file):
        pickle.dump(self.data, open(file, "rw"))
        
    def load_data(self, file):
        self.data = pickle.load(open(file, "rb"))