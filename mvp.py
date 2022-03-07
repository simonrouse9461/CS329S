import requests
import json
import urllib
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
from newspaper import Article, ArticleException
from transformers import pipeline

from stqdm import stqdm as tqdm
tqdm.pandas()

import nltk
nltk.download('punkt')

SENTIMENT_MODEL_NAME = "oandreae/financial_sentiment_model"
COINAPI_KEY = "7C343D69-E559-4848-A8F6-E60F3631B67E"
COINAPI_ENDPOINT = "https://rest.coinapi.io/v1"
CRYPTONEWS_API_ENDPOINT = "https://cryptonews-api.com/api/v1"
CRYPTONEWS_API_TOKEN = "oonymosrym98pjbpimxrmzv0yqoiotvlqcmprqzb"
CRYPTO_LIST = [
    "BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "LUNA", "DOGE", "AVAX", "DOT"
]
NEWS_SOURCES = [
    "Altcoin Buzz",
    "AMBCrypto",
    "BeInCrypto",
    "Benzinga",
    "Bitcoin",
    "Bitcoin Market Journal",
    "Bitcoin Magazine",
    "Bitcoinist",
    "Bit News",
    "Blockgeeks",
    "Blockonomi",
    "Bloomberg Markets and Finance",
    "Bloomberg Technology",
    "BTCManager",
    "Business Insider",
    "CNBC",
    "CNBC Television",
    "CNN",
    "Coinbureau",
    "Coindesk",
    "Coindoo",
    "Coinfomania",
    "Coingape",
    "Coin Idol",
    "CoinMarketCap",
    "Coin News Asia",
    "Coinnounce",
    "Crypto Briefing",
    "Crypto Daily",
    "Crypto Economy",
    "CryptoNews",
    "CryptoNinjas",
    "Cryptopolitan",
    "CryptoPotato",
    "Crypto Reporter",
    "CryptoSlate",
    "CryptoTicker",
    "Cryptoverze",
    "DailyFX",
    "DCForecasts",
    "Decrypt",
    "FinanceMagnates",
    "Forbes",
    "Fox Business",
    "InvestingCube",
    "Investorplace",
    "Koinpost",
    "Modern Consensus",
    "NewsBTC",
    "Reuters",
    "The Block",
    "TCU",
    "The Cryptonomist",
    "The Currency Analytics",
    "The Daily Hodl",
    "The Motley Fool",
    "Trustnode",
    "UToday",
    "Yahoo Finance",
    "8BTC",
]

st.set_page_config(layout="wide")

st.title("Crypto Market Sentiment Analysis")

cypto_selectbox = st.sidebar.selectbox(
    "Please select a cryptocurrency:",
    CRYPTO_LIST, CRYPTO_LIST.index("SOL")
)

start_date = st.sidebar.date_input("Please select a start date:", pd.Timestamp.now() - pd.DateOffset(days=10))

time_selectbox = st.sidebar.selectbox(
    "Please select a time interval:",
    ["1 Day"]
)

bollinger_window = st.sidebar.number_input(
    "Please select bollinger band window size", 0, 30, 3
)

news_selectbox_container = st.sidebar.container()

select_all = st.sidebar.checkbox("Select all", False)

news_selectbox = news_selectbox_container.multiselect(
    "Please select news sources:",
    NEWS_SOURCES, NEWS_SOURCES if select_all else None
)

def fetch_xrate(ticker, start, end=None):
    headers = {"X-CoinAPI-Key": COINAPI_KEY}
    params = {"time_start": start.strftime("%Y-%m-%d"),
              "period_id": "1DAY"}
    if end is not None:
        params["time_end"] = end.strftime("%Y-%m-%d")
    paramstr = urllib.parse.urlencode(params)
    url = f'{COINAPI_ENDPOINT}/exchangerate/{ticker}/USD/history?{paramstr}'
    return json.loads(requests.get(url, headers=headers).text)

def get_bollinger_bands(prices, window):
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    bollinger_top = sma + std * 2 # Calculate top band
    bollinger_bottom = sma - std * 2 # Calculate bottom band
    return bollinger_top, bollinger_bottom

def encode_datetime_range(start, end=None):
    if end is None:
        end = pd.Timestamp.now()
    start_date, start_time = start.strftime("%m%d%Y %H%M%S").split()
    end_date, end_time = end.strftime("%m%d%Y %H%M%S").split()
    return {
        "date": f"{start_date}-{end_date}",
        "time": f"{start_time}-{end_time}"
    }

def fetch_page(tickers, start, end, source, items, page):
    params = {"tickers": ",".join(tickers), 
              "token": CRYPTONEWS_API_TOKEN,
              "items": items,
              "page": page}
    if start is not None:
        params["date"] = encode_datetime_range(start, end)["date"]
    if source is not None:
        params["source"] = ",".join(source)
    paramstr = urllib.parse.urlencode(params)
    print(paramstr)
    data = urllib.request.urlopen(f"{CRYPTONEWS_API_ENDPOINT}?{paramstr}").read()
    df = pd.DataFrame(json.loads(data)['data'])
    if len(df) == 0:
        return None
    df["date"] = pd.to_datetime(df["date"])
    return df

data = None
if len(news_selectbox) > 0:
    results = []
    page = 1
    while (df := fetch_page(tickers=[cypto_selectbox], 
                            start=start_date, 
                            end=None, 
                            source=news_selectbox,
                            items=50,
                            page=page)) is not None:
        results.append(df)
        page += 1
    if len(results) > 0:
        data = pd.concat(results).reset_index(drop=True)
        
chart_placeholder = st.empty()  
fig = make_subplots(specs=[[{"secondary_y": True}]])

xrate = pd.DataFrame(fetch_xrate("SOL", start=start_date))
fig.add_traces(px.line(xrate, 
                       x="time_period_start", 
                       y="rate_close", 
                       color_discrete_sequence=[px.colors.qualitative.Plotly[3]]).data)

if bollinger_window > 0 and bollinger_window < len(xrate):
    xrate["bollinger_top"], xrate["bollinger_bottom"] = get_bollinger_bands(xrate["rate_close"], window=bollinger_window)
    fig.add_traces(px.line(xrate, 
                           x="time_period_start", 
                           y="bollinger_top", 
                           color_discrete_sequence=[px.colors.qualitative.Plotly[2]]).data)
    fig.add_traces(px.line(xrate, 
                           x="time_period_start", 
                           y="bollinger_bottom", 
                           color_discrete_sequence=[px.colors.qualitative.Plotly[1]]).data)


fig.update_yaxes(title_text="Exchange Rate (USD)", secondary_y=False)
chart_placeholder.plotly_chart(fig, use_container_width=True)

if data is None:
    st.warning(f"No records found")
else:
    info_placeholder = st.empty()
    st.header("Raw Data")
    raw_data_placeholder = st.empty()
    
    info_placeholder.success(f"{len(data)} records found")
    raw_data_placeholder.write(data)
    
    def scrape_article(url, nlp):
        article = Article(url)
        try:
            article.download()
            article.parse()
            if nlp:
                article.nlp()
        except ArticleException:
            return None
        return article

    tqdm.pandas(desc="Scraping and processing news articles")
    articles = data["news_url"].progress_apply(lambda url: scrape_article(url=url, nlp=True))
    notnan = ~articles.isna()
    data.loc[notnan, "news_content"] = articles[notnan].apply(lambda art: art.text)
    data.loc[notnan, "news_summary"] = articles[notnan].apply(lambda art: art.summary)
    data.loc[notnan, "news_keywords"] = articles[notnan].apply(lambda art: art.keywords)
    
    info_placeholder.success(f"{len(data)} records found, {sum(notnan)} articles scraped")
    raw_data_placeholder.write(data)
    
    # st.header("Analysis Result")
    # chart_placeholder = st.empty()
    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    
    
    with st.spinner('Loading sentiment analysis model...'):
        classifier = pipeline("text-classification", model=SENTIMENT_MODEL_NAME)
    notnan = ~data["news_summary"].isna()
    with st.spinner('Analyzing articles...'):
        analysis = pd.DataFrame(classifier(data.loc[notnan, "news_summary"].to_list()))
    data.loc[notnan, "model_prediction"] = analysis["label"].values
    data.loc[notnan, "model_confidence"] = analysis["score"].values
    
    info_placeholder.success(f"{len(data)} records found, {sum(notnan)} articles scraped and processed")
    raw_data_placeholder.write(data)
    
    summary = (data
               .groupby(by=[data["date"].dt.date, "model_prediction"])[["model_confidence"]]
               .sum()
               .reset_index()
               .astype({"date": "datetime64"})
               .pivot(index="date", columns="model_prediction", values="model_confidence")
               .fillna(0)
               .resample("D", kind="timestamp")
               .sum())
    
    plot2 = px.bar(summary, 
                   barmode="group", 
                   labels={"value": "sentiment scores"},
                   color_discrete_sequence=[px.colors.qualitative.Plotly[1], 
                                            px.colors.qualitative.Plotly[0],
                                            px.colors.qualitative.Plotly[2]])
    plot2.update_traces(yaxis="y2")
    fig.add_traces(plot2.data)
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
    fig.update_layout(legend_title_text='Sentiments')
    chart_placeholder.plotly_chart(fig, use_container_width=True)
    