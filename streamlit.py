import requests
import json
import urllib
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
from stqdm import stqdm as tqdm
tqdm.pandas()
import nltk
nltk.download('punkt')

from api import *
from predictor import *
from sentiment import *


SENTIMENT_MODEL_NAME = "oandreae/financial_sentiment_model"
COINAPI_KEY = [
    "7C343D69-E559-4848-A8F6-E60F3631B67E",
    "9C2AF8D6-C0D2-4A28-A344-3271CF1FDBFC",
    "E092FAB1-7B5E-4F27-AC48-80E463458A51",
]
CRYPTONEWS_API_TOKEN = "oonymosrym98pjbpimxrmzv0yqoiotvlqcmprqzb"
GCP_CREDENTIAL = {
  "type": "service_account",
  "project_id": "delta-chess-269600",
  "private_key_id": "503f41a450c9fabcd3eb9eed550dccca635ab4dc",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDPDf4RrZp31l37\nA2DHztAL0PSNAGkeavnIawcEqtTQScpyFTn8FliSF+8M15s4d/UZYFsxFKdBl26F\nyKtCquqhBn5hmhZkza7Ev2nMjwvwWQdrMjvglCAM10R6Nhoit6tCn1zAbzVZ8HGM\nIRD8f8HM8LcLj6RgZaTGKZwRamM5G0SaZfSAzGWlMpBUd3JPNtjOvuvdmsCP+5oc\nCRFJNsAq6aj3WCfn4ufSPRd7oynN6WNKC2Svkz1vk/Mos9viQ19Y7UMpY5SssYvB\nq5PKUcxlA/qtDb3IZiXVeKzjPS7jZyjZgns9GYGcLXK2PPcx6YFrq0kLRjH/0Hhi\nG2Xd7XB5AgMBAAECggEAC3paGD7IblY61XxldA4Z6F3IAL0CFsaVXC/arr6Nl5JP\nn61fjoLqnAapoMue9i1oL+TwiTL85pzvaokqPULeSNjkTqLNFP917cAyrD1NyyAU\nUjPxr8xmTm/sgt3W6w/FdacB7ET97yNuF/eJSoYGh0bczs5CrXlU/gs2dJyyLWsk\n8gFBWoPazA7g827SSmaRyQ6uja/Av+kSARTQ7KWbctPXP0DCMwk9eMEL9GVlICSR\nPXU7iAa/FyM1BGXMTL9LtIu43sTm/voJany0F99um96tP0rZQYMlHwPg0dShM5Nv\nxWFeSBK3iQcrzf6Q/yRZigz/mLTjJlbThddeuESMUQKBgQD5Xg9wEWfK/h2wcMWA\n3p6LQcY/fbOAYLObr6DOu/KoMhrZudloYFUa++EDPX5pmT2sXM0LbltX774G8AVk\nRlmU1WFaR1tATnCCG4GFeGR0+BIz6h9ntAOD4M1k6V37GOWAxLY4EHvOdrnPg8S6\nwTGhoQcHuLcDT4w4wZFs4BS+nQKBgQDUj9M3JqiCs12GBj0zqVRKefu7E21QsoAF\nCViLu4Yu6bU5tNzbltBanRr9YX4QW3LAcBBn9reSN7qSQ+YD3eWsFYUvqnOYF/96\nNyxXI/JhjSB8kFSK5zBYqSFkdup5d3QaUXVnFJZmJVTbIoS1hSD10aZ25ZYHEjcC\nxMP/E74EjQKBgC4KU8dZL1SnPkwJRi3Y7GTBrByk1LNrJz4jWwlQYijzt0ljquQ+\nhPgKcSzr+Z79kAl2yNTHd03xEaCuSBCPfJKiIutMKWjiEpuzAGLoK4P9GT9Ehq/a\n6Js8si9jdtqZaiYwK4SGZpVkDkJmDbh9WvCAjo+6Zu/RjA4ejv8PEEVxAoGBAKxQ\ni+FBrYmG7mIf3K1sr7BQgwl9DjlE+xMaKHXeZ0DQpOFLBV/eOrm6co7F4fRQrg3i\nyun8z4PxOYYpFOY9lFqUd4vUmjDKA4mIAKIDuhHq3lMcjeyszjyRn0haPmqJs81C\nC/KsdeAIk2mx6fNdIQMmGdR4+c5xrbbI3DqEPp5hAoGAWoAIsvoyL+8kt8RClG2O\nQ+NNOVIHXOVO5VOv3OjEt72oGO3ACKIBl1mx1R1JT/or0IS+zWM8MVA7lLzR+53a\n87RJey8ChMw3ODnem5FmOqj1nxxF6bd+Q8T/MGrDhEilFnS/qr2AP0Z0XZ6UHiYP\nGFTH9Hj9+ghCIc1iK1AdGZs=\n-----END PRIVATE KEY-----\n",
  "client_email": "cs329s@delta-chess-269600.iam.gserviceaccount.com",
  "client_id": "103421754271257027827",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/cs329s%40delta-chess-269600.iam.gserviceaccount.com"
}
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

coinapi = CoinAPI(api_key=COINAPI_KEY)
cryptonews = CryptoNews(api_key=CRYPTONEWS_API_TOKEN)
sentiment = SentimentAnalysis(GCP_CREDENTIAL)
predictor = {
    "Prophet": DeepProphet(),
    "ARIMA": Arima(),
}

st.set_page_config(layout="wide")

st.title("Crypto Market Sentiment Analysis")

### Side Bar ###
widgets = {}

widgets["ticker"] = st.sidebar.selectbox("Please select a cryptocurrency:",
                                         CRYPTO_LIST, CRYPTO_LIST.index("SOL"))
widgets["start_date"] = st.sidebar.date_input("Please select a start date:", 
                                              pd.Timestamp.now() - pd.DateOffset(days=30))
widgets["bollinger_window"] = st.sidebar.number_input("Please select bollinger band window size",
                                                      0, 30, 7)
widgets["predictor"] = st.sidebar.selectbox("Please select a predictor algorithm:",
                                            list(predictor.keys()), 0)
widgets["pred_period"] = st.sidebar.number_input("Please select a prediction period:",
                                                 0, 30, 5)
news_selectbox_container = st.sidebar.container()

widgets["select_all"]  = st.sidebar.checkbox("Select all", False)
    
widgets["news_source"] = news_selectbox_container.multiselect("Please select news sources:",
                                                              NEWS_SOURCES, NEWS_SOURCES if widgets["select_all"] else None)
################

news_data = cryptonews.fetch(widgets["ticker"],
                             start=widgets["start_date"],
                             source=widgets["news_source"]) if len(widgets["news_source"]) > 0 else None

chart_placeholder = st.empty()

fig = make_subplots(specs=[[{"secondary_y": True}]])

price = coinapi.fetch(widgets["ticker"], start=widgets["start_date"])

fig.add_traces(px.line(price, 
                       x="time_period_start", 
                       y="rate_close", 
                       color_discrete_sequence=[px.colors.qualitative.Plotly[3]]).data)

if widgets["bollinger_window"] > 0 and widgets["bollinger_window"] < len(price):
    price["bollinger_top"], price["bollinger_bottom"] = get_bollinger_bands(price["rate_close"], window=widgets["bollinger_window"])
    train_data = price[["time_period_start", "rate_close"]].rename({"time_period_start": "date", "rate_close": "price"}, axis=1)
    prediction = predictor[widgets["predictor"]].fit(train_data).predict(widgets["pred_period"])
    fig.add_traces(px.line(price, 
                           x="time_period_start", 
                           y="bollinger_top", 
                           color_discrete_sequence=[px.colors.qualitative.Plotly[2]]).data)
    fig.add_traces(px.line(price, 
                           x="time_period_start", 
                           y="bollinger_bottom", 
                           color_discrete_sequence=[px.colors.qualitative.Plotly[1]]).data)
    fig.add_traces(px.line(prediction, 
                           x="date", 
                           y="price", 
                           color_discrete_sequence=[px.colors.qualitative.Plotly[0]]).data)

fig.update_yaxes(title_text="Exchange Rate (USD)", secondary_y=False)
chart_placeholder.plotly_chart(fig, use_container_width=True)

info_placeholder = st.empty()

if news_data is None:
    info_placeholder.warning(f"No records found")
    
else:
    # st.header("Raw Data")
    
    raw_data_placeholder = st.expander("Raw Data").empty()
    
    info_placeholder.success(f"{len(news_data)} records found")
    
    raw_data_placeholder.write(news_data)
    
#     def scrape_article(url, nlp):
#         article = Article(url)
#         try:
#             article.download()
#             article.parse()
#             if nlp:
#                 article.nlp()
#         except ArticleException:
#             return None
#         return article

#     tqdm.pandas(desc="Scraping and processing news articles")
#     articles = news_data["news_url"].progress_apply(lambda url: scrape_article(url=url, nlp=True))
#     notnan = ~articles.isna()
#     news_data.loc[notnan, "news_content"] = articles[notnan].apply(lambda art: art.text)
#     news_data.loc[notnan, "news_summary"] = articles[notnan].apply(lambda art: art.summary)
#     news_data.loc[notnan, "news_keywords"] = articles[notnan].apply(lambda art: art.keywords)
    
#     info_placeholder.success(f"{len(news_data)} records found, {sum(notnan)} articles scraped")
#     raw_data_placeholder.write(news_data)
    
    # st.header("Analysis Result")
    # chart_placeholder = st.empty()
    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    
    # with st.spinner('Loading sentiment analysis model...'):
    #     classifier = pipeline("text-classification", model=SENTIMENT_MODEL_NAME)
    # notnan = ~news_data["news_summary"].isna()
    # with st.spinner('Scraping and analyzing news articles...'):
    #     analysis = pd.DataFrame(classifier(news_data.loc[notnan, "news_summary"].to_list()))
    # news_data.loc[notnan, "model_prediction"] = analysis["label"].values
    # news_data.loc[notnan, "model_confidence"] = analysis["score"].values
    
    with st.spinner('Scraping and analyzing news articles...'):
        sentiment.submit(news_data["news_url"])
        state = sentiment.wait()
    if state == "ERROR":
        info_placeholder.error(f"Error: Spark job failed! Please check cluster status.")
    else:
        result = sentiment.retrieve_result()
        merged_data = news_data.merge(result, left_on="news_url", right_on="url", how="left").drop("url", axis=1)

        info_placeholder.success(f"{len(news_data)} records found, {sum(~merged_data['text'].isna())} articles successfully processed")
        raw_data_placeholder.write(merged_data)

        summary = (merged_data
                   .groupby(by=[news_data["date"].dt.date, "label"])[["score"]]
                   .sum()
                   .reset_index()
                   .astype({"date": "datetime64"})
                   .pivot(index="date", columns="label", values="score")
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
    