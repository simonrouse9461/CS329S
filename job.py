import os
import json
import random
import nltk
from newspaper import Article, ArticleException
from transformers import pipeline
from google.cloud import storage
import pyspark


conf = pyspark.SparkConf().setMaster("yarn").setAppName("Sentiment Analysis")
sc = pyspark.SparkContext.getOrCreate(conf=conf)


BUCKET_ID = "dataproc-staging-us-central1-547349113865-aesxzk1e"
NUM_EXECUTORS = sc.getConf().get("spark.executor.instances")


def nltk_init(path="/tmp/nltk_data"):
    os.makedirs(path, exist_ok=True)
    nltk.data.path = [path]
    try:
        nltk.download('punkt', download_dir=path)
    except:
        pass


def scrape_article(inputs):
    nltk_init()
    key, val = inputs
    article = Article(key)
    try:
        article.download()
        article.parse()
        article.nlp()
    except ArticleException:
        return None
    return key, {"text": article.text,
                 "summary": article.summary,
                 "keywords": article.keywords}


def sentiment(inputs):
    key, val = inputs
    if val["summary"] is not None:
        result = sent_model_bc.value(val["summary"])[0]
        result.update(val)
        return key, result
    else:
        return None


def format_output(inputs):
    key, val = inputs
    val["url"] = key
    return val


sent_model = pipeline("text-classification", model="oandreae/financial_sentiment_model")
sent_model_bc = sc.broadcast(sent_model)


storage_client = storage.Client()
bucket = storage_client.get_bucket(BUCKET_ID)
input_blob = bucket.get_blob('notebooks/jupyter/input.txt')
url_list = input_blob.download_as_string().decode().split("\n")


rdd = (sc.parallelize(enumerate(url_list), NUM_EXECUTORS)
       .map(lambda x: (x[1], x[0]))
       .map(scrape_article)
       .filter(lambda x: x is not None)
       .map(sentiment)
       .filter(lambda x: x is not None)
       .map(format_output))


results = rdd.collect()
output_blob = bucket.blob('notebooks/jupyter/output.json')
output_blob.upload_from_string(json.dumps(results))

