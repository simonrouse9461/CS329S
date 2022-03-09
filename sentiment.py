import json
import urllib
import pickle
from tqdm.auto import tqdm
import pandas as pd
from google.cloud import storage, dataproc

tqdm.pandas()


class SentimentAnalysis:
    
    GCP_PROJECT = "delta-chess-269600"
    GCP_REGION = "us-central1"
    DATAPROC_CLUSTER = "cs329s-final"
    GCS_BUCKET = "dataproc-staging-us-central1-547349113865-aesxzk1e"
    GCS_JOB_SCRIPT = "notebooks/jupyter/job.py"
    GCS_JOB_INPUT = "notebooks/jupyter/input.txt"
    GCS_JOB_OUTPUT = "notebooks/jupyter/output.json"

    def __init__(self, credential):
        self.storage = storage.Client.from_service_account_info(credential)
        self.bucket = self.storage.get_bucket(self.GCS_BUCKET)
        self.dataproc = dataproc.JobControllerClient.from_service_account_info(credential, client_options={
            "api_endpoint": f"{self.GCP_REGION}-dataproc.googleapis.com:443"
        })
        
    def submit(self, urls):
        blob = self.bucket.blob(self.GCS_JOB_INPUT)
        blob.upload_from_string("\n".join(urls))
        self.job = self.dataproc.submit_job(request={
            "project_id": self.GCP_PROJECT,
            "region": self.GCP_REGION,
            "job": {
                "placement": {
                    "cluster_name": self.DATAPROC_CLUSTER
                },
                "pyspark_job": {
                    "main_python_file_uri": f"gs://{self.GCS_BUCKET}/{self.GCS_JOB_SCRIPT}"
                },
            }
        })
        
    def fetch_job_info(self):
        self.job = self.dataproc.get_job(request={
            "project_id": self.GCP_PROJECT,
            "region": self.GCP_REGION,
            "job_id": self.job.reference.job_id
        })
        return self.job
    
    def wait(self):
        while self.fetch_job_info().status.state.name not in ["DONE", "ERROR"]:
            pass
        return self.fetch_job_info().status.state.name
    
    def retrieve_result(self):
        blob = self.bucket.get_blob(self.GCS_JOB_OUTPUT)
        return pd.DataFrame(json.loads(blob.download_as_string()))
        