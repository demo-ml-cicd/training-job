from typing import NoReturn

import click
import pandas as pd
from google.cloud.storage import Bucket
from ml_python_package.train import train_model, serialize_model
from google.cloud import storage


def download_data_from_s3(s3_bucket: str, s3_path: str, output: str) -> NoReturn:
    client = storage.Client()
    remote_path = f"gs://{s3_bucket}/{s3_path}"
    with open(output, "wb") as f:
        client.download_blob_to_file(remote_path, f)


def upload_data_to_s3(s3_bucket: str, local_path: str, s3_path: str) -> NoReturn:
    client = storage.Client()
    bucket: Bucket = client.bucket(s3_bucket)
    blob = bucket.blob(s3_path)
    blob.upload_from_filename(local_path)


@click.command()
@click.option("--dataset-path", "-d", required=True)
@click.option("--output-path", "-o", required=True)
@click.option("--s3-bucket", required=True)
def train_job(dataset_path: str, output_path: str, s3_bucket: str):
    local_path = "dataset.csv"
    download_data_from_s3(s3_bucket, dataset_path, output=local_path)
    data = pd.read_csv(local_path)

    train_df = data.drop(["target"], 1)
    target = data["target"]
    model = train_model(train_df, target, "RandomForestClassifier")

    local_model_path = "model.pkl"
    serialize_model(model, local_model_path)
    upload_data_to_s3(s3_bucket, local_model_path, output_path)


if __name__ == '__main__':
    train_job()
