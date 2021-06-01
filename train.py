# -*- coding: utf-8 -*-
from typing import NoReturn

import click
import pandas as pd
from boto3 import client
from ml_python_package.train import train_model, build_model, serialize_model

def download_data_from_s3(s3_bucket: str, s3_path: str, output: str) -> NoReturn:
    s3 = client("s3")
    s3.download_file(s3_bucket, s3_path, output)


def upload_data_to_s3(s3_bucket: str, local_path: str, s3_path: str) -> NoReturn:
    s3 = client("s3")
    s3.upload_file(local_path, s3_bucket, s3_path)


@click.command()
@click.option("--dataset-path", "-d", required=True)
@click.option("--output-path", "-o", required=True)
@click.option("--s3-bucket", required=True)
def train_job(dataset_path: str, output_path: str, s3_bucket: str):
    local_path = "./data.csv"
    download_data_from_s3(s3_bucket, dataset_path, local_path)
    data = pd.read_csv(local_path)
    train_df = data.drop(["target"], 1)
    target = data["target"]

    model = train_model(train_df, target, "RandomForestClassifier")
    local_model_path = "model.pkl"
    serialize_model(model, local_model_path)

    upload_data_to_s3(s3_bucket, local_model_path, output_path)


if __name__ == '__main__':
    train_job()
