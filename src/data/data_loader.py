import os
import sys
sys.path.append('./')
from src.utils.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

from src.data.data_transformation import DataTransformation

from src.models.model_trainer import ModelTrainer

from dvclive import Live
from pathlib import Path

import boto3
from botocore.exceptions import NoCredentialsError
from io import StringIO


class DataIngestionConfig:
    # print(Path.home())
    # print(Path.cwd())
    output_folder: Path = Path.cwd() / "models"

    train_data_path: Path = output_folder / 'train.csv'
    test_data_path: Path = output_folder / 'test.csv'
    raw_data_path: Path = output_folder / 'data.csv'

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:  
            # input_file_path = Path.cwd() / "data" / "interim" / "properties_post_feature_selection_v2.csv"
            # df = pd.read_csv(input_file_path)

            def read_data_from_s3(s3_bucket, s3_key):
                # Retrieve AWS credentials from GitHub Secrets
                # print("AWS_ACCESS_KEY_ID:", os.environ.get('AWS_ACCESS_KEY_ID'))
                # print("AWS_SECRET_ACCESS_KEY:", os.environ.get('AWS_SECRET_ACCESS_KEY'))
                # print("AWS_REGION:", os.environ.get('AWS_REGION'))


                aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
                aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
                aws_region = os.environ.get('AWS_REGION')

                # Initialize a session using GitHub Secrets for AWS credentials
                session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=aws_region
                )

                # Create an S3 client
                s3_client = session.client('s3')

                # Download the file from S3
                try:
                    # Use StringIO to read the CSV directly from the S3 object
                    s3_response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
                    # s3_data = s3_response['Body'].read().decode('utf-8')
                    # s3_csv = StringIO(s3_data)

                    # Read the CSV into a DataFrame
                    df = pd.read_csv(s3_response['Body'])
                    print(df.shape)

                    return df

                except NoCredentialsError:
                    print("Credentials not available")
                except Exception as e:
                    print(f"An error occurred: {e}")

            # Example usage:
            s3_bucket = 'mlops-kn'
            s3_key = 'data/interim/properties_post_feature_selection_v2.csv'

            df = read_data_from_s3(s3_bucket, s3_key)
            print(df.shape)
  

            # df.drop(columns=['servant room', 'study room', 'others'], inplace=True)

            self.ingestion_config.output_folder.mkdir(parents=True, exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    score = modeltrainer.initiate_model_trainer(train_arr,test_arr)
    with Live(save_dvc_exp=True) as live:
      live.log_metric('R2 score',score)

    artifacts_dir = Path.cwd() / 'artifacts'
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    score_file_path = artifacts_dir / 'score.txt'
    with open(score_file_path, 'w') as score_file:
        score_file.write(str(score))