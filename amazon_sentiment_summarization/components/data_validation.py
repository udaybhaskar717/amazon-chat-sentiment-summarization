from amazon_sentiment_summarization.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from amazon_sentiment_summarization.entity.config_entity import DataValidationConfig
from amazon_sentiment_summarization.exception.exception import AmazonSentimentException
from amazon_sentiment_summarization.logging.logger import logging
from amazon_sentiment_summarization.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os, sys
from amazon_sentiment_summarization.utils.main_utils.utils import read_yaml_file, write_yaml_file

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            logging.info("Initializing DataValidation component.")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            logging.info(f"Reading schema configuration from: {SCHEMA_FILE_PATH}")
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logging.info("Schema configuration loaded successfully.")
        except Exception as e:
            logging.error(f"Error during initialization of DataValidation: {e}")
            raise AmazonSentimentException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from file: {file_path}")
            if not os.path.exists(file_path):
                raise AmazonSentimentException(f"File not found: {file_path}", sys)
            
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                return pd.read_excel(file_path, engine='openpyxl')
            else:
                raise AmazonSentimentException("Unsupported file format. Please provide a CSV or Excel file.", sys)
        except Exception as e:
            logging.error(f"Error reading data from {file_path}: {e}")
            raise AmazonSentimentException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            logging.info("Validating number of columns in the dataset.")
            expected_columns = set(self._schema_config.get('columns', {}).keys())
            actual_columns = set(dataframe.columns)
            
            if not expected_columns:
                raise ValueError("Schema file is missing 'columns' key.")
            
            if expected_columns.issubset(actual_columns):
                logging.info("Column validation successful.")
                return True
            
            logging.error(f"Column mismatch. Expected: {expected_columns}, Found: {actual_columns}")
            return False
        except Exception as e:
            logging.error(f"Error during column validation: {e}")
            raise AmazonSentimentException(e, sys)

    def validate_missing_values(self, dataframe: pd.DataFrame) -> bool:
        try:
            logging.info("Validating missing values in the dataset.")
            if dataframe.isnull().sum().sum() > 0:
                logging.error("Dataset contains missing values.")
                return False
            logging.info("No missing values found.")
            return True
        except Exception as e:
            logging.error(f"Error during missing values validation: {e}")
            raise AmazonSentimentException(e, sys)

    def validate_sentiment_labels(self, dataframe: pd.DataFrame) -> bool:
        try:
            logging.info("Validating sentiment labels in the dataset.")
            if 'sentiment' not in dataframe.columns:
                logging.error("Sentiment column missing in dataset.")
                return False
            
            valid_labels = {"Curious to dive deeper", "Happy", "Neutral", "Surprised", "Disgusted", "Sad", "Fearful", "Angry"}  
            if not set(dataframe['sentiment'].dropna().unique()).issubset(valid_labels):
                logging.error("Invalid sentiment labels detected.")
                return False
            
            logging.info("Sentiment labels validation successful.")
            return True
        except Exception as e:
            logging.error(f"Error during sentiment labels validation: {e}")
            raise AmazonSentimentException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation process.")
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)
            logging.info("Training and testing datasets loaded successfully.")

            if not self.validate_number_of_columns(train_df) or not self.validate_number_of_columns(test_df):
                logging.error("Column validation failed.")
                raise AmazonSentimentException("Column validation failed.", sys)
            # if not self.validate_missing_values(train_df) or not self.validate_missing_values(test_df):
            #     logging.error("Missing values found in the dataset.")
            #     raise AmazonSentimentException("Missing values found.", sys)
            if not self.validate_sentiment_labels(train_df) or not self.validate_sentiment_labels(test_df):
                logging.error("Invalid sentiment labels found.")
                raise AmazonSentimentException("Invalid sentiment labels found.", sys)

            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Directory created for validated data: {dir_path}")

            logging.info("Saving validated training and testing datasets.")
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)
            logging.info("Validated datasets saved successfully.")

            return DataValidationArtifact(
                validation_status=True,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=None,
            )
        except Exception as e:
            logging.error(f"Error during data validation process: {e}")
            raise AmazonSentimentException(e, sys)
