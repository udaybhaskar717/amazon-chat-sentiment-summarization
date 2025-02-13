from amazon_sentiment_summarization.components.data_ingestion import DataIngestion
from amazon_sentiment_summarization.components.data_validation import DataValidation
from amazon_sentiment_summarization.exception.exception import AmazonSentimentException
from amazon_sentiment_summarization.logging.logger import logging
from amazon_sentiment_summarization.entity.config_entity import DataIngestionConfig, DataValidationConfig
from amazon_sentiment_summarization.entity.config_entity import TrainingPipelineConfig
# from amazon_sentiment_summarization.components.data_transformation import DataTransformation
# from amazon_sentiment_summarization.entity.config_entity import ModelTrainerConfig
# from amazon_sentiment_summarization.components.model_trainer import ModelTrainer

import sys

if __name__ == "__main__":
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(training_pipeline_config=trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()  # Corrected variable name
        logging.info("Data Ingestion Completed")
        print(dataingestionartifact)

        # Data Validation
        datavalidationconfig = DataValidationConfig(training_pipeline_config=trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact, datavalidationconfig)  # Corrected variable name
        logging.info("Initiate data validation")
        datavalidationartifact = data_validation.initiate_data_validation()
        logging.info("Data Validation Completed")
        print(datavalidationartifact)
        # data_transformation_config=DataTransformationConfig(trainingpipelineconfig)
        # logging.info("data Transformation started")
        # data_transformation=DataTransformation(datavalidationartifact,data_transformation_config)
        # data_transformation_artifact=data_transformation.initiate_data_transformation()
        # print(data_transformation_artifact)
        # logging.info("data Transformation completed")

        # logging.info("Model Training sstared")
        # model_trainer_config=ModelTrainerConfig(trainingpipelineconfig)
        # model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        # model_trainer_artifact=model_trainer.initiate_model_trainer()
        # logging.info("Model Training artifact created")
        
    except Exception as e:
        raise AmazonSentimentException(e, sys)
