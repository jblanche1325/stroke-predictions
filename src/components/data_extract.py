import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataExtractionConfig:
    train_data_path: str=os.path.join('interim', 'train.csv')
    test_data_path: str=os.path.join('interim', 'test.csv')
    raw_data_path: str=os.path.join('interim', 'data.csv')

class DataExtraction:
    def __init__(self):
        self.extraction_config =  DataExtractionConfig()

    def initiate_data_extraction(self):
        logging.info('Data extraction method or component')
        try:
            df = pd.read_csv('notebook\data\healthcare-dataset-stroke-data.csv')
            logging.info('Read the data as Pandas DataFrame')

            os.makedirs(os.path.dirname(self.extraction_config.train_data_path), exist_ok=True)

            df.to_csv(self.extraction_config.raw_data_path, index=False, header=True)

            logging.info('Train test split')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=865)

            train_set.to_csv(self.extraction_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.extraction_config.test_data_path, index=False, header=True)

            logging.info('Extraction complete')

            return(
                self.extraction_config.train_data_path,
                self.extraction_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    obj = DataExtraction()
    obj.initiate_data_extraction()