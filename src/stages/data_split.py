import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Text
import yaml
from src.utils.logs import get_logger

class DataSplit:
    """
    DataSplit class for splitting processed data into training and testing sets.

    Parameters:
    - config_path (str): The file path to the configuration file.

    Attributes:
    - config (dict): Configuration settings loaded from the specified file.
    - logger: Logger object for recording log messages.

    Methods:
    - load_data(): Load the processed data from the specified file.
    - data_split(): Split features into training and test sets based on the provided configuration.
    - save_sets(): Save the resulting training and test sets to separate CSV files.
    """

    def __init__(self, config_path: Text):
        """
        Initialize DataSplit instance.

        Parameters:
        - config_path (str): The file path to the configuration file.
        """
        with open(config_path) as conf_file:
            self.config = yaml.safe_load(conf_file)

        self.logger = get_logger('DATA_SPLIT', log_level=self.config['base']['log_level'])
    
    def load_data(self):
        """
        Load processed data from the specified file.
        """
        self.logger.info('Load processed data')
        self.dataset = pd.read_csv(self.config['data_process']['save_path'])

    def data_split(self):
        """
        Split features into training and test sets based on configuration.
        """
        self.logger.info('Split features into train and test sets')
        self.train_dataset, self.test_dataset = train_test_split(
            self.dataset,
            test_size=self.config['data_split']['test_size'],
            random_state=self.config['base']['random_state']
        )

    def save_sets(self):
        """
        Save training and test sets to separate CSV files.
        """
        self.logger.info('Save train and test sets')
        self.train_dataset.to_csv(self.config['data_split']['trainset_path'], index=False)
        self.test_dataset.to_csv(self.config['data_split']['testset_path'], index=False)    

if __name__ == '__main__':
    #Set config file
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    # Create an instance of DataSplit
    data_spliter = DataSplit(config_path = args.config)

    # Load processed data
    data_spliter.load_data()

    # Perform split in the processed data
    data_spliter.data_split()

    # Save train and test sets
    data_spliter.save_sets()