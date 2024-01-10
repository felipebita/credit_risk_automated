import pandas as pd
from category_encoders       import OneHotEncoder
import argparse
import yaml
from typing import Text
from src.utils.logs import get_logger

class DataPrep:
    """
    DataPrep class for preparing and transforming raw data.

    Parameters:
    - data_path (str): The file path to the raw data file.

    Attributes:
    - data_path (str): The file path to the raw data file.
    - raw_data (pd.DataFrame): The raw data loaded from the specified file.
    - prepared_data (pd.DataFrame): The prepared data after applying encoding and transformations.

    Methods:
    - load_data(): Load raw data from the specified file.
    - encoder(): Perform one-hot encoding on specified categorical columns.
    - loan_grade_prep(): Map loan grade categories to numeric values.
    - def default_onfile_prep(): Map default on file categories to numeric values.
    - subs_char_names(): Substitute underscores in column names with empty strings.
    - save_prepdata(): Save the prepared data after transformations.
    """
    
    def __init__(self, config_path: Text):
        """
        Initialize DataSplit instance.

        Parameters:
        - config_path (str): The file path to the configuration file.
        """
        with open(config_path) as conf_file:
            self.config = yaml.safe_load(conf_file)

        self.logger = get_logger('DATA_PREP', log_level=self.config['base']['log_level'])
    
    def load_data(self):
        """
        Load raw data from the specified file.
        """
        self.logger.info('Get dataset path')
        self.raw_data = pd.read_csv(self.config['data_process']['load_path'])

    
    def encoder(self):
        """
        Perform one-hot encoding on specified categorical columns.
        """
        self.logger.info('Encode variables')
        columns = ['person_home_ownership','loan_intent']
        enc = OneHotEncoder(cols=columns, use_cat_names=True)
        self.prepared_data = enc.fit_transform(self.raw_data)
    
    def loan_grade_prep(self):
        """
        Map loan grade categories to numeric values.
        """
        self.logger.info('Prepare "loan grade" variable')
        grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        self.prepared_data['loan_grade'] = self.prepared_data['loan_grade'].map(grade_mapping)
    
    def default_onfile_prep(self):
        """
        Map default on file categories to numeric values.
        """
        self.logger.info('Prepare "default on file" variable')
        grade_mapping = {'N': 0, 'Y': 1}
        self.prepared_data['cb_person_default_on_file'] = self.prepared_data['cb_person_default_on_file'].map(grade_mapping)

    def subs_char_names(self):
        """
        Substitute underscores in column names with empty strings.
        """
        self.logger.info('Prepare variables names')
        new_names = [string.replace("_", "") for string in self.prepared_data.columns.values]
        self.prepared_data.columns = new_names
    
    def save_prepdata(self):
        """
        Save the prepared data after transformations.
        """
        self.logger.info('Save prepared data')
        self.prepared_data.to_csv(self.config['data_process']['save_path']) 

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    # Create an instance of DataPrep
    data_preparer = DataPrep(config_path = args.config)

    # Load raw data
    data_preparer.load_data()

    # Perform one-hot encoding on specified categorical columns
    data_preparer.encoder()

    # Map loan grade categories to numeric values
    data_preparer.loan_grade_prep()

    # Map loan grade categories to numeric values
    data_preparer.default_onfile_prep()

    # Substitute underscores in column names with empty strings
    data_preparer.subs_char_names()

    # Save prepared dataset
    data_preparer.save_prepdata()
