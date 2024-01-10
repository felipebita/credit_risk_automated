import argparse
import joblib
import pandas as pd
from typing import Text
import yaml
from src.utils.logs import get_logger
from src.train.train import train


class TrainModel:
    """
    TrainModel class for training and saving machine learning models.

    Parameters:
    - config_path (str): The file path to the configuration file.

    Attributes:
    - config (dict): Dictionary containing configuration parameters.
    - logger (Logger): Logger instance for logging information.
    - estimator_name (str): Name of the machine learning estimator.
    - train_df (pd.DataFrame): DataFrame containing the training dataset.
    - model: Trained machine learning model.

    Methods:
    - get_estimator(): Extract the name of the machine learning estimator from the configuration.
    - load_traindata(): Load the training dataset from the specified file path in the configuration.
    - train_model(): Train a machine learning model using the specified estimator and hyperparameters.
    - save_model(): Save the trained machine learning model to a specified file path.
    """

    def __init__(self, config_path: Text):
        """
        Initialize TrainModel instance.

        Args:
        - config_path (str): The file path to the configuration file.
        """
        with open(config_path) as conf_file:
            self.config = yaml.safe_load(conf_file)

        self.logger = get_logger('TRAIN', log_level=self.config['base']['log_level'])
    
    def get_estimator(self):
        """
        Extract the name of the machine learning estimator from the configuration.
        """
        self.logger.info('Get estimator name')
        self.estimator_name = self.config['train']['estimator_name']
        self.logger.info(f'Estimator: {self.estimator_name}')

    def load_traindata(self):
        """
        Load the training dataset from the specified file path in the configuration.
        """
        self.logger.info('Load train dataset')
        self.train_df = pd.read_csv(self.config['data_split']['trainset_path'])

    def train_model(self):
        """
        Train a machine learning model using the specified estimator and hyperparameters.
        """
        self.logger.info('Train model')
        self.model = train(
            df = self.train_df,
            target_column = self.config['train']['target'],
            estimator_name = self.estimator_name,
            param_grid = self.config['train']['estimators'][self.estimator_name]['param_grid'],
            cv = self.config['train']['cv']
        )
        self.model.cv_results_
        self.logger.info(f'Best params: {self.model.best_params_}')
        self.logger.info(f'Best score: {self.model.best_score_}')

    def save_model(self):
        """
        Save the trained machine learning model to a specified file path.
        """
        self.logger.info('Save model')
        models_path = self.config['train']['model_path']
        joblib.dump(self.model, models_path)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    # Create an instance of TrainModel
    trainer = TrainModel(config_path=args.config)

    # Extract the estimator name
    trainer.get_estimator()

    # Load the training dataset
    trainer.load_traindata()

    # Train the machine learning model
    trainer.train_model()

    # Save the trained model
    trainer.save_model()