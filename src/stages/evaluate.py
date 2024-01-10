import argparse
import joblib
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score
from typing import Text, Dict
import yaml

from src.report.visualize import plot_confusion_matrix
from src.utils.logs import get_logger

class EvaluateModel:
    def __init__(self, config_path: Text):
        """
        Initialize DataSplit instance.

        Parameters:
        - config_path (str): The file path to the configuration file.
        """
        with open(config_path) as conf_file:
            self.config = yaml.safe_load(conf_file)

        self.logger = get_logger('EVALUATE', log_level=self.config['base']['log_level'])

    def load_model_data(self):
        self.logger.info('Load model and test dataset')
        self.test_df = pd.read_csv(self.config['data_split']['testset_path'])
        model_path = self.config['train']['model_path']
        self.model = joblib.load(model_path)
        
    def run_model(self):
        self.logger.info('Run model on test dataset')
        target_column=self.config['train']['target']
        self.y_test = self.test_df.loc[:, target_column].values
        X_test = self.test_df.drop(target_column, axis=1).values
        self.prediction = self.model.predict(X_test)

    def get_scores(self):
        self.logger.info('Get prediction score')
        f1 = f1_score(y_true=self.y_test, y_pred=self.prediction, average='macro')
        cm = confusion_matrix(self.y_test, self.prediction)
        self.report = {
            'f1': f1,
            'cm': cm,
            'actual': self.y_test,
            'predicted': self.prediction
        }
        self.logger.info('Save score in reports')
        json.dump(
            obj={'f1_score': self.report['f1']},
            fp=open(self.config['evaluate']['metrics_file'], 'w')
        )

    def write_confusion_matrix_data(self):
        def convert_to_labels(indexes, labels):
            result = []
            for i in indexes:
                result.append(labels[i])
            return result
        self.logger.info('Write confusion matrix data in reports')
        self.labels = ['Not Default', 'Default']
        assert len(self.prediction) == len(self.y_test)
        predicted_labels = convert_to_labels(self.prediction, self.labels)
        true_labels = convert_to_labels(self.y_test, self.labels)
        cf = pd.DataFrame(list(zip(true_labels, predicted_labels)), columns=["y_true", "predicted"])
        cf.to_csv(self.config['evaluate']['confusion_matrix_data'], index=False)

    def save_confusion_matrix(self):
        self.logger.info('Save confusion matrix image in reports')
        
        plt = plot_confusion_matrix(cm=self.report['cm'],
                                    target_names=self.labels)
        
        plt.savefig(self.config['evaluate']['confusion_matrix_image'])

if __name__ == '__main__':
    #Set config file
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    #Create and instance of EvaluateModel:
    evaluater = EvaluateModel(config_path = args.config)

    evaluater.load_model_data()

    evaluater.run_model()

    evaluater.get_scores()

    evaluater.write_confusion_matrix_data()

    evaluater.save_confusion_matrix()
