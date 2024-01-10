import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, make_scorer
from typing import Dict, Text
from xgboost import XGBClassifier


class UnsupportedClassifier(Exception):
    """
    Exception raised for unsupported classifiers.

    Attributes:
    - estimator_name (str): The name of the unsupported estimator.
    """
    def __init__(self, estimator_name):
        """
        Initialize UnsupportedClassifier instance.

        Args:
        - estimator_name (str): The name of the unsupported estimator.
        """
        self.msg = f'Unsupported estimator {estimator_name}'
        super().__init__(self.msg)


def get_supported_estimator() -> Dict:
    """
    Returns a dictionary of supported classifiers.

    Returns:
    - Dict: Dictionary containing supported classifiers.
    """

    return {
        'xgb': XGBClassifier,
    }


def train(df: pd.DataFrame, target_column: Text,
          estimator_name: Text, param_grid: Dict,  cv: int):
    """
        Train a machine learning model using GridSearchCV.

        Args:
        - df (pd.DataFrame): The dataset.
        - target_column (str): The name of the target column.
        - estimator_name (str): The name of the estimator to be used.
        - param_grid (Dict): The grid of hyperparameters for GridSearchCV.
        - cv (int): The number of cross-validation folds.

        Returns:
        - trained model: The trained machine learning model.
        """
    # Get supported estimators
    estimators = get_supported_estimator()

    # Check if the specified estimator is supported
    if estimator_name not in estimators.keys():
        raise UnsupportedClassifier(estimator_name)
    
    # Create a GridSearchCV instance with f1_score as the scoring metric
    estimator = estimators[estimator_name]()
    f1_scorer = make_scorer(f1_score, average='weighted')
    clf = GridSearchCV(estimator=estimator,
                       param_grid=param_grid,
                       cv=cv,
                       verbose=1,
                       scoring=f1_scorer)
    
    # Get X and Y from the dataset
    y_train = df.loc[:, target_column].values.astype('int32')
    X_train = df.drop(target_column, axis=1).values.astype('float32')
    
    # Fit the model using GridSearchCV
    clf.fit(X_train, y_train)

    return clf