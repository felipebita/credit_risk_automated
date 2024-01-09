import pandas as pd
from category_encoders       import OneHotEncoder

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
    - subs_char_names(): Substitute underscores in column names with empty strings.
    - get_prep_data(): Return the prepared data after transformations.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = None
        self.prepared_data = None
    
    def load_data(self):
        """
        Load raw data from the specified file.

        Raises:
        - Exception: If there is an error loading the data.
        """
        try:
            self.raw_data = pd.read_csv(self.data_path)
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def encoder(self):
        """
        Perform one-hot encoding on specified categorical columns.
        """
        columns = ['person_home_ownership','loan_intent']
        enc = OneHotEncoder(cols=columns, use_cat_names=True)
        self.prepared_data = enc.fit_transform(self.raw_data)
    
    def loan_grade_prep(self):
        """
        Map loan grade categories to numeric values.
        """
        grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        self.prepared_data['loan_grade'] = self.prepared_data['loan_grade'].map(grade_mapping)
    
    def default_onfile_prep(self):
        """
        Map default on file categories to numeric values.
        """
        grade_mapping = {'N': 0, 'Y': 1}
        self.prepared_data['cb_person_default_on_file'] = self.prepared_data['cb_person_default_on_file'].map(grade_mapping)

    def subs_char_names(self):
        """
        Substitute underscores in column names with empty strings.
        """
        new_names = [string.replace("_", "") for string in self.prepared_data.columns.values]
        self.prepared_data.columns = new_names
    
    def get_prepdata(self):
        """
        Return the prepared data after transformations.

        Returns:
        - pd.DataFrame: The prepared data.
        """
        return self.prepared_data  
    
    # Example Usage within the module
    if __name__ == "__main__":
        # Example Usage:
        # Assuming you have a CSV file 'raw_data.csv' with columns 'person_home_ownership', 'loan_intent', 'cb_person_default_on_file', 'loan_grade', ...

        # Create an instance of DataPrep
        data_preparer = DataPrep(data_path='raw_data.csv')

        # Load raw data
        data_preparer.load_data()

        # Perform one-hot encoding on specified categorical columns
        data_preparer.encoder()

        # Map loan grade categories to numeric values
        data_preparer.loan_grade_prep()

        # Substitute underscores in column names with empty strings
        data_preparer.subs_char_names()

        # Get the final prepared dataset
        prepared_data = data_preparer.get_prep_data()

        # Display the prepared dataset
        print("Final Prepared Dataset:")
        print(prepared_data)