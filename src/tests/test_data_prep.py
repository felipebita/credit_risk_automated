import pandas as pd
from data_prep import DataPrep
import pytest
from category_encoders import OneHotEncoder

@pytest.fixture
def sample_data():
    # Create a sample DataFrame for testing
    data = {
        'person_age': [25, 68, 35, 41, 30],
        'person_income': [33000, 88000, 25000, 56000, 75500],
        'person_home_ownership': ['OWN', 'RENT', 'OTHER', 'OWN', 'MORTGAGE'],
        'person_emp_length': [6.00, 15.00, 120.00, 20.00, 18.00],
        'loan_intent': ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'VENTURE'],
        'loan_grade': ['A', 'B', 'C', 'D', 'E'],
        'loan_amount': [6000, 1500, 9000, 3200, 18000],
        'loan_int_rate': [16.2, 10.2, 15.8, 14.3, 15.4],
        'loan_status': [0, 1, 0, 1, 0],
        'loan_percent_income': [0.2, 0.05, 0.42, 0.08, 0.2],
        'cb_person_default_on_file': ['N', 'N', 'Y', 'N', 'Y'],
        'cb_person_cred_hist_length': [3, 6, 5, 4, 6]
    }
    return pd.DataFrame(data)

@pytest.fixture
def result_data():
    # Create a result DataFrame for testing
    data = {
        'personage': [25, 68, 35, 41, 30],
        'personincome': [33000, 88000, 25000, 56000, 75500],
        'personhomeownershipOWN': [1, 0, 0, 1, 0],
        'personhomeownershipRENT': [0, 1, 0, 0, 0],
        'personhomeownershipOTHER': [0, 0, 1, 0, 0],
        'personhomeownershipMORTGAGE': [0, 0, 0, 0, 1],
        'personemplength': [6.0, 15.0, 120.0, 20.0, 18.0],
        'loanintentPERSONAL': [1, 0, 0, 0, 0],
        'loanintentEDUCATION': [0, 1, 0, 0, 0],
        'loanintentMEDICAL': [0, 0, 1, 0, 0],
        'loanintentVENTURE': [0, 0, 0, 1, 1],
        'loangrade': [1, 2, 3, 4, 5],
        'loanamount': [6000, 1500, 9000, 3200, 18000],
        'loanintrate': [16.2, 10.2, 15.8, 14.3, 15.4],
        'loanstatus': [0, 1, 0, 1, 0],
        'loanpercentincome': [0.2, 0.05, 0.42, 0.08, 0.2],
        'cbpersondefaultonfile': [0, 0, 1, 0, 1],
        'cbpersoncredhistlength': [3, 6, 5, 4, 6]}
    return pd.DataFrame(data)

def test_load_data(sample_data, tmp_path):
    # Save sample data to a temporary CSV file
    csv_path = tmp_path / "sample_data.csv"
    sample_data.to_csv(csv_path, index=False)

    # Test loading data from the temporary CSV file
    prep = DataPrep(data_path=csv_path)
    prep.load_data()

    # Assert that raw_data is not None
    assert prep.raw_data is not None

def test_encoder(sample_data):
    prep = DataPrep(data_path=None)
    prep.raw_data = sample_data

    prep.encoder()
    assert 'person_home_ownership_OWN' in prep.prepared_data.columns
    assert 'loan_intent_EDUCATION' in prep.prepared_data.columns
    assert 'cb_person_default_on_file_Y' in prep.prepared_data.columns

def test_loan_grade_prep(sample_data):
    prep = DataPrep(data_path=None)
    prep.raw_data = sample_data

    prep.load_data()
    prep.encoder()
    prep.loan_grade_prep()
    assert all(grade in [1, 2, 3] for grade in prep.prepared_data['loan_grade'])

def test_subs_char_names(sample_data):
    prep = DataPrep(data_path=None)
    prep.raw_data = sample_data

    prep.load_data()
    prep.encoder()
    prep.loan_grade_prep()
    prep.subs_char_names()
    
    assert 'personhomeownershipOWN' in prep.prepared_data.columns
    assert 'loanintentEDUCATION' in prep.prepared_data.columns
    assert 'cbpersondefaultonfileY' in prep.prepared_data.columns

def test_get_prepdata(sample_data):
    prep = DataPrep(data_path=None)
    prep.raw_data = sample_data

    prep.load_data()
    prep.encoder()
    prep.loan_grade_prep()
    prep.subs_char_names()

    result = prep.get_prepdata()
    assert result.shape == prep.prepared_data.shape
    assert result.equals(prep.prepared_data)