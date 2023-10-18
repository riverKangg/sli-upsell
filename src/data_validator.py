import pandas as pd
from utils import *

class DataValidator:
    def __init__(self, df):
        """
        Initialize the DataValidator with a DataFrame.

        :param df: The DataFrame to be validated.
        """
        self.df = df

    def are_columns_included(self, lst):
        """
        Check if the DataFrame includes the specified columns.

        :param lst: A list of column names to be checked.
        :return: True if all columns are included, False otherwise.
        """
        missing_columns = list(filter(lambda column: column not in self.df.columns, lst))
        return len(missing_columns) == 0

    def check_for_null(self, col_list):
        """
        Check if the specified columns contain any null values.

        :param col_list: A list of column names to be checked for null values.
        :return: True if no null values are found, False otherwise.
        """
        for column in col_list:
            if self.df[column].isnull().any():
                return False
        return True

    def is_duplicate(self, data):
        """
        Check for duplicate rows in the DataFrame.

        :param data: A subset of the DataFrame to check for duplicates.
        :return: True if duplicates are found, False otherwise.
        """
        is_duplicate = data.duplicated().any()
        return is_duplicate

    def validate_modeling_data(self):
        """
        Validate modeling data in the DataFrame.

        :return: A tuple containing a boolean indicating data validity and a message describing the validation result.
        """
        if not self.are_columns_included(data_keys):
            return False, "Missing key values."
        if not self.check_for_null(data_keys):
            return False, "Key values contain NULL."
        if self.is_duplicate(self.df['계약자고객ID']):
            return False, "Duplicate values found in '계약자고객ID'."
        if not self.are_columns_included(['PERF']):
            return False, "No target data ('PERF') found."
        if not self.check_for_null(['PERF']):
            return False, "Target data ('PERF') contains NULL."
        return True, "Data is valid."

    def validate_score_data(self):
        """
        Validate scoring data in the DataFrame.

        :return: A tuple containing a boolean indicating data validity and a message describing the validation result.
        """
        if not self.are_columns_included(data_keys):
            return False, "Missing key values."
        if not self.check_for_null(data_keys):
            return False, "Key values contain NULL."
        if self.is_duplicate(self.df['계약자고객ID']):
            return False, "Duplicate values found in '계약자고객ID'."
        if self.are_columns_included(['PERF']):
            return False, "Target data ('PERF') should be removed."
        return True, f"{self.df}: Data is valid."

# Test the DataValidator with different datasets.
if __name__ == "__main__":
    datasets = ['sample_data_202211.csv', 'sample_data_202304.csv', 'sample_score_data_202306.csv']
    for dataset in datasets:
        df = pd.read_csv(f'{data_path}/{dataset}')
        validator = DataValidator(df)
        if 'score' not in dataset:
            is_valid, message = validator.validate_modeling_data()
        else:
            is_valid, message = validator.validate_score_data()

        if is_valid:
            print(f"{dataset}: Data is valid.")
        else:
            print(f"Data validation error: {message}")