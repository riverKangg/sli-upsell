import sys
import warnings
import pandas as pd
from unidecode import unidecode
from sklearn.model_selection import train_test_split
from utils import *

warnings.filterwarnings(action='ignore')

class DataPreprocessor:
    def __init__(self, train_data, test_data=None, columns_to_keep=None, target_col="PERF"):
        """
        Initialize the DataPreprocessor.

        :param train_data: The training dataset as a DataFrame.
        :param test_data: The test dataset as a DataFrame (optional, only for development data preparation).
        :param target_col: The target column name in the dataset.
        :param columns_to_keep: A list of columns to keep in the dataset (optional).
        """
        self.target_col = target_col

        if columns_to_keep:
            remain_cols = list(filter(lambda x: x not in train_data.columns, columns_to_keep))
            if not remain_cols:
                self.before_dum = True
            else:
                self.before_dum = False
        else:
            self.before_dum = False
        self.columns_to_keep = columns_to_keep

        self.is_dev_data = test_data is not None

        if self.is_dev_data:
            self.train_data = train_data.drop(columns=[target_col])
            self.test_data = test_data.drop(columns=[target_col])
            self.train_y = train_data[target_col]
            self.Y_test = test_data[target_col]
        else:
            self.train_data = train_data

        if test_data is not None and (target_col not in train_data.columns or target_col not in test_data.columns):
            print('Check target column.')
            sys.exit()

    def _check_column_matching(self):
        """
        Check if the columns in the training and test datasets match.
        """
        if not self.is_dev_data:
            return

        if not self.train_data.columns.equals(self.test_data.columns):
            print('Dataset mismatch.')
            sys.exit()

    def _process_data(self, data):
        """
        Preprocess the data, including handling missing values and one-hot encoding.

        :param data: The input data as a DataFrame.
        :return: Processed data as a DataFrame.
        """
        data = data[self.columns_to_keep] if self.columns_to_keep and self.before_dum else data
        data = data.drop(columns=data_keys)
        data = data.fillna(0)

        data = pd.get_dummies(data)

        for col in data.columns:
            if any(ord(c) > 127 for c in col):
                new_col = unidecode(col)
                data = data.rename(columns={col: new_col})

        data = data[self.columns_to_keep] if self.columns_to_keep and not self.before_dum else data

        return data

    def prepare_dev_data(self):
        """
        Prepare development data, splitting it into training and validation sets.

        :return: X_train, X_val, Y_train, Y_val, X_test, Y_test
        """
        self._check_column_matching()

        X_train, X_val, Y_train, Y_val = train_test_split(
            self._process_data(self.train_data),
            self.train_y,
            test_size=0.3,
            random_state=42
        )

        X_test = self._process_data(self.test_data)

        common_features = list(set(X_train.columns) & set(X_test.columns))
        X_train = X_train[common_features]
        X_val = X_val[common_features]
        X_test = X_test[common_features]

        return X_train, X_val, Y_train, Y_val, X_test, self.Y_test

    def prepare_scoring_data(self):
        """
        Prepare data for scoring, excluding the target column.

        :return: Processed scoring data as a DataFrame.
        """
        self._check_column_matching()

        return self._process_data(self.train_data)

    def process_data(self):
        """
        Process the data based on the context (development or scoring).

        :return: Processed data based on the context.
        """
        if self.is_dev_data:
            return self.prepare_dev_data()
        else:
            return self.prepare_scoring_data()

if __name__ == "__main__":
    trainset = pd.read_csv('./data/sample_data_202211.csv')
    testset = pd.read_csv('./data/sample_data_202304.csv')
    dp = DataPreprocessor(trainset, testset)
    X_train, X_val, Y_train, Y_val, X_test, Y_test = dp.process_data()

    dp2 = DataPreprocessor(trainset.drop(columns=['PERF']))
    scoring_dataset = dp2.process_data()