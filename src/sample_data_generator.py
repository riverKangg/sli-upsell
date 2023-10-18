import os
import random
import string
import pandas as pd
import seaborn as sns

from utils import *

class DataGenerator:
    def __init__(self, num_samples=1000, yyyymm='202304'):
        """
        Initialize the DataGenerator.

        :param num_samples: The number of samples to generate.
        :param yyyymm: The year and month (e.g., '202304').
        """
        self.num_samples = num_samples
        self.yyyymm = yyyymm
        self.df = None

    def generate_sample_data(self):
        """
        Generate sample data with '마감년월', '계약자고객ID', and '계약자주민등록번호암호화'.
        """
        # Generate unique '계약자고객ID' (unique numbers)
        customer_ids = list(range(1, self.num_samples + 1))

        # Generate '계약자주민등록번호암호화' (random strings)
        def random_string(length):
            letters = string.ascii_letters + string.digits
            return ''.join(random.choice(letters) for _ in range(length))

        encryption = [random_string(10) for _ in range(self.num_samples)]

        # Create a DataFrame
        data = {
            '마감년월': self.yyyymm,
            '계약자고객ID': customer_ids,
            '계약자주민등록번호암호화': encryption,
        }

        self.df = pd.DataFrame(data)

    def load_titanic_data(self):
        """
        Load and preprocess Titanic dataset.
        """
        titanic_df = sns.load_dataset('titanic')
        titanic_df = titanic_df.rename(columns={'survived': 'PERF'}).drop(columns=['alive'])
        titanic_df = titanic_df[titanic_df.age>0]
        titanic_df = titanic_df.sample(frac=1).reset_index(drop=True)
        titanic_df = titanic_df.iloc[:self.num_samples]
        return titanic_df

    def join_titanic_data(self):
        """
        Join Titanic dataset with the generated data.
        """
        titanic_df = self.load_titanic_data()
        self.df = self.df.join(titanic_df)

    def save_data_to_csv(self, filepath=data_path):
        """
        Save generated data to a CSV file.

        :param filepath: The path to save the CSV file (default is data_path).
        """
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        filename = f'{filepath}/sample_data_{self.yyyymm}.csv'
        self.df.to_csv(filename, index=False)

    def save_scoring_data_to_csv(self, filepath=data_path):
        """
        Save the scoring data (without 'PERF') to a CSV file.

        :param filepath: The path to save the CSV file (default is data_path).
        """
        self.df = self.df.drop(columns=['PERF'])
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        filename = f'{filepath}/sample_score_data_{self.yyyymm}.csv'
        self.df.to_csv(filename, index=False)


if __name__ == "__main__":
    data_generator = DataGenerator(num_samples=500, yyyymm='202211')
    data_generator.generate_sample_data()
    data_generator.join_titanic_data()
    data_generator.save_data_to_csv()

    data_generator = DataGenerator(num_samples=500, yyyymm='202304')
    data_generator.generate_sample_data()
    data_generator.join_titanic_data()
    data_generator.save_data_to_csv()

    data_generator = DataGenerator(num_samples=500, yyyymm='202306')
    data_generator.generate_sample_data()
    data_generator.join_titanic_data()
    data_generator.save_scoring_data_to_csv()