import pandas as pd
import numpy as np
import random
import string
import seaborn as sns

class DataGenerator:
    def __init__(self, num_samples=1000, yyyymm='202304'):
        self.num_samples = num_samples
        self.yyyymm = yyyymm
        self.df = None

    def generate_sample_data(self):
        # '계약자고객ID' 생성 (고유한 숫자)
        contract_ids = list(range(1, self.num_samples + 1))

        # '주민등록번호암호화' 생성 (임의의 문자열)
        def random_string(length):
            letters = string.ascii_letters + string.digits
            return ''.join(random.choice(letters) for _ in range(length))

        encryption = [random_string(10) for _ in range(self.num_samples)]

        # 데이터프레임 생성
        data = {
            '마감년월': self.yyyymm,
            '계약자고객ID': contract_ids,
            '주민등록번호암호화': encryption,
        }

        self.df = pd.DataFrame(data)

    def load_titanic_data(self):
        titanic_df = sns.load_dataset('titanic')
        titanic_df = titanic_df.rename(columns={'survived': 'PERF'})
        titanic_df = titanic_df.sample(frac=1).reset_index(drop=True)
        titanic_df = titanic_df.iloc[:self.num_samples]
        return titanic_df

    def join_titanic_data(self):
        titanic_df = self.load_titanic_data()
        self.df = self.df.join(titanic_df)

    def save_data_to_csv(self, filename):
        self.df.to_csv(filename, index=False)

if __name__ == "__main__":
    data_generator = DataGenerator(num_samples=10000, yyyymm='202304')
    data_generator.generate_sample_data()
    data_generator.join_titanic_data()
    data_generator.save_data_to_csv('../data/sample_data.csv')
