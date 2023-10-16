
import re
import pandas as pd


class DataValidator:
    def __init__(self, df):
        self.df = df
        self.required_columns = ['마감년월', '계약자고객ID', '계약자주민등록번호암호화']

    def are_columns_included(self):
        missing_columns = list(filter(lambda column: column not in df.columns, self.required_columns))
        return len(missing_columns) == 0

    def check_for_null(self):
        for column in self.required_columns:
            if self.df[column].isnull().any():
                return False
        return True

    def is_duplicate(self, data):
        is_duplicate = data.duplicated().any()
        return is_duplicate

    def validate_data(self):
        if not self.are_columns_included():
            return False, "누락된 키값이 있습니다."
        if not self.check_for_null():
            return False, "키값에 NULL이 포함되어 있습니다."
        if self.is_duplicate(self.df['계약자고객ID']):
            return False, "계약자고객ID에 중복이 있습니다."

        return True, "데이터가 유효합니다."


# 모듈을 다른 스크립트에서 사용할 수 있도록 테스트 코드를 작성합니다.
if __name__ == "__main__":
    df = pd.read_csv('../data/sample_data.csv')

    validator = DataValidator(df)
    is_valid, message = validator.validate_data()

    if is_valid:
        print("데이터가 유효합니다.")
    else:
        print(f"데이터 유효성 검사 오류: {message}")
