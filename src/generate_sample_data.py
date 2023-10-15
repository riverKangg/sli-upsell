import pandas as pd
import numpy as np
import random
import string

# 샘플 데이터 생성
num_samples = 100  # 생성할 샘플 수

# '마감년월' 생성 (날짜 형식)
end_dates = pd.date_range(start="2022-01-01", periods=num_samples, freq='M')

# '계약자고객ID' 생성 (고유한 숫자)
contract_ids = list(range(1, num_samples + 1))

# '주민등록번호암호화' 생성 (임의의 문자열)
def random_string(length):
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

encryption = [random_string(10) for _ in range(num_samples)]

#
perf = np.random.randint(0, 1, size=num_samples)

# 데이터프레임 생성
data = {
    '마감년월': end_dates,
    '계약자고객ID': contract_ids,
    '주민등록번호암호화': encryption,
    'PERF' : perf
}

df = pd.DataFrame(data)

# 숫자형 컬럼 추가
df['숫자형컬럼'] = np.random.randint(1, 100, size=num_samples)

# 범주형 컬럼 추가
categories = ['카테고리A', '카테고리B', '카테고리C']
df['범주형컬럼'] = [random.choice(categories) for _ in range(num_samples)]

# 문자형 컬럼 추가
strings = [random_string(8) for _ in range(num_samples)]
df['문자형컬럼'] = strings

# CSV 파일로 저장
df.to_csv('../data/sample_data.csv', index=False)
