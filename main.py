import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

print("== 물가 예측 (2024.2 ~ 2025.1 까지의 데이터) ==\n0: 금\n1: 무\n2: 휘발류")
cmd = input() # 0, 1, 2

# 파일명 매핑
file_names = ['./assets/gold.csv', './assets/radish.csv', './assets/gasoline.csv']

# CSV 파일 불러오기
df = pd.read_csv(file_names[int(cmd)])
df['date'] = pd.to_datetime(df['date'])
base_date = df['date'][0]
df['days'] = (df['date'] - base_date).dt.days

# X, y 준비
data_X = df['days'].values.reshape(-1, 1)
data_y = df['price'].values

# 모델 생성 및 학습
model = LinearRegression()
model.fit(data_X, data_y)

# 예측
date = input("날짜 입력(2025-05-07 형식): ")
may_date = pd.to_datetime(date)
may_days = (may_date - base_date).days
may_pred = model.predict(np.array([[may_days]]))

print(f"=== {['금', '무', '휘발류'][int(cmd)]} 데이터 예측 ===")
print(f'{may_date} ({may_days}일 후) 예측값: {may_pred[0]:.2f}')
