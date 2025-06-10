import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr # Added for calculating correlation

print("== 물가 예측 (2024.2 ~ 2025.1 까지의 데이터) ==\n0: 현미\n1: 무\n2: 휘발유")
cmd = input("품목 번호를 입력하세요 (0, 1, 또는 2): ") # 0, 1, 2

# 파일명 매핑
file_names = ['./assets/brown-rice.csv', './assets/radish.csv', './assets/gasoline.csv']
item_names = ['현미', '무', '휘발유'] # For clearer output

try:
    selected_item_index = int(cmd)
    if not (0 <= selected_item_index < len(file_names)):
        raise ValueError("Invalid selection")
    selected_file = file_names[selected_item_index]
    selected_item_name = item_names[selected_item_index]
except ValueError:
    print("잘못된 입력입니다. 0, 1, 또는 2를 입력해주세요.")
    exit()

# CSV 파일 불러오기
try:
    df = pd.read_csv(selected_file)
except FileNotFoundError:
    print(f"오류: {selected_file} 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    print("팁: CSV 파일들이 스크립트와 동일한 디렉토리에 있거나, './assets/' 폴더 내에 있는지 확인하세요.")
    exit()

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date') # Ensure data is sorted by date
base_date = df['date'].iloc[0]
df['days'] = (df['date'] - base_date).dt.days

# X, y 준비
# 1. 환율 예측 모델 (시간에 따른 환율 변화)
X_exchange = df['days'].values.reshape(-1, 1)
y_exchange = df['exchange'].values

exchange_model = LinearRegression()
exchange_model.fit(X_exchange, y_exchange)

# 2. 물가 예측 모델 (시간과 환율에 따른 물가 변화)
# 여기서 X는 'days'와 'exchange' 두 가지 특성을 가집니다.
X_price = df[['days', 'exchange']].values
y_price = df['price'].values

price_model = LinearRegression()
price_model.fit(X_price, y_price)

# 과거 데이터에서 물가와 환율 간의 상관계수 계산 및 출력
correlation, _ = pearsonr(df['exchange'], df['price'])
print(f"\n=== {selected_item_name} 데이터 분석 정보 ===")
print(f"과거 데이터에서 '{selected_item_name}' 가격과 환율 간의 Pearson 상관계수: {correlation:.4f}")
if abs(correlation) > 0.7:
    print("해석: 가격과 환율 간에 강한 상관관계가 있습니다.")
elif abs(correlation) > 0.4:
    print("해석: 가격과 환율 간에 어느 정도 상관관계가 있습니다.")
else:
    print("해석: 가격과 환율 간의 상관관계가 약하거나 없습니다.")

# 예측
date_input_str = input("미래 날짜 입력 (YYYY-MM-DD 형식, 예: 2025-05-07): ")
try:
    future_date = pd.to_datetime(date_input_str)
except ValueError:
    print("잘못된 날짜 형식입니다. YYYY-MM-DD 형식으로 입력해주세요.")
    exit()

if future_date <= df['date'].iloc[-1]:
    print(f"경고: 입력하신 날짜({future_date.strftime('%Y-%m-%d')})는 학습 데이터의 마지막 날짜({df['date'].iloc[-1].strftime('%Y-%m-%d')})보다 이전이거나 같습니다.")
    print("예측은 가능하지만, 미래 예측이 아닌 점 참고바랍니다.")


future_days = (future_date - base_date).days

# 1. 입력된 미래 날짜의 예상 환율 예측
predicted_future_exchange = exchange_model.predict(np.array([[future_days]]))[0]

# 2. 예상 환율과 미래 날짜를 사용하여 물가 예측
#    price_model은 [days, exchange] 두 특성을 기대합니다.
predicted_future_price = price_model.predict(np.array([[future_days, predicted_future_exchange]]))[0]

print(f"\n=== {selected_item_name} 물가 예측 결과 ===")
print(f"입력 날짜: {future_date.strftime('%Y-%m-%d')} ({future_days}일 경과)")
print(f"해당 날짜의 예상 환율: {predicted_future_exchange:.2f}")
print(f"해당 날짜의 예상 '{selected_item_name}' 가격: {predicted_future_price:.2f}")

# - [] matplotlib 으로 데이터 시각화하기 (This part is still a to-do as in your original comment)