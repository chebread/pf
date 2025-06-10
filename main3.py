import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt # 데이터 시각화를 위해 matplotlib 추가
import matplotlib.font_manager as fm # 한글 폰트 설정을 위해 추가

# 한글 폰트 설정 (Windows, macOS, Linux 환경에 맞게 자동 설정)
try:
    font_name = None
    if plt.sys.platform == 'win32': # Windows
        font_name = 'Malgun Gothic'
    elif plt.sys.platform == 'darwin': # macOS
        font_name = 'AppleGothic'
    elif plt.sys.platform.startswith('linux'): # Linux
        # 사용 가능한 한글 폰트를 찾습니다. (Nanum 폰트 계열 우선)
        font_files = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        nanum_fonts = [f for f in font_files if 'NanumGothic' in f or 'NanumBarunGothic' in f]
        if nanum_fonts:
            font_name = fm.FontProperties(fname=nanum_fonts[0]).get_name()
        else: # Nanum 폰트가 없을 경우, 시스템의 다른 한글 폰트 시도
            korean_fonts = [f for f in font_files if 'DroidSansFallback' in f or 'UnDotum' in f or 'Baekmuk' in f]
            if korean_fonts:
                 font_name = fm.FontProperties(fname=korean_fonts[0]).get_name()

    if font_name:
        plt.rcParams['font.family'] = font_name
        print(f"한글 폰트 '{font_name}'으로 설정되었습니다.")
    else:
        print("경고: 적절한 한글 폰트를 찾지 못했습니다. 그래프의 한글이 깨지거나 기본 영문 폰트로 표시될 수 있습니다.")
        if plt.sys.platform.startswith('linux'):
            print("리눅스 사용자의 경우, 'sudo apt-get install fonts-nanum*' 등으로 Nanum 폰트 설치를 권장합니다.")

    plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지
except Exception as e:
    print(f"폰트 설정 중 오류 발생: {e}. 기본 폰트로 진행합니다.")


print("== 물가 예측 (2024.2 ~ 2025.1 까지의 데이터) ==\n0: 현미\n1: 무\n2: 휘발유")
cmd = input("품목 번호를 입력하세요 (0, 1, 또는 2): ")

file_names = ['./assets/brown-rice.csv', './assets/radish.csv', './assets/gasoline.csv']
item_names = ['현미', '무', '휘발유']

try:
    selected_item_index = int(cmd)
    if not (0 <= selected_item_index < len(file_names)):
        raise ValueError("잘못된 선택입니다")
    selected_file = file_names[selected_item_index]
    selected_item_name = item_names[selected_item_index]
except ValueError:
    print("잘못된 입력입니다. 0, 1, 또는 2를 입력해주세요.")
    exit()

try:
    df = pd.read_csv(selected_file)
except FileNotFoundError:
    print(f"오류: {selected_file} 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    print("팁: CSV 파일들이 스크립트와 동일한 디렉토리에 있거나, './assets/' 폴더 내에 있는지 확인하세요.")
    exit()

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')
base_date = df['date'].iloc[0]
df['days'] = (df['date'] - base_date).dt.days

# 1. 환율 예측 모델 (시간('days')에 따른 환율 변화)
X_exchange_time = df['days'].values.reshape(-1, 1)
y_exchange_val = df['exchange'].values
exchange_model_time = LinearRegression()
exchange_model_time.fit(X_exchange_time, y_exchange_val)

# 2. 물가 예측 모델 (시간('days')과 환율('exchange')에 따른 물가 변화)
X_price_multivar = df[['days', 'exchange']].values
y_price_val = df['price'].values
price_model_multivar = LinearRegression()
price_model_multivar.fit(X_price_multivar, y_price_val)

# 과거 데이터에서 물가와 환율 간의 상관계수 계산
correlation, _ = pearsonr(df['exchange'], df['price'])
print(f"\n=== {selected_item_name} 데이터 분석 정보 ===")
print(f"과거 데이터에서 '{selected_item_name}' 가격과 환율 간의 Pearson 상관계수: {correlation:.4f}")
if abs(correlation) > 0.7:
    print("해석: 가격과 환율 간에 강한 선형 관계가 있는 것으로 보입니다.")
elif abs(correlation) > 0.4:
    print("해석: 가격과 환율 간에 어느 정도 선형 관계가 있는 것으로 보입니다.")
else:
    print("해석: 가격과 환율 간의 선형 관계가 약하거나 거의 없는 것으로 보입니다.")

# --- 데이터 시각화 (두 개의 그래프 순차 표시) ---
print("\n과거 데이터의 환율과 물가 관계를 시각화합니다.")
print("첫 번째 그래프(추세선 포함) 창을 닫으면 두 번째 그래프(산점도)가 나타납니다.")

# --- 1. 산점도 + 추세선 그래프 ---
X_trend_scatter = df['exchange'].values.reshape(-1, 1)
y_trend_scatter = df['price'].values
trend_model_scatter = LinearRegression()
trend_model_scatter.fit(X_trend_scatter, y_trend_scatter)
exchange_range_for_trendline = np.array([df['exchange'].min(), df['exchange'].max()]).reshape(-1, 1)
price_pred_for_trendline = trend_model_scatter.predict(exchange_range_for_trendline)

plt.figure(figsize=(10, 6))
plt.scatter(df['exchange'], df['price'], alpha=0.7, label=f'{selected_item_name} 실제 데이터')
plt.plot(exchange_range_for_trendline, price_pred_for_trendline, color='red', linestyle='--', linewidth=2, label='추세선 (환율-물가)')
plt.title(f'{selected_item_name} 물가와 환율 관계 (추세선 포함)')
plt.xlabel('환율 (원)')
plt.ylabel(f'{selected_item_name} 물가 (원)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() # 첫 번째 그래프 표시 (창을 닫아야 다음으로 진행)

# --- 2. 산점도만 있는 그래프 ---
plt.figure(figsize=(10, 6)) # 새 그래프 창 생성
plt.scatter(df['exchange'], df['price'], alpha=0.7, label=f'{selected_item_name} 실제 데이터')
plt.title(f'{selected_item_name} 물가와 환율 관계 (산점도)')
plt.xlabel('환율 (원)')
plt.ylabel(f'{selected_item_name} 물가 (원)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() # 두 번째 그래프 표시 (창을 닫아야 다음으로 진행)

# 예측
date_input_str = input("미래 날짜 입력 (YYYY-MM-DD 형식, 예: 2025-05-07): ")
try:
    future_date = pd.to_datetime(date_input_str)
except ValueError:
    print("잘못된 날짜 형식입니다. YYYY-MM-DD 형식으로 입력해주세요.")
    exit()

if future_date <= df['date'].iloc[-1]:
    print(f"경고: 입력하신 날짜({future_date.strftime('%Y-%m-%d')})는 학습 데이터의 마지막 날짜({df['date'].iloc[-1].strftime('%Y-%m-%d')})보다 이전이거나 같습니다.")
    print("예측은 가능하지만, 이는 미래 예측이 아닌 과거 또는 현재 데이터에 대한 예측일 수 있습니다.")

future_days = (future_date - base_date).days

# 1. 입력된 미래 날짜의 예상 환율 예측
predicted_future_exchange = exchange_model_time.predict(np.array([[future_days]]))[0]

# 2. 예상 환율과 미래 날짜를 사용하여 물가 예측
predicted_future_price = price_model_multivar.predict(np.array([[future_days, predicted_future_exchange]]))[0]

print(f"\n=== {selected_item_name} 물가 예측 결과 ===")
print(f"입력 날짜: {future_date.strftime('%Y-%m-%d')} (기준일로부터 {future_days}일 경과)")
print(f"해당 날짜의 예상 환율: {predicted_future_exchange:.2f} 원")
print(f"해당 날짜의 예상 '{selected_item_name}' 가격: {predicted_future_price:.2f} 원")