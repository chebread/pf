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
        font_files = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        nanum_fonts = [f for f in font_files if 'NanumGothic' in f]
        if nanum_fonts:
            font_name = fm.FontProperties(fname=nanum_fonts[0]).get_name()

    if font_name:
        plt.rcParams['font.family'] = font_name
        print(f"한글 폰트 '{font_name}'으로 설정되었습니다.")
    else:
        # 적절한 폰트를 찾지 못한 경우, 나눔고딕 설치 권장 메시지 표시
        print("경고: 적절한 한글 폰트를 찾지 못했습니다. 그래프의 한글이 깨질 수 있습니다.")
        if plt.sys.platform.startswith('linux'):
            print("리눅스 사용자의 경우, 'sudo apt-get install fonts-nanum*' 명령어로 나눔 폰트 설치를 권장합니다.")
    # 마이너스 부호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"폰트 설정 중 오류 발생: {e}. 기본 폰트로 진행합니다.")


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


# --- matplotlib 데이터 시각화 (환율 vs 물가 산점도) ---
print("\n과거 데이터의 환율과 물가 관계를 시각화합니다...")

plt.figure(figsize=(10, 6)) # 그래프 크기 설정
# x축은 'exchange', y축은 'price'로 하는 산점도 그래프
plt.scatter(df['exchange'], df['price'], alpha=0.7, label=f'{selected_item_name} 실제 데이터')
plt.title(f'{selected_item_name} 물가와 환율 관계')
plt.xlabel('환율 (원)')
plt.ylabel(f'{selected_item_name} 물가 (원)')
plt.legend()
plt.grid(True)
plt.show() # 그래프 보여주기 (이 창을 닫아야 다음 단계로 진행됩니다)


# 예측
date_input_str = input("미래 날짜 입력 (YYYY-MM-DD 형식, 예: 2025-05-07): ")
try:
    future_date = pd.to_datetime(date_input_str)
except ValueError:
    print("잘못된 날짜 형식입니다.")