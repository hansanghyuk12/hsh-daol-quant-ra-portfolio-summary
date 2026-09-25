import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
from CustomHMM import CustomHMM
import yfinance as yf
import seaborn as sns
from sklearn.preprocessing import StandardScaler
sns.set()


def select_period_for_rollingna(train_start, train_end, backward) : 
    
    # Backward Month 계산 구조
    months = np.array(range(1,13))

    # [롤링 결측값을 고려한 Train / Test Perid 조정] 
    train_end_month = int(train_end[5:7])
    train_end_year = int(train_end[0:4])

    train_start_month = int(train_start[5:7])
    train_start_year = int(train_start[0:4])

    idx_end = np.where(months == train_end_month)[0][0]
    idx_start = np.where(months == train_start_month)[0][0]

    test_back_to_idx = idx_end - backward
    train_back_to_idx = idx_start - backward

    test_start_year = train_end_year - 1  if test_back_to_idx < 0 else train_end_year 
    train_start_year = train_start_year - 1  if train_back_to_idx < 0 else train_start_year 

    test_start_month = str(months[test_back_to_idx]).zfill(2)
    train_start_month = str(months[train_back_to_idx]).zfill(2)

    test_start_day = train_end[8:10]
    train_start_day = train_start[8:10]

    test_start = f'{test_start_year}-{test_start_month}-{test_start_day}'
    train_start = f'{train_start_year}-{train_start_month}-{train_start_day}'

    return train_start, test_start

def hmm_stock_data(ticker, train_start, train_end, market = 'sp500', vol_period = 3, 
                   sr_period = 3, return_type = 'all', scale = False) :
    
    # 지수에 따른 티커 수정
    ticker = ticker if market in ['sp500', 'index'] else f'{ticker}.KS' if market == 'kospi' else ticker

    # 롤링 결측값을 고려한 Train / Test Perid 조정
    backward = np.array([vol_period, sr_period]).max()
    train_start, test_start = select_period_for_rollingna(train_start, train_end, backward)

    # end 인자가 부여된 경우 : 특정 기간 / # end 인자가 부여되지 않은 경우 : 시작 기간부터 현재
    train_ohlcv_data = yf.download(ticker, start = train_start, end = train_end, auto_adjust=True) 
    test_ohlcv_data = yf.download(ticker, start = test_start, auto_adjust=True)
    
    # 종가 데이터 필터링
    close_train = train_ohlcv_data['Close']
    close = test_ohlcv_data['Close']
    
    # [HMM 분석용 데이터 프레임 생성]
    
    # 0. 종가 가격 -> 추후 제거
    train_data = close_train.copy()  
    y = close.copy()  
    
    # 1. 수익률 
    train_data['rets'] = train_data.pct_change().fillna(0)
    y['rets'] = y.pct_change().fillna(0)

    # 2. n개월 롤링 실현 변동성
    vol_lookback = 21 * vol_period
    train_data[f'{vol_period}m_vol'] = train_data['rets'].rolling(vol_lookback).std() * np.sqrt(252)
    y[f'{vol_period}m_vol'] = y['rets'].rolling(vol_lookback).std() * np.sqrt(252)

    # 3. n개월 샤프비율 
    sr_lookback = 21 * sr_period
    train_data[f'{sr_period}m_sr'] = train_data['rets'].rolling(sr_lookback).mean() * np.sqrt(252) / train_data['rets'].rolling(sr_lookback).std()
    y[f'{sr_period}m_sr'] = y['rets'].rolling(sr_lookback).mean() * np.sqrt(252) / y['rets'].rolling(sr_lookback).std()
    
    # Close 가격 칼럼 제거
    train_data.drop(ticker, axis = 1, inplace=True)
    y.drop(ticker, axis = 1, inplace=True)

    # 롤링으로 인한 결측값 제거 
    train_data.dropna(inplace=True)
    y.dropna(inplace=True)

    # 칼럼 대표 이름 티커로 변경
    train_data.columns.name = ticker
    y.columns.name = ticker

    # 종가 데이터프레임 칼럼 이름 변경 및 칼럼 이름 price 변경
    close_train.columns.name = ticker
    close_train.rename(columns={ticker : 'price'}, inplace=True)
    close.columns.name = ticker
    close.rename(columns={ticker : 'price'}, inplace=True)

    # 종가 데이터 프레임 인덱스 통일
    common_train = close_train.index.intersection(train_data.index)
    close_train = close_train.loc[common_train]
    common_test = close.index.intersection(y.index)
    close = close.loc[common_test]

    # 스케일링 된 변수를 사용하는 경우 
    if scale :  
        # 스케일러 생성
        scaler = StandardScaler()
        # Train Data 스케일링
        train_data_arr = scaler.fit_transform(train_data)
        train_data = pd.DataFrame(train_data_arr, index=train_data.index, columns=train_data.columns)
        # Test Data 스케일링
        y_arr = scaler.fit_transform(y)
        y = pd.DataFrame(y_arr, index=y.index, columns=y.columns)
    else : 
        pass

    if return_type == 'all' :
        return close_train, train_data, close, y
    else :
        return train_data, y

# MDD 계산 함수
def cal_historical_mdd(port_rets) :
    
    # 포트폴리오 누적 수익률
    port_cum_rets = (1 + port_rets).cumprod()

    # 포트폴리오 역사적 전고점 계산
    port_hwm = port_cum_rets.cummax()

    # 낙폭(DD) 계산
    port_dd = (port_cum_rets / port_hwm) - 1

    # 최대낙폭(MDD) 계산
    port_mdd = port_dd.min()

    return port_mdd, port_dd

def show_price_by_regime(df, path_col, price_col = 'price') :
    
    # 상태 개수 지정
    state_num = len(df[path_col].unique())

    # 존재하는 상태 추출
    state_existed = sorted(df[path_col].unique())

    # 상태 별 색깔 추출
    pal = sns.color_palette('Set1', state_num)

    # 2열 기준 행 수 지정
    subplot_row = int(np.ceil(state_num/2)) + 1 if state_num / 2 == np.ceil(state_num/2) else int(np.ceil(state_num/2)) 

    # 상태 별 가격 시각화 
    plt.figure(figsize=(20,10))

    # Summary Plot
    plt.subplot(subplot_row, 2, 1)
    plt.plot(df[price_col],c = 'grey', alpha = 0.3, label = 'Price')
    for i, state in enumerate(state_existed) :
        idx = df[path_col] == state
        index_by_state = df.index[idx]
        price_by_state = df.loc[idx, 'price']
        plt.scatter(index_by_state, price_by_state, s = 10, label = f'State {state}', c = pal[i])
    plt.title(f'{path_col} : Total Combined States')
    plt.legend()

    # Plot by Regime/State
    for i, state in enumerate(state_existed) :
        plt.subplot(subplot_row, 2, i+2) 
        plt.plot(df[price_col],c = 'grey', alpha = 0.3, label = 'Price')
        idx = df[path_col] == state
        index_by_state = df.index[idx]
        price_by_state = df.loc[idx, 'price']
        plt.scatter(index_by_state, price_by_state, s = 10, label = f'State {state}', c = pal[i])
        plt.title(f'{path_col} : Regime {state} Price')
        plt.legend()
    
    # 배치 최적화 
    plt.tight_layout()
    plt.show()


    