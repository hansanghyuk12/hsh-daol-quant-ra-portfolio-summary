import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
import yfinance as yf
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import seaborn as sns
import os 
import time
sns.set()


class SharpeLeadingAnalysis :

    """ 생성자 ( 초기화 함수 ) """
    def __init__(self, prices, sector_dict, target_market) :
        
        self.prices = prices
        self.target_market = target_market
        self.sector_dict = sector_dict
        self.close = self.prices['Close']
        self.rets = self.close.pct_change().fillna(0)
        self.ma5 = self.close.rolling(5).mean()
        self.ma20 = self.close.rolling(20).mean()
        self.ma60 = self.close.rolling(60).mean()
        self.ma120 = self.close.rolling(120).mean()
    

    """ StopLoss 계산을 위한 ARC 구하는 함수 """
    def cal_ARC(self, ticker, prices, arc_param = 3, lookback = 14) :
    
        # True Range
        def cal_true_range(target_date, stock_price_df) :

            # Today/Yesterday Date idx
            today_date_idx = stock_price_df.index.get_loc(target_date)
            yesterday_date_idx = today_date_idx - 1

            # t-high, t-low, {t-1}-close
            t_high = stock_price_df['High'].iloc[today_date_idx]
            t_low = stock_price_df['Low'].iloc[today_date_idx]
            t1_close = stock_price_df['Close'].iloc[yesterday_date_idx]

            # Calculate True Range
            true_range = np.max([np.abs(t_high - t_low), np.abs(t1_close - t_high), np.abs(t1_close - t_low)])

            return true_range

        # OHLC 데이터 분리
        closes = prices['Close']
        highs = prices['High']
        lows = prices['Low']
        opens = prices['Open']

        # 특정 종목의 OHLC 데이터 추출 
        close = closes[ticker]
        high = highs[ticker]
        low = lows[ticker]
        open_price = opens[ticker]

        # 특정 종목의 OHLC 가격 데이터 프레임
        stock_price_df = pd.DataFrame(close)
        stock_price_df.columns = ['Close']
        stock_price_df['Open'] = open_price
        stock_price_df['Low'] = low
        stock_price_df['High'] = high
        stock_price_df.columns.name = ticker

        # True Range, ATR, ARC = 3ATR
        stock_price_df['TR'] = pd.Series({target_date : cal_true_range(target_date, stock_price_df) for i, target_date in enumerate(close.index) if i != 0})
        stock_price_df.dropna(inplace=True)
        stock_price_df['ATR'] = stock_price_df['TR'].rolling(lookback).mean()
        stock_price_df[f'ARC'] = stock_price_df['ATR'] * arc_param

        return stock_price_df

    """일반 Rolling Sharpe Ratio 계산 함수"""
    def rolling_sr(self, lookback) :

        # SR
        rolling_sr = (self.rets.rolling(lookback).mean() * np.sqrt(252)) / self.rets.rolling(lookback).std() 

        # SR 랭킹
        sr_rank = rolling_sr.rank(axis=1, ascending=False)

        return rolling_sr, sr_rank
    
    """Exponential Weighted Rolling Sharpe Ratio 계산 함수"""
    def rolling_ewsr(self, lookback) :

        # EWSR
        rolling_ewsr = (self.rets.ewm(span=lookback).mean() * np.sqrt(252)) / self.rets.ewm(span=lookback).std() 

        # EWSR 랭킹
        ewsr_rank = rolling_ewsr.rank(axis=1, ascending=False)
        
        return rolling_ewsr, ewsr_rank
    
    """ 단일 lookback 샤프비율 주도주 분석 기초 데이터 프레임 생성 함수 """
    def make_sr_leading_df(self, rolling_sr_df, sr_rank_df,  target_date , sr_type = 'SR', top= 20, arc_param = 1, lookback_type = '6M') :
        
        
        print(f'{self.target_market} {lookback_type} {sr_type} 주도주 분석 기초 데이터프레임 생성 중')
        # 전고점 계산 기간 지정
        if lookback_type == '6M' :
            prev_high_period = 20 * 6
        elif lookback_type == '3M' :
            prev_high_period = 20 * 3
        elif lookback_type == '1M' :
            prev_high_period = 20 * 1
        elif lookback_type == '1W' :
            prev_high_period = 5
        else : 
            raise ValueError('Invalid Lookback Type')

        # 특징 일자 기준 주도주 분석 : 기초 데이터
        top_rank_by_date = (sr_rank_df[sr_rank_df <= top]).loc[target_date]
        top_sr_by_date = (rolling_sr_df[sr_rank_df <= top]).loc[target_date]
        top_ret_by_date = (self.rets[sr_rank_df <= top]).loc[target_date]
        top_close_by_date = (self.close[sr_rank_df <= top]).loc[target_date]
        top_ma5_by_date = (self.ma5[sr_rank_df <= top]).loc[target_date]
        top_ma20_by_date = (self.ma20[sr_rank_df <= top]).loc[target_date]
        top_ma60_by_date = (self.ma60[sr_rank_df <= top]).loc[target_date]
        top_ma120_by_date = (self.ma120[sr_rank_df <= top]).loc[target_date]
        top_hpoint_by_date = (self.close.iloc[-prev_high_period:].cummax()[sr_rank_df <= top]).loc[target_date]
        top_rank = top_rank_by_date[~top_rank_by_date.isna()] 
        top_sr = top_sr_by_date[~top_sr_by_date.isna()] 
        top_ret = top_ret_by_date[~top_ret_by_date.isna()] * 100
        top_close = top_close_by_date[~top_close_by_date.isna()] 
        top_hpoint = top_hpoint_by_date[~top_hpoint_by_date.isna()] 
        top_ma5 = top_ma5_by_date[~top_ma5_by_date.isna()] 
        top_ma20 = top_ma20_by_date[~top_ma20_by_date.isna()] 
        top_ma60 = top_ma60_by_date[~top_ma60_by_date.isna()] 
        top_ma120 = top_ma120_by_date[~top_ma120_by_date.isna()] 

        # 특징 일자 기준 주도주 분석 : 데이터 통합 정리
        daily_sr_leading_df = pd.DataFrame(top_sr)
        daily_sr_leading_df.columns = ['SR']
        daily_sr_leading_df[f'SR_Rank'] = top_rank
        daily_sr_leading_df['Sector'] = [self.sector_dict[stk_nm] for stk_nm in daily_sr_leading_df.index]
        daily_sr_leading_df['1D_Ret(%)'] = top_ret
        daily_sr_leading_df['Price(C)'] = top_close 
        daily_sr_leading_df['Is_Over_MA_20'] = top_close > top_ma20 
        daily_sr_leading_df[f'Prev_High({lookback_type})'] = top_hpoint 
        daily_sr_leading_df['ARC'] = np.nan
        for ticker in daily_sr_leading_df.index :
            daily_sr_leading_df.loc[ticker, 'ARC'] = self.cal_ARC(ticker, self.prices, arc_param=arc_param)['ARC'].loc[target_date]
        daily_sr_leading_df[f'Stop_Loss'] = daily_sr_leading_df[f'Prev_High({lookback_type})'] - daily_sr_leading_df['ARC']
        daily_sr_leading_df[f'Stop_Loss(%)'] = ((daily_sr_leading_df['Stop_Loss'] / daily_sr_leading_df[f'Price(C)']) - 1) * 100
        daily_sr_leading_df['MA_5'] = top_ma5 
        daily_sr_leading_df['MA_20'] = top_ma20 
        daily_sr_leading_df['MA_60'] = top_ma60 
        daily_sr_leading_df['MA_120'] = top_ma120
        daily_sr_leading_df.sort_values(by=['SR_Rank'], inplace=True)

        # 특징 일자 기준 주도주 분석 : 개별 칼럼 맞춤 포맷팅 적용
        daily_sr_leading_df['SR'] = [f'{val:.2f}' for val in daily_sr_leading_df['SR']]
        daily_sr_leading_df['SR_Rank'] = [f'{val:.0f}' for val in daily_sr_leading_df['SR_Rank']]
        daily_sr_leading_df['1D_Ret(%)'] = [f'{val:+.2f}%' for val in daily_sr_leading_df['1D_Ret(%)']]
        daily_sr_leading_df['Price(C)'] = [f'{val:,.0f}' for val in daily_sr_leading_df['Price(C)']]
        daily_sr_leading_df[f'Prev_High({lookback_type})'] = [f'{val:,.0f}' for val in daily_sr_leading_df[f'Prev_High({lookback_type})']]
        daily_sr_leading_df['ARC'] = [f'{val:,.0f}' for val in daily_sr_leading_df['ARC']]
        daily_sr_leading_df['Stop_Loss'] = [f'{val:,.0f}' for val in daily_sr_leading_df['Stop_Loss']]
        daily_sr_leading_df['Stop_Loss(%)'] = [f'{val:+.2f}%' for val in daily_sr_leading_df['Stop_Loss(%)']]
        daily_sr_leading_df['MA_5'] = [f'{val:,.0f}' for val in daily_sr_leading_df['MA_5']]
        daily_sr_leading_df['MA_20'] = [f'{val:,.0f}' for val in daily_sr_leading_df['MA_20']]
        daily_sr_leading_df['MA_60'] = [f'{val:,.0f}' for val in daily_sr_leading_df['MA_60']]
        daily_sr_leading_df['MA_120'] = [f'{val:,.0f}' for val in daily_sr_leading_df['MA_120']]
        daily_sr_leading_df['Is_Over_MA_20'] = [str(val).upper() for val in daily_sr_leading_df['Is_Over_MA_20']]
        daily_sr_leading_df.reset_index(inplace=True)
        daily_sr_leading_df.rename(columns = {'Ticker' : 'Stock'}, inplace=True)

        return daily_sr_leading_df
    
    """ 단일 lookback 샤프비율 주도주 분석 데이터프레임 table 시각화 자료 생성 및 저장 함수 """
    def show_daily_sharpe_leading(self, daily_leading_df, target_date, sr_type = 'SR', lookback_type= '6M', show_mode = True) :

        # 샤프비율 종류 추출 
        if sr_type == 'SR' :
            sr_name = 'Rolling Sharpe Ratio'
        elif sr_type == 'EWSR' :
            sr_name = 'Exponential Weighted Rolling Sharpe Ratio'
        else :
            raise ValueError('Invalid Sharpe Ratio Type')

        # 텍스트 컬러 지정
        text_colors = [['#cccccc'] * len(daily_leading_df)] * len(daily_leading_df.columns)
        ret_colors = ['#ff4b4b' if '+' in x else '#4b96ff' if '-' in x else '#cccccc' for x in daily_leading_df['1D_Ret(%)']]
        stop_loss_pct_colors = ['#00ff88' if '-' in str(x) else '#d6a3fb' if '+' in str(x) else '#cccccc' for x in daily_leading_df['Stop_Loss(%)']]
        ret_col_idx = list(daily_leading_df.columns).index('1D_Ret(%)')
        sl_pct_col_idx = list(daily_leading_df.columns).index('Stop_Loss(%)')
        text_colors[ret_col_idx] = ret_colors
        text_colors[sl_pct_col_idx] = stop_loss_pct_colors

        # 피규어 객체 생성
        fig = go.Figure(
            
            # 테이블 객체 생성
            go.Table(
                
                # 칼럼 쎌 길이 조절
                columnwidth = [100, 40, 60, 90, 80, 80, 100, 80] + [70] * 6, 
                
                # 칼럼 속성 조절
                header = dict(
                    
                    # 칼럼명 지정
                    values = list(daily_leading_df.columns),
                    
                    # 칼럼 배경/글자색 지정 및 배치 조절
                    fill_color='#2d2d2d',
                    font=dict(color='white', size=13),
                    
                    align='center'
                    ),
                
                # 값 속성 조절
                cells = dict(
                    values = [daily_leading_df[col] for col in daily_leading_df.columns],
                    fill_color = ['#1e1e1e' if i % 2 == 0 else '#262626' for i in range(len(daily_leading_df))],
                    font = dict(color = text_colors, size = 13),
                    align = ['left', 'center', 'center', 'left'] + ['right'] * 2 + ['center'] + ['right'] * (len(daily_leading_df.columns)-7),
                    height = 37
                    )
            )
        )

        # 오늘 일자 추출 
        now_date = datetime.today().strftime('%Y-%m-%d')

        fig.update_layout(
            
            # 제목 지정 
            title=dict(
                text=f'<b>{self.target_market} @ {lookback_type} {sr_name} Leading Stock Analysis @ {now_date}</b>', 
                font=dict(size=20, color='white'),
                x = 0.5, y = 0.96, xanchor = 'center', yanchor = 'top'
                ),
            
            # 가로 세로 높이 조절
            width= 1800, 
            height = 900, 
            
            # 마진 제거 
            margin=dict(l=20, r=20, t=85, b=20),
            paper_bgcolor= 'black',
            plot_bgcolor='black'
            )

        # 이미지 저장
        
        # Root Save DIR 
        target_dir = f'./데이터/VisualizationResult/{target_date}'

        # 당일 저장 주도주 분석 폴더가 있는 경우 
        if os.path.exists(target_dir):
            pass

        # 당일 저장 주도주 분석 폴더가 없는 경우 : 생성
        else:  
            os.makedirs(target_dir, exist_ok=True)
        
        fig.write_image(f'{self.target_market}_{lookback_type}_{sr_type}_{now_date}.png', scale=2)

        if show_mode : 
            fig.show()


    """ Multi TimeFrame lookback 샤프비율 주도주 분석 데이터프레임 table 시각화 자료 생성 및 저장 함수 """
    def show_daily_sharpe_leading_integrated(self, df_list, lookback_type_list, target_date, sr_type = 'SR', show_mode = True) :
        # 샤프비율 종류 추출 
        if sr_type == 'SR' :
            sr_name = 'Rolling Sharpe Ratio'
        elif sr_type == 'EWSR' :
            sr_name = 'Exponential Weighted Rolling Sharpe Ratio'
        else :
            raise ValueError('Invalid Sharpe Ratio Type')

        # 2 x 2 서브플랏 생성
        fig = make_subplots(
            # 행/열 구조
            rows=2, cols=2,
            
            # 서브플랏 Type 지정
            specs=[[{"type": "table"}, {"type": "table"}], [{"type": "table"}, {"type": "table"}]],
            
            # 서브플랏 사이 공백 지정
            horizontal_spacing=0.02, vertical_spacing=0.02,   
            
            # 서브플랏 제목 지정
            subplot_titles=[f"<b>{self.target_market} @ {lb_type} {sr_type} Leading Stock Analysis @ {target_date}</b>" for lb_type in lookback_type_list]
        )

        # 서브플랏 별 테이블 시각화 
        for i, daily_leading_df in enumerate(df_list):
            
            # 서브플랏 위치 지정
            row, col = (i // 2) + 1 , (i % 2) + 1

            # 텍스트 컬러 지정
            text_colors = [['#cccccc'] * len(daily_leading_df)] * len(daily_leading_df.columns)
            ret_colors = ['#ff4b4b' if '+' in x else '#4b96ff' if '-' in x else '#cccccc' for x in daily_leading_df['1D_Ret(%)']]
            stop_loss_pct_colors = ['#00ff88' if '-' in str(x) else '#d6a3fb' if '+' in str(x) else '#cccccc' for x in daily_leading_df['Stop_Loss(%)']]
            ret_col_idx = list(daily_leading_df.columns).index('1D_Ret(%)')
            sl_pct_col_idx = list(daily_leading_df.columns).index('Stop_Loss(%)')
            text_colors[ret_col_idx] = ret_colors
            text_colors[sl_pct_col_idx] = stop_loss_pct_colors

            fig.add_trace(
                
                # 테이블 객체 생성
                go.Table(
                    
                    # 칼럼 쎌 길이 조절
                    columnwidth = [100, 40, 60, 90, 80, 80, 100, 80] + [70] * 6, 
                    
                    # 칼럼 속성 조절
                    header = dict(
                        
                        # 칼럼명 지정
                        values = list(daily_leading_df.columns),
                        
                        # 칼럼 배경/글자색 지정 및 배치 조절
                        fill_color='#2d2d2d',
                        font=dict(color='white', size=11),
                        
                        align='center'
                        ),
                    
                    # 값 속성 조절
                    cells = dict(
                        values = [daily_leading_df[col] for col in daily_leading_df.columns],
                        fill_color = ['#1e1e1e' if i % 2 == 0 else '#262626' for i in range(len(daily_leading_df))],
                        font = dict(color = text_colors, size = 11),
                        align = ['left', 'center', 'center', 'left'] + ['right'] * 2 + ['center'] + ['right'] * (len(daily_leading_df.columns)-7),
                        height = 37
                        )
                ),

                # 서브플랏 위치 지정
                row = row, col = col
            )


        # 서브플롯 타이틀 색 지정 및 배치 조정
        for i in fig['layout']['annotations']:
            i['font'] = dict(size=18, color='white')
            i['y'] = i['y'] + 0.02

        # 레이아웃 조정
        fig.update_layout(

            width= 3000, 
            height = 2000, 
            margin=dict(l=20, r=20, t=100, b=20),
            paper_bgcolor= 'black',
            plot_bgcolor='black'
        ) 

        # 이미지 저장
        
        # Root Save DIR 
        target_dir = f'./데이터/VisualizationResult/{target_date}'

        # 당일 저장 주도주 분석 폴더가 있는 경우 
        if os.path.exists(target_dir):
            pass

        # 당일 저장 주도주 분석 폴더가 없는 경우 : 생성
        else:  
            os.makedirs(target_dir, exist_ok=True)
        
        fig.write_image(f'{target_dir}/{self.target_market}_Integrated_{sr_type}_{target_date}.png', scale=2)

        if show_mode : 
            fig.show()


try : 

    """"@@@@@@@@@@@@@@@@@@@@@@@@@@코스피200@@@@@@@@@@@@@@@@@@@@@@@@"""

    # 주가지수 상장종목 데이터 
    data = pd.read_csv('./데이터/StockInfo/KOSPI200_종목티커정보.csv', encoding='cp949').copy()
    ticker_data = data[['종목코드', '종목명']].values
    tic_nm_mapper = {f'{ticker}.KS' : stk_nm for ticker, stk_nm in ticker_data}

    # 상장종목 OHLCV 가격 데이터 다운로드 
    prices_tmp = yf.download(list(tic_nm_mapper.keys()), start= '2020-01-01', auto_adjust=True)

    # OHLC 분리 후 종목명 매칭
    close_df = prices_tmp['Close'].rename(columns=tic_nm_mapper)
    open_df  = prices_tmp['Open'].rename(columns=tic_nm_mapper)
    high_df  = prices_tmp['High'].rename(columns=tic_nm_mapper)
    low_df   = prices_tmp['Low'].rename(columns=tic_nm_mapper)

    # 재병합 및 형식 통일
    ks_prices = pd.concat( [open_df, high_df, low_df, close_df], axis=1, keys=['Open', 'High', 'Low', 'Close'])
    ks_prices.columns.names = ['Price', 'Ticker']

    # 업종 정보 추출 
    with open('./데이터/StockInfo/korea_wics_sector_data.pkl', 'rb') as f:
        wics = pickle.load(f)

    ks_sector_dict = {stk_nm : sector for stk_nm, sector in wics['KOSPI200'][['stock_name', 'wics_sub']].values}

    ks200 = SharpeLeadingAnalysis(prices = ks_prices, sector_dict = ks_sector_dict, target_market='KOSPI200')

    # Multi TimeFrame 샤프비율 값 및 순위 데이터프레임 생성
    rolling_6m_sr , sr_6m_rank = ks200.rolling_sr(20 * 6)    
    rolling_3m_sr , sr_3m_rank = ks200.rolling_sr(20 * 3) 
    rolling_1m_sr , sr_1m_rank = ks200.rolling_sr(20 * 1) 
    rolling_1w_sr , sr_1w_rank = ks200.rolling_sr(5) 
    rolling_6m_ewsr , ewsr_6m_rank = ks200.rolling_ewsr(20 * 6)    
    rolling_3m_ewsr , ewsr_3m_rank = ks200.rolling_ewsr(20 * 3) 
    rolling_1m_ewsr , ewsr_1m_rank = ks200.rolling_ewsr(20 * 1) 
    rolling_1w_ewsr , ewsr_1w_rank = ks200.rolling_ewsr(5) 


    # 티겟 일자 추출 : Target Date를 수동으로 지정 또는 최근 영업일으로 자동 지정 
    target_date = rolling_6m_sr.iloc[-1].name.strftime('%Y-%m-%d')
    # target_date = '2026-03-31'

    print(f'KOSPI200 상장종목의 {target_date} 일자 주도주 분석 기초 데이터프레임 생성 시작')
    # Multi TimeFrame 샤프비율 주도주 분석 기초 데이터프레임 생성
    daily_sr_6m_leading_df = ks200.make_sr_leading_df(rolling_6m_sr, sr_6m_rank,  target_date = target_date , sr_type = 'SR', top= 20, arc_param = 1, lookback_type = '6M')
    daily_sr_3m_leading_df = ks200.make_sr_leading_df(rolling_3m_sr, sr_3m_rank,  target_date = target_date , sr_type = 'SR', top= 20, arc_param = 1, lookback_type = '3M')
    daily_sr_1m_leading_df = ks200.make_sr_leading_df(rolling_1m_sr, sr_1m_rank,  target_date = target_date , sr_type = 'SR', top= 20, arc_param = 1, lookback_type = '1M')
    daily_sr_1w_leading_df = ks200.make_sr_leading_df(rolling_1w_sr, sr_1w_rank,  target_date = target_date , sr_type = 'SR', top= 20, arc_param = 1, lookback_type = '1W')
    daily_ewsr_6m_leading_df = ks200.make_sr_leading_df(rolling_6m_ewsr, ewsr_6m_rank,  target_date = target_date , sr_type = 'EWSR', top= 20, arc_param = 1, lookback_type = '6M')
    daily_ewsr_3m_leading_df = ks200.make_sr_leading_df(rolling_3m_ewsr, ewsr_3m_rank,  target_date = target_date , sr_type = 'EWSR', top= 20, arc_param = 1, lookback_type = '3M')
    daily_ewsr_1m_leading_df = ks200.make_sr_leading_df(rolling_1m_ewsr, ewsr_1m_rank,  target_date = target_date , sr_type = 'EWSR', top= 20, arc_param = 1, lookback_type = '1M')
    daily_ewsr_1w_leading_df = ks200.make_sr_leading_df(rolling_1w_ewsr, ewsr_1w_rank,  target_date = target_date , sr_type = 'EWSR', top= 20, arc_param = 1, lookback_type = '1W')

    print('KOSPI200 상장종목의 최근일자 주도주 분석 데이터프레임 통합 시각화 이미지 생성 중')

    # ks200.show_daily_sharpe_leading(daily_sr_6m_leading_df, target_date = target_date, sr_type = 'SR', lookback_type= '6M', show_mode = False)
    # ks200.show_daily_sharpe_leading(daily_sr_3m_leading_df, target_date = target_date, sr_type = 'SR', lookback_type= '3M', show_mode = False)
    # ks200.show_daily_sharpe_leading(daily_sr_1m_leading_df, target_date = target_date, sr_type = 'SR', lookback_type= '1M', show_mode = False)
    # ks200.show_daily_sharpe_leading(daily_sr_1w_leading_df, target_date = target_date, sr_type = 'SR', lookback_type= '1W', show_mode = False)
    # ks200.show_daily_sharpe_leading(daily_ewsr_6m_leading_df, target_date = target_date, sr_type = 'EWSR', lookback_type= '6M', show_mode = False)
    # ks200.show_daily_sharpe_leading(daily_ewsr_3m_leading_df, target_date = target_date, sr_type = 'EWSR', lookback_type= '3M', show_mode = False)
    # ks200.show_daily_sharpe_leading(daily_ewsr_1m_leading_df, target_date = target_date, sr_type = 'EWSR', lookback_type= '1M', show_mode = False)
    # ks200.show_daily_sharpe_leading(daily_ewsr_1w_leading_df, target_date = target_date, sr_type = 'EWSR', lookback_type= '1W', show_mode = False)


    # 일반 샤프비율 버전 
    sr_df_list = [daily_sr_6m_leading_df, daily_sr_3m_leading_df, daily_sr_1m_leading_df, daily_sr_1w_leading_df]
    sr_lookback_type_list = ['6M', '3M', '1M', '1W']
    ks200.show_daily_sharpe_leading_integrated(sr_df_list, sr_lookback_type_list, target_date, sr_type = 'SR', show_mode = False)

    # EW 샤프비율 버전 
    ewsr_df_list = [daily_ewsr_6m_leading_df, daily_ewsr_3m_leading_df, daily_ewsr_1m_leading_df, daily_ewsr_1w_leading_df]
    ewsr_lookback_type_list = ['6M', '3M', '1M', '1W']
    ks200.show_daily_sharpe_leading_integrated(ewsr_df_list, ewsr_lookback_type_list, target_date, sr_type = 'EWSR', show_mode = False)

    """"@@@@@@@@@@@@@@@@@@@@@@@@@@코스닥150@@@@@@@@@@@@@@@@@@@@@@@@"""

    # 주가지수 상장종목 데이터 
    data = pd.read_csv('./데이터/StockInfo/KOSDAQ150_종목티커정보.csv', encoding='cp949').copy()
    ticker_data = data[['종목코드', '종목명']].values
    tic_nm_mapper = {f'{ticker}.KS' : stk_nm for ticker, stk_nm in ticker_data}

    # 상장종목 OHLCV 가격 데이터 다운로드 
    prices_tmp = yf.download(list(tic_nm_mapper.keys()), start= '2020-01-01', auto_adjust=True)

    # OHLC 분리 후 종목명 매칭
    close_df = prices_tmp['Close'].rename(columns=tic_nm_mapper)
    open_df  = prices_tmp['Open'].rename(columns=tic_nm_mapper)
    high_df  = prices_tmp['High'].rename(columns=tic_nm_mapper)
    low_df   = prices_tmp['Low'].rename(columns=tic_nm_mapper)

    # 재병합 및 형식 통일
    kd_prices = pd.concat( [open_df, high_df, low_df, close_df], axis=1, keys=['Open', 'High', 'Low', 'Close'])
    kd_prices.columns.names = ['Price', 'Ticker']

    # 업종 정보 추출 
    with open('./데이터/StockInfo/korea_wics_sector_data.pkl', 'rb') as f:
        wics = pickle.load(f)

    kd_sector_dict = {stk_nm : sector for stk_nm, sector in wics['KOSDAQ150'][['stock_name', 'wics_sub']].values}


    kd150 = SharpeLeadingAnalysis(prices = kd_prices, sector_dict = kd_sector_dict, target_market='KOSDAQ150')

    # Multi TimeFrame 샤프비율 값 및 순위 데이터프레임 생성
    rolling_6m_sr , sr_6m_rank = kd150.rolling_sr(20 * 6)    
    rolling_3m_sr , sr_3m_rank = kd150.rolling_sr(20 * 3) 
    rolling_1m_sr , sr_1m_rank = kd150.rolling_sr(20 * 1) 
    rolling_1w_sr , sr_1w_rank = kd150.rolling_sr(5) 
    rolling_6m_ewsr , ewsr_6m_rank = kd150.rolling_ewsr(20 * 6)    
    rolling_3m_ewsr , ewsr_3m_rank = kd150.rolling_ewsr(20 * 3) 
    rolling_1m_ewsr , ewsr_1m_rank = kd150.rolling_ewsr(20 * 1) 
    rolling_1w_ewsr , ewsr_1w_rank = kd150.rolling_ewsr(5) 

    # 티겟 일자 추출 : Target Date를 수동으로 지정 또는 최근 영업일으로 자동 지정 
    target_date = rolling_6m_sr.iloc[-1].name.strftime('%Y-%m-%d')
    # target_date = '2026-03-31'

    print(f'KOSDAQ150 상장종목의 {target_date} 일자 주도주 분석 기초 데이터프레임 생성 시작')
    # Multi TimeFrame 샤프비율 주도주 분석 기초 데이터프레임 생성
    daily_sr_6m_leading_df = kd150.make_sr_leading_df(rolling_6m_sr, sr_6m_rank,  target_date = target_date , sr_type = 'SR', top= 20, arc_param = 1, lookback_type = '6M')
    daily_sr_3m_leading_df = kd150.make_sr_leading_df(rolling_3m_sr, sr_3m_rank,  target_date = target_date , sr_type = 'SR', top= 20, arc_param = 1, lookback_type = '3M')
    daily_sr_1m_leading_df = kd150.make_sr_leading_df(rolling_1m_sr, sr_1m_rank,  target_date = target_date , sr_type = 'SR', top= 20, arc_param = 1, lookback_type = '1M')
    daily_sr_1w_leading_df = kd150.make_sr_leading_df(rolling_1w_sr, sr_1w_rank,  target_date = target_date , sr_type = 'SR', top= 20, arc_param = 1, lookback_type = '1W')
    daily_ewsr_6m_leading_df = kd150.make_sr_leading_df(rolling_6m_ewsr, ewsr_6m_rank,  target_date = target_date , sr_type = 'EWSR', top= 20, arc_param = 1, lookback_type = '6M')
    daily_ewsr_3m_leading_df = kd150.make_sr_leading_df(rolling_3m_ewsr, ewsr_3m_rank,  target_date = target_date , sr_type = 'EWSR', top= 20, arc_param = 1, lookback_type = '3M')
    daily_ewsr_1m_leading_df = kd150.make_sr_leading_df(rolling_1m_ewsr, ewsr_1m_rank,  target_date = target_date , sr_type = 'EWSR', top= 20, arc_param = 1, lookback_type = '1M')
    daily_ewsr_1w_leading_df = kd150.make_sr_leading_df(rolling_1w_ewsr, ewsr_1w_rank,  target_date = target_date , sr_type = 'EWSR', top= 20, arc_param = 1, lookback_type = '1W')


    # print('KOSDAQ150 상장종목의 최근일자 주도주 분석 데이터프레임 통합 시각화 이미지 생성 중')
    # kd150.show_daily_sharpe_leading(daily_sr_6m_leading_df, target_date = target_date, sr_type = 'SR', lookback_type= '6M', show_mode = False)
    # kd150.show_daily_sharpe_leading(daily_sr_3m_leading_df, target_date = target_date, sr_type = 'SR', lookback_type= '3M', show_mode = False)
    # kd150.show_daily_sharpe_leading(daily_sr_1m_leading_df, target_date = target_date, sr_type = 'SR', lookback_type= '1M', show_mode = False)
    # kd150.show_daily_sharpe_leading(daily_sr_1w_leading_df, target_date = target_date, sr_type = 'SR', lookback_type= '1W', show_mode = False)
    # kd150.show_daily_sharpe_leading(daily_ewsr_6m_leading_df, target_date = target_date, sr_type = 'EWSR', lookback_type= '6M', show_mode = False)
    # kd150.show_daily_sharpe_leading(daily_ewsr_3m_leading_df, target_date = target_date, sr_type = 'EWSR', lookback_type= '3M', show_mode = False)
    # kd150.show_daily_sharpe_leading(daily_ewsr_1m_leading_df, target_date = target_date, sr_type = 'EWSR', lookback_type= '1M', show_mode = False)
    # kd150.show_daily_sharpe_leading(daily_ewsr_1w_leading_df, target_date = target_date, sr_type = 'EWSR', lookback_type= '1W', show_mode = False)


    # 일반 샤프비율 버전 
    sr_df_list = [daily_sr_6m_leading_df, daily_sr_3m_leading_df, daily_sr_1m_leading_df, daily_sr_1w_leading_df]
    sr_lookback_type_list = ['6M', '3M', '1M', '1W']
    kd150.show_daily_sharpe_leading_integrated(sr_df_list, sr_lookback_type_list, target_date, sr_type = 'SR', show_mode = False)

    # EW 샤프비율 버전 
    ewsr_df_list = [daily_ewsr_6m_leading_df, daily_ewsr_3m_leading_df, daily_ewsr_1m_leading_df, daily_ewsr_1w_leading_df]
    ewsr_lookback_type_list = ['6M', '3M', '1M', '1W']
    kd150.show_daily_sharpe_leading_integrated(ewsr_df_list, ewsr_lookback_type_list, target_date, sr_type = 'EWSR', show_mode = False)

except Exception as e :
    print(e)
    time.sleep(15)