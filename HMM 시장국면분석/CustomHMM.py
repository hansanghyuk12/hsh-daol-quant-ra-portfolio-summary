import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
sns.set()
from tqdm import tqdm
from sklearn.cluster import KMeans
import scipy.stats as stats
import matplotlib.gridspec as gridspec

# Multivariate Gaussian HMM and analysis
class CustomHMM :
    def __init__(self, y_train, k, max_iter=100) :
        
        self.y_train = y_train 
        self.k = k 
        self.max_iter = max_iter
        self.columns = y_train.columns

        # 파라미터 초기화
        self.Pi, self.A, self.mus, self.covs, self.sample_num_by_cluster = self.initializer(self.y_train, self.k)

        # HMM(Lambda^*) 
        self.Pi , self.A, self.mus, self.covs, self.ll_history = self.baum_welch(self.y_train, self.k, show_history=False, max_iter=self.max_iter)


    # 파라미터 초기화 함수
    def initializer(self, y_train, k) :

        """Baum - Welch 초기화 함수"""
        
        # 관측 변수 개수 
        m = y_train.shape[1]

        # 1e-6
        epsilon = 1e-6 

        # 상태 별 표본 개수 저장 딕셔너리
        sample_num_by_cluster = {}
        
        # # 변수 칼럼 정의
        # cols = list(y_train.columns)

        # [K-means를 활용한 Cluster/State 별 평균과 공분산 초기화]

        # 1. K-means Cluster 결과 반영 
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(y_train.values)
        kmeans_df = y_train.copy()
        kmeans_df['cluster'] = kmeans.labels_

        # 2. Cluster 별 평균과 공분산 구하기

        # 2-1. 초기화 평균, 공분산 저장 array
        initial_mus = np.zeros((k, m))
        initial_covs = np.zeros((k, m, m))

        # 2-2. Cluster 별 데이터 필터링 
        for j in range(k) :
            
            # 클러스터 별 데이터 필터링 
            cluster_data = kmeans_df[kmeans_df['cluster'] == j]
            cluster_data.drop('cluster', axis=1, inplace=True)

            # 클러스터 별 샘플 수 저장
            sample_num_by_cluster[j] = len(cluster_data)
            
            # 클러스터 별 평균
            initial_mus[j] = cluster_data.mean(axis=0)

            # 클러스터 별 공분산 
            initial_covs[j] = cluster_data.cov() + epsilon * np.eye(m)

        # 상태 확률 행렬 Pi 초기화 
        initial_Pi = np.ones([k,1]) / k 

        # 상태 전이 확률 행렬 A 초기화 (rowsum = 1)
        initial_A = np.full((k, k), 1.0 / k) 

        return initial_Pi, initial_A, initial_mus, initial_covs, sample_num_by_cluster

    # 가독성을 위한 vec화 함수
    def vec(self, array, k) :
        vec_array = array.reshape([k, 1])
        return vec_array
    
    def get_bt_time_series(self, y, mus = None, covs = None) :
        
        # 인자가 전달되지 않으면 클래스가 학습한 파라미터 사용
        mus = mus if mus is not None else self.mus
        covs = covs if covs is not None else self.covs

        # 시계열 길이 및 변수 개수 추출 
        T, m = y.shape

        # 상태 개수 추출
        k = mus.shape[0]

        # ln b_t 시계열 저장 리스트 (viterbi algorithm 에서 사용)
        ln_b = np.zeros((T,k))

        # 관측값 array
        Y = y.values

        # 상태 별 관측 확률 추출
        for j in range(k) :
            
            # 상태 별 관측 확률 저장 array
            save_obs_prob = np.zeros(T)

            # 상태 별 평균과 공분산
            cov_j = covs[j]
            mu_j = mus[j]

            # ln det(Cov_j) 추출
            sign, log_det = np.linalg.slogdet(cov_j)

            # inv(Cov_j) 추출
            inv_cov_j = np.linalg.solve(cov_j, np.eye(m))

            # # 상태 별 관측 확률밀도 시계열 계산 및 저장
            diff = Y - mu_j
            quad_form = np.sum((diff @ inv_cov_j) * diff, axis=1)
            ln_b[:, j] = -0.5 * (m * np.log(2 * np.pi) + log_det + quad_form)

        # 상태 별 관측 확률밀도 시계열 원본 변환 (forward / backward algorithm 에서 사용)
        b = np.exp(ln_b)

        return ln_b , b

    # Scaled Forward Algorithm
    def forward(self, y, Pi = None, A = None, return_b = False) :
   
        """Scaling Forward Algorithm"""

        # 인자가 전달되지 않으면 클래스가 학습한 파라미터 사용
        Pi = Pi if Pi is not None else self.Pi
        A = A if A is not None else self.A

        # 시점 별 상태 별 관측 확률 추출
        ln_b, b = self.get_bt_time_series(y = y)

        # 시계열 길이 및 상태 개수 추출 
        T, k = b.shape

        # 전방 확률 벡터 (alpha_t) 시계열 array : unscale, scale 버전
        alpha_unscaled = np.zeros((T,k))
        alpha_scaled = np.zeros((T,k))

        # t = 1 : 초기 전방 확률 계산 및 저장
        alpha_1_unscaled = Pi.flatten() * b[0]
        alpha_unscaled[0] = alpha_1_unscaled
        alpha_scaled[0] = alpha_1_unscaled / alpha_1_unscaled.sum()

        # t = 2 ~ T : 전방 확률 계산 및 저장
        for t in range(1, T) :
            alpha_t_unscaled = ((A.T @ self.vec(alpha_scaled[t-1], k)) * self.vec(b[t], k)).flatten()
            alpha_unscaled[t] = alpha_t_unscaled
            alpha_scaled[t] = alpha_t_unscaled / alpha_t_unscaled.sum()

        # Likelyhood
        forward_likelyhood = np.log(alpha_unscaled.sum(axis=1)).sum()
        if return_b :
            return alpha_unscaled, alpha_scaled, forward_likelyhood, b
        else :     
            return alpha_unscaled, alpha_scaled, forward_likelyhood
    
    # Scaled Backward Algorithm
    def backward(self, y, Pi = None, A = None) :
        
        """Scaling Backward Algorithm"""
        
        # 인자가 전달되지 않으면 클래스가 학습한 파라미터 사용
        Pi = Pi if Pi is not None else self.Pi
        A = A if A is not None else self.A

        # 시점 별 상태 별 관측 확률 추출
        ln_b, b = self.get_bt_time_series(y = y)

        # 시계열 길이 및 상태 개수 추출 
        T, k = b.shape

        # 후방 확률 벡터 (beta_t) 시계열 array : unscale, scale 버전
        beta_unscaled = np.zeros((T,k))
        beta_scaled = np.zeros((T,k))

        # t = T : 초기 후방 확률 계산 및 저장
        beta_unscaled[T-1] = np.repeat(1, k)
        beta_scaled[T-1] = np.repeat(1, k)

        # t = T-1 ~ 1 : 후방 확률 계산 및 저장
        for t in range(T-2, -1, -1):
            beta_t_unscaled = (A @ (self.vec(b[t+1], k) * self.vec(beta_scaled[t+1], k))).flatten()
            beta_unscaled[t] = beta_t_unscaled
            beta_scaled[t] =  beta_t_unscaled / beta_t_unscaled.sum()

        # Likelyhood
        backward_likelyhood = np.log(Pi.T @ (self.vec(b[0], k) * self.vec(beta_scaled[0], k))).reshape(-1)[0] + np.log(beta_unscaled[:-1].sum(axis=1)).sum()

        return beta_unscaled, beta_scaled, backward_likelyhood
    
    # Log Viterbi Algorithm
    
    def viterbi(self, y, Pi=None, A=None, mus=None, covs=None):
        
        # 인자가 전달되지 않으면 클래스가 학습한 파라미터 사용
        Pi = Pi if Pi is not None else self.Pi
        A = A if A is not None else self.A
        mus = mus if mus is not None else self.mus
        covs = covs if covs is not None else self.covs

        ln_b, _ = self.get_bt_time_series(y, mus, covs)
        # 시계열 길이 및 상태 개수 추출 
        T, k = ln_b.shape

        # Viterbi 확률 벡터 (v_t) 시계열 array
        viterbi_prob = np.zeros((T, k))

        # 경로 추적을 위한 backpointer (tau_t) array
        tau = np.zeros((T, k), dtype=int)

        # 최적 경로 저장 array
        best_path = np.zeros(T, dtype=int)

        # t = 1 : 초기 Viterbi 확률 계산 및 저장
        viterbi_prob[0] = np.log(Pi).flatten() + ln_b[0]

        # t = 2 ~ T : Viterbi 확률 계산 및 저장
        for t in range(1,T) :
            log_A_T = np.log(A.T + 1e-12)
            mt = (log_A_T + np.ones([k,1]) @ self.vec(viterbi_prob[t-1], k).T)
            viterbi_prob[t] = (self.vec(mt.max(axis=1), k) + self.vec(ln_b[t], k)).flatten()
            tau[t] = mt.argmax(axis=1)

        # Backtracking at t = T
        best_path[T-1] = np.argmax(viterbi_prob[T-1])

        # Backtracking at t = T-1 ~ 1
        for t in range(T-2, -1, -1):
            best_path[t] = tau[t+1, best_path[t+1]]

        return viterbi_prob, best_path
    
   # Scaled, Laplace Smoothing, None Zero Transition Baum-Welch Algorithm
    def baum_welch(self, y_train, k, max_iter = 100, tol = 1e-5 , show_history = True) :

        # 관측 변수 array
        Y = y_train.values

        # 로그 우도 기록용 list             
        ll_history = []        

        # 이전 시점 로그 우도 초기화
        old_ll = -np.inf     

        # 1e-6
        epsilon = 1e-6

        # 시계열 길이 및 변수 개수 추출
        T, m = y_train.shape

        # E-Step 초기화 : init_A, init_Pi, [init_mu_j, init_cov_j] from k means -> init_b
        now_Pi, now_A, now_mus, now_covs, _ = self.initializer(y_train, k)

        for i in tqdm(range(max_iter), desc= f'EM Iteration [ max iter {max_iter} ]') :
            
            # E-Step : 전방 / 후방 확률 계산 
            alpha_unscaled, alpha_scaled, f_ll , now_b = self.forward(y = y_train, Pi = now_Pi, A = now_A, return_b = True)
            _ , beta_scaled, b_ll = self.backward(y = y_train, Pi = now_Pi, A = now_A)

            # log-likelyhood 계산 및 저장
            current_ll = np.log(alpha_unscaled.sum(axis=1)).sum()
            ll_history.append(current_ll)

            # 수렴 여부 판단 |L_new - L_old| < tolerance
            if abs(current_ll - old_ll) < tol:
                print(f'Iteration {i}: Convergence reached. (Log-Likelyhood : {np.round(current_ll, 2)})')
                break

            # old log likelyhood 
            old_ll = current_ll
            
            # [ E-Step : gamma_t, xi_t 생성 ]

            # 1. gamma_t (k x 1) , xi_t (k x k) 저장 array
            Gamma = np.zeros([T, k])
            Xi = np.zeros([T-1, k, k]) 

            # 2. 시점 별 gamma_t 계산 및 저장
            for t in range(T) :
                com_t = self.vec(alpha_scaled[t], k) * self.vec(beta_scaled[t], k) 
                gamma_t = com_t / (np.ones([1,k]) @ com_t)
                Gamma[t] = gamma_t.flatten()

            # 3. 시점 별 xi_t 계산 및 저장
            for t in range(T-1) :
                pt = np.diag(beta_scaled[t+1] * now_b[t+1])
                at = np.diag(alpha_scaled[t]) @ now_A @ pt
                Xi[t] = at / (np.ones([1,k]) @ at @ np.ones([k,1]))

            # M-step : update Pi
            now_Pi = self.vec(Gamma[0], k)

            # M-step : update A
            Xi_sum = Xi.sum(axis=0) + (epsilon * np.ones([k,k]))
            Gamma_sum_diag = np.diag(Gamma[:-1].sum(axis=0) + (k * epsilon))
            now_A = np.linalg.solve(Gamma_sum_diag, Xi_sum)

            # M-step : update Mu
            diag_gamma_sum = np.diag((np.ones([1,T]) @ Gamma).flatten() + (k * epsilon)) 
            inv_diag_gamma_sum = np.linalg.solve(diag_gamma_sum, np.eye(k))
            now_mus = inv_diag_gamma_sum @ Gamma.T @ Y

            # M-step : update Cov
            for j in range(k) :
                diff = Y - now_mus[j]
                gamma_col_j = Gamma[:,j]
                now_covs[j] = (diff.T * gamma_col_j) @ diff / gamma_col_j.sum() + epsilon * np.eye(m)
        
        # Log-Likelyhood History 시각화 여부
        if show_history : 
            plt.figure(figsize=(7,5))
            plt.plot(ll_history, label = 'Log-Likelyhood')
            plt.title('Log-Likelihood History per Iteration')
            plt.xlabel('Iteration')
            plt.legend()
            plt.show()            
        
        fitted_Pi , fitted_A, fitted_mus, fitted_covs = now_Pi , now_A, now_mus, now_covs
        
        return fitted_Pi , fitted_A, fitted_mus, fitted_covs, ll_history 
    
    # Log-Likely Fitting History 시각화 함수
    def show_ll_history(self, ll_history=None) :
        
        # 인자가 전달되지 않으면 클래스가 학습한 ll_history 사용
        ll_history = ll_history if ll_history is not None else self.ll_history
        plt.figure(figsize=(7,5))
        plt.plot(ll_history, label = 'Log-Likelyhood')
        plt.title('Log-Likelihood History per Iteration')
        plt.xlabel('Iteration')
        plt.legend()
        plt.show()  

    def filtering(self, y_test, Pi=None, A=None, mus=None, covs=None):
        
        # 인자가 전달되지 않으면 클래스가 학습한 파라미터 사용
        Pi = Pi if Pi is not None else self.Pi
        A = A if A is not None else self.A
        mus = mus if mus is not None else self.mus
        covs = covs if covs is not None else self.covs

        # 적합 파라미터를 활용한 관측 확률 밀도 계산
        _, b_test = self.get_bt_time_series(y_test, mus,covs)
        
        # 적합 파라미터를 활용한 filtering 결과 및 시점 별 상태 확률 최댓값 추출
        _ , f_result, _ = self.forward(y_test)
        f_argmax = np.argmax(f_result, axis=1)

        return f_result, f_argmax 


    def smoothing(self, y_test, Pi=None, A=None, mus=None, covs=None):
        
        # 인자가 전달되지 않으면 클래스가 학습한 파라미터 사용
        Pi = Pi if Pi is not None else self.Pi
        A = A if A is not None else self.A
        mus = mus if mus is not None else self.mus
        covs = covs if covs is not None else self.covs

        # 적합 파라미터를 활용한 관측 확률 밀도 계산
        _, b_test = self.get_bt_time_series(y_test, mus, covs)
        
        # 적합 파라미터를 활용한 시점 별 전방 / 후방 확률 추출
        _ , f_scaled, _ = self.forward(y_test)
        _ , b_scaled, _ = self.backward(y_test)
        
        # Smoothing 결과 및 시점 별 상태 확률 최댓값 추출
        unnorm_sm = f_scaled * b_scaled
        sm_result = unnorm_sm / unnorm_sm.sum(axis=1, keepdims=True)
        sm_argmax = np.argmax(sm_result, axis=1)
        return sm_result, sm_argmax 

    
    def predict_state(self, y_test, h, A=None):
        
        # 인자가 전달되지 않으면 클래스가 학습한 파라미터 사용
        A = A if A is not None else self.A

        # 필터링 결과 추출
        f_result, _ = self.filtering(y_test)

        # 상태 전이 확률 행렬 A h번 거듭제곱
        A_h = np.linalg.matrix_power(A, h)
        
        # 미래 상태 확률 계산
        pred_result = f_result @ A_h
        pred_argmax = np.argmax(pred_result, axis=1)
        
        return pred_result, pred_argmax
    

    
    def show_gaussian(self) :
        
        # 변수 개수 지정
        variable_num = len(self.columns) 

        # 2열 기준 행 수 지정
        subplot_row = int(np.ceil(variable_num/2)) + 1 if variable_num / 2 == np.ceil(variable_num/2) else int(np.ceil(variable_num/2)) 

        # 변수 별 상태 별 정규분포 시각화 
        plt.figure(figsize=(20,10))

        # 변수 마다 상태 별 정규분포 플랏 시각화
        for i, col in enumerate(self.columns) :
            
            # 변수 별 Subplot 지정
            plt.subplot(subplot_row, 2, i+1)
            
            # 변수 별 상태 별 평균 / 표준편차 추출
            mus_by_state = self.mus[:, i]    
            std_by_state = np.sqrt(self.covs[:, i, i])
            
            # 정규분포 표시 범위 추출 (x 축)
            x_min = mus_by_state.min() - 3 * std_by_state.max()
            x_max = mus_by_state.max() + 3 * std_by_state.max()
            x = np.linspace(x_min, x_max, 200)

            # 상태 별 정규분포 시각화 
            for s in range(self.k) :
                
                # 상태 별 평균 / 표준편차 추출
                mu = mus_by_state[s]
                std = std_by_state[s]

                # 상태 별 정규분포 확률밀도 추출
                y = stats.norm.pdf(x, mu, std)

                # 상태 별 색깔 지정
                color = plt.cm.get_cmap('Set1')(s)

                # 정규 분포 시각화 
                plt.plot(x, y, label = f'State {s} (mu={mu:.2f}, var={std**2:.2f})', color=color, lw=2)
                plt.fill_between(x, 0, y, color=color, alpha=0.1)
                
            plt.title(f'Gaussian of {col} by State', fontsize=14)
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()

        plt.tight_layout()
        plt.show()

    def filtering_path_prob(self, y, close) :
        
        # Test Data Filtering
        test_f, test_fpath = self.filtering(y)

        # Close data와 filtering_path 매칭하기
        close['fpath'] = test_fpath

        # 상태 개수 지정
        state_num = len(close['fpath'].unique())

        # 존재하는 상태 추출
        state_existed = sorted(close['fpath'].unique())

        # 상태 별 색깔 추출
        pal = sns.color_palette('Set1', state_num)

        # 2열 기준 행 수 지정
        subplot_row = int(np.ceil(state_num/2)) + 1 if state_num / 2 == np.ceil(state_num/2) else int(np.ceil(state_num/2)) 
        
        # 피규어 생성
        plt.figure(figsize=(10,8))

        # GridSpec 설정
        gs = gridspec.GridSpec(2,1, height_ratios=[3,1])

        # Filtering 최적 경로 종가 매칭 시각화하기
        ax0 = plt.subplot(gs[0])
        ax0.plot(close['price'],c = 'grey', alpha = 0.3, label = 'Price')
        for i, state in enumerate(state_existed) :
            idx = close['fpath'] == state
            index_by_state = close.index[idx]
            price_by_state = close.loc[idx, 'price']
            ax0.scatter(index_by_state, price_by_state, s = 10, label = f'State {state}', c = pal[i])

        # y축 범위 조절
        y_min = close['price'].min()
        y_max = close['price'].max()
        margin = (y_max - y_min) * 0.05
        ax0.set_ylim(y_min - margin, y_max + margin)
        ax0.set_title('Filtering State Path & Prob')
        ax0.legend()

        # 상태 별 filtering 확률 시계열
        ax1 = plt.subplot(gs[1])
        for i in range(state_num) :
            ax1.plot(test_f[:, i], label = f'S_{i} prob', c = pal[i])
        ax1.legend()

        plt.tight_layout()
        plt.show()