# -*- coding: utf-8 -*-
"""
Copper Multi-Horizon Prediction: Linear Regression Baseline + LSTM
====================================================================
Step 1: Load features (parquet fast-path OR CSV+AlphaLib full build)
Step 2: Return distribution analysis (histogram, QQ, JB, ADF, ACF)
Step 3: Feature-return correlations (|corr| > 0.10 filter)
Step 4: Feature selection summary by category
Step 4.5: Alpha signal evaluation (IC, IR, standalone alpha strategies)
Step 5: Linear Regression baseline (per horizon: log_ret20, log_ret30, log_ret60)
Step 6: LSTM model training (per horizon, 2-layer 64 hidden, z-score target)
Step 7: Model evaluation -- LR vs LSTM comparison per horizon
Step 8: Trading strategies (per-horizon LSTM, combined, master ensemble)
Step 9: Results & plots -> claude/model_results.png

Train: 2005-2018 | Val: 2019-2020 | Test: 2021-2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tools.tools import add_constant
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE  = "C:/Users/sunAr/Documents/sunArise/quant/commodity"
SPLIT_VAL  = "2019-01-01"
SPLIT_TEST = "2021-01-01"
TARGET_VOL = 0.20
COPPER     = "HG=F"
SEQ_LEN    = 20
HIDDEN     = 64
N_LAYERS   = 2
DROPOUT    = 0.2
BATCH_SIZE = 64
LR_RATE    = 0.001
MAX_EPOCHS = 150
PATIENCE   = 10
CORR_THRESH = 0.10
MISSING_THRESH = 0.30
ZSCORE_WINDOW = 252

# Set USE_PARQUET=True to skip CSV+AlphaLib and load from precomputed parquet
USE_PARQUET = True

# Horizons to model (dropped log_ret5)
HORIZONS = {
    "log_ret20": 20,
    "log_ret30": 30,
    "log_ret60": 60,
}

# All horizons (for distribution analysis)
ALL_HORIZONS = {
    "log_ret1":  1,
    "log_ret5":  5,
    "log_ret20": 20,
    "log_ret30": 30,
    "log_ret60": 60,
}

np.random.seed(42)
torch.manual_seed(42)

# ======================================================================
# ALPHA HELPER FUNCTIONS
# ======================================================================
def sma(df, window=21):
    return df.rolling(window, min_periods=window // 2).mean()

def ts_sum(df, window=21):
    return df.rolling(window, min_periods=window // 2).sum()

def ts_min(df, window=21):
    return df.rolling(window, min_periods=window // 2).min()

def ts_median(df, window=21):
    return df.rolling(window, min_periods=window // 2).median()

def ts_max(df, window=21):
    return df.rolling(window, min_periods=window // 2).max()

def delay(df, n):
    return df.shift(n)

def rank(df, window=21):
    return df.rolling(window, min_periods=window // 2).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

def rolling_volatility(df, window=21):
    return df.rolling(window, min_periods=window // 2).std()

def rolling_skewness(df, window=21):
    if isinstance(df, pd.DataFrame):
        return pd.DataFrame({
            ticker: x.rolling(window, min_periods=window // 2).apply(
                lambda w: skew(w, bias=False), raw=True)
            for ticker, x in df.items()
        })
    return df.rolling(window, min_periods=window // 2).apply(
        lambda w: skew(w, bias=False), raw=True)

def rolling_kurtosis(df, window=21):
    return df.rolling(window, min_periods=window // 2).apply(
        lambda w: kurtosis(w, bias=False, fisher=True), raw=True)

def rolling_entropy(df, window=21, bins=30):
    def entropy_func(x):
        x = x[np.isfinite(x)]
        if len(x) < 3 or np.all(x == x[0]):
            return np.nan
        hist, _ = np.histogram(x, bins=bins, density=True)
        hist = np.clip(hist, 1e-12, None)
        p = hist / np.sum(hist)
        return -np.sum(p * np.log(p))
    return df.rolling(window, min_periods=window // 2).apply(entropy_func, raw=True)

def hurst_exponent(df, max_tau=256):
    x = np.asarray(df)
    n = len(x)
    if n < 16 or np.std(x) == 0:
        return np.nan
    lags = [2, 4, 8, 16, 32, 64, 128, 256]
    lags = [lag for lag in lags if lag < len(x) and lag <= max_tau]
    if len(lags) < 3:
        return np.nan
    tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
    if any(t <= 0 for t in tau):
        return np.nan
    slope, _ = np.polyfit(np.log(lags), np.log(tau), 1)
    return slope

def rolling_hurst_one_ticker(series, window=64, max_tau=64):
    return series.rolling(window, min_periods=window // 2).apply(
        lambda w: hurst_exponent(w, max_tau), raw=True)

def rolling_beta(y, x, window):
    if len(y) < window:
        return pd.Series(index=y.index, dtype=float)
    x_const = add_constant(x)
    model = RollingOLS(y, x_const, window=window)
    params = model.fit().params
    return params[x.name]

def rolling_corr(y, x, window):
    return y.rolling(window).corr(x)

def compute_ATR(high, low, close, window=14):
    prev_close = close.shift(1)
    high_diff = high - prev_close
    low_diff = prev_close - low
    high_low_diff = high - low
    tr = np.maximum(high_diff, low_diff)
    tr = np.maximum(tr, high_low_diff)
    return tr.rolling(window).mean()


# ======================================================================
# ALPHA LIBRARY
# ======================================================================
class AlphaLib:
    def __init__(self, daily_info):
        self.open = daily_info['open']
        self.high = daily_info['high']
        self.low = daily_info['low']
        self.close = daily_info['close']
        self.vwap = daily_info['vwap']
        self.volume = daily_info['volume']
        self.amount = daily_info['amount']
        self.returns = daily_info['returns']

    def calcu_alpha(self):
        alpha_dict = {}
        print("  Computing price-volume alphas...")
        alpha_dict['alpha01'] = ((self.close / sma(self.close, 10) - 1) * self.amount).div(self.amount.sum(1), axis=0)
        alpha_dict['alpha02'] = sma(self.amount, 5).div(self.amount.sum(1), axis=0)
        alpha_dict['alpha06'] = sma((self.close / sma(self.close, 5) - 1) * self.amount, 5).div(self.amount.sum(1), axis=0)
        alpha_dict['alpha07'] = ts_max((self.close / sma(self.close, 15) - 1) * self.amount, 15).div(self.amount.sum(1), axis=0)
        alpha_dict['alpha08'] = ts_min((self.close / ts_max(self.close, 60) - 1) * self.amount, 10).div(self.amount.sum(1), axis=0)
        alpha_dict['alpha09'] = ts_max((self.close / ts_min(self.close, 60) - 1) * self.amount, 50).div(self.amount.sum(1), axis=0)
        alpha_dict['alpha10'] = ts_max((self.close / ts_min(self.close, 20) - 1) * self.amount, 15).div(self.amount.sum(1), axis=0)
        alpha_dict['alpha12'] = ts_max((ts_max(self.high, 30) / ts_min(self.low, 30) - 1) * self.amount, 20).div(self.amount.sum(1), axis=0)
        alpha_dict['alpha13'] = ts_max(self.high - self.low, 30) / ts_min(self.close + self.open, 30)
        alpha_dict['alpha14'] = ts_sum(self.amount * (self.close - self.open), 5)
        alpha_dict['alpha15'] = ts_max(self.amount * (self.high - self.low), 15)
        alpha_dict['alpha16'] = ts_sum(self.amount * (self.close - self.vwap), 5)
        alpha_dict['alpha17'] = ts_min(self.amount * (self.low - self.vwap), 15)
        alpha_dict['alpha18'] = ts_min(self.amount * (self.open - self.vwap), 15)
        alpha_dict['alpha19'] = ts_min(self.amount * (self.open - self.low), 10)
        alpha_dict['alpha20'] = ts_max(self.amount * (self.close - self.low), 10)
        alpha_dict['alpha21'] = ts_median(self.amount, 15) / ts_sum(self.amount, 15)
        alpha_dict['alpha23'] = ts_max(self.amount, 15) / ts_min(self.amount, 10)
        alpha_dict['alpha24'] = ts_sum(self.amount.div(self.amount.sum(1), axis=0), 5)
        alpha_dict['alpha28'] = (ts_max(self.close, 5) / delay(self.close, 5)) * ts_min(self.close, 5) / self.close

        tmp2 = abs((self.close - self.open) / (self.high - self.low + 0.01))
        alpha_dict['alpha29'] = ts_max(tmp2, 6) / ts_min(tmp2, 6)
        alpha_dict['alpha30'] = tmp2 / delay(tmp2, 4)

        tmp3 = abs((self.low - self.open) / (self.close - self.low + 0.01))
        alpha_dict['alpha31'] = ts_max(tmp3, 4) / ts_min(tmp3, 4)

        tmp4 = abs((self.high - self.open) / (self.close - self.low + 0.01))
        alpha_dict['alpha32'] = ts_max(tmp4, 2) / ts_min(tmp4, 2)

        print("  Computing momentum & factor alphas...")
        alpha_dict['mom20'] = (self.close - delay(self.close, 20)) / delay(self.close, 20)
        alpha_dict['mom60'] = (self.close - delay(self.close, 60)) / delay(self.close, 60)
        alpha_dict['mom120'] = (self.close - delay(self.close, 120)) / delay(self.close, 120)
        alpha_dict['sharpe_mom20'] = (self.close - delay(self.close, 20)) / rolling_volatility(self.close, 20)

        alpha_dict['alpha_w_005'] = rank((self.open - (ts_sum(self.vwap, 10) / 10))) * (-1 * abs(rank((self.close - self.vwap))))
        alpha_dict['01'] = -delay(self.returns, 1)
        alpha_dict['02'] = -delay(self.returns, 5)
        alpha_dict['03'] = -delay(self.returns, 10)
        alpha_dict['04'] = -delay(self.returns, 20)
        alpha_dict['05'] = (self.close / delay(self.close, 20)) - 1
        alpha_dict['06'] = (self.close / delay(self.close, 60)) - 1
        alpha_dict['07'] = (self.close / delay(self.close, 120)) - 1
        alpha_dict['08'] = sma(self.returns, 5)
        alpha_dict['09'] = sma(self.returns, 10)
        alpha_dict['10'] = sma(self.returns, 20)
        alpha_dict['11'] = sma(self.returns, 60)

        iv_proxy = rolling_volatility(self.returns, 21)
        alpha_dict['60'] = rolling_volatility(iv_proxy, 21)

        turnover = self.volume / self.volume.rolling(21).mean()
        alpha_dict['os_ratior'] = turnover
        alpha_dict['os_ratio_chg'] = (turnover - sma(turnover, 126)) / sma(turnover, 126)

        trend_10 = (self.close > sma(self.close, 210)).astype(float)
        mom_12 = (self.close / delay(self.close, 252) - 1)
        vol_12 = rolling_volatility(self.returns, 252)
        alpha_dict['trend_10'] = trend_10 * (mom_12 / vol_12)
        alpha_dict['mom_12'] = mom_12
        alpha_dict['vol_12'] = vol_12

        gap = self.open / delay(self.close, 1) - 1
        alpha_dict['post_gap_drift'] = gap * delay(mom_12, 5)

        drawdown_20 = self.close / ts_max(self.close, 20) - 1
        alpha_dict['mom_stoploss'] = mom_12 * (drawdown_20 > -0.05).astype(float)

        if 'SPY' in self.close.columns:
            alpha_dict['pair_spread_spy'] = self.close.div(self.close['SPY'], axis=0)
        alpha_dict['style_mom_proxy'] = rank(mom_12)
        alpha_dict['max_daily_ret_21'] = self.returns.rolling(21).max()
        alpha_dict['vol_skew_proxy'] = rolling_skewness(self.returns, 21) * rolling_volatility(self.returns, 21)
        alpha_dict['seasonality_12m'] = delay(self.returns, 252)

        alpha_dict['size'] = -rank(self.amount)
        cycl = ts_sum(self.returns, 30)
        alpha_dict['cycl'] = rank(-cycl)
        z = (self.close - sma(self.close, 63)) / rolling_volatility(self.close, 63)
        alpha_dict['etf_stat'] = rank(-abs(z))

        long_mom = self.close / delay(self.close, 252) - 1
        short_react = ts_sum(self.returns, 30) / rolling_volatility(self.returns, 3)
        alpha_dict['long_short'] = rank(long_mom) * rank(short_react)

        vol_mean = sma(self.volume, 66)
        vol_std = rolling_volatility(vol_mean, 66)
        abnormal = (self.volume - vol_mean) / vol_std
        alpha_dict['abnormal'] = rank(abnormal.where(self.returns > 0.01))

        drawdown = self.close / ts_max(self.close, 300) - 1
        stability = rolling_volatility(self.returns, 63)
        extreme_loser = drawdown < -0.5
        stable = rank(stability) < 0.3
        alpha_dict['falling_knife'] = rank(extreme_loser) * rank(stable)

        print("  Computing hurst exponent (copper + key tickers only)...")
        hurst_tickers = [t for t in [COPPER, 'CL=F', 'GC=F', 'SPY', '^VIX']
                         if t in self.close.columns]
        hurst = pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
        for c in hurst_tickers:
            hurst[c] = rolling_hurst_one_ticker(self.close[c], 64)
        trend_63 = self.close / delay(self.close, 63) - 1
        alpha_dict['hurst'] = (hurst > 0.55) * trend_63 - (hurst < 0.75) * trend_63

        alpha_dict['entropy'] = -rolling_entropy(self.returns, 63)
        alpha_dict['range'] = (self.high - self.close) / (self.close - self.low)
        alpha_dict['skew+proxy'] = (self.high - self.low) / self.low

        print("  Computing residual momentum & misc alphas...")
        ret = np.log(self.close / self.close.shift(1))
        KEY_TICKERS = [t for t in [COPPER, 'CL=F', 'GC=F', 'SI=F', 'GLD', 'TLT',
                                    'SPY', 'QQQ', 'IWM', 'DX-Y.NYB', 'EURUSD=X',
                                    '^VIX', '^TNX', '^IRX', 'BZ=F', 'NG=F', 'ZC=F', 'ZS=F',
                                    'FCX', 'SCCO']
                       if t in self.close.columns]
        if 'SPY' in ret.columns:
            spy_ret = ret['SPY']
            beta_df = pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
            for ticker in KEY_TICKERS:
                if ticker == 'SPY':
                    continue
                tmp = pd.concat([ret[ticker], spy_ret], axis=1).dropna()
                if len(tmp) >= 252:
                    beta_df.loc[tmp.index, ticker] = rolling_beta(tmp.iloc[:, 0], tmp.iloc[:, 1], 252)
            alpha_dict['residual_momentum'] = mom_12 - beta_df.mul(mom_12.get('SPY', 0), axis=0)
        else:
            alpha_dict['residual_momentum'] = mom_12

        alpha_dict['intraday_range'] = (self.high - self.low) / self.close
        long_rev = -(self.close / delay(self.close, 1260) - 1)
        alpha_dict['long_rev_mom'] = long_rev * (mom_12 > 0)
        alpha_dict['earnings_yield_proxy'] = 1 / self.close
        alpha_dict['comomentum'] = rolling_corr(self.returns, self.returns.mean(axis=1), 52)
        alpha_dict['dtc_proxy'] = self.volume.rolling(21).mean() / self.volume
        alpha_dict['skew_mom'] = -alpha_dict['max_daily_ret_21'] * mom_12
        alpha_dict['overnight_ret'] = self.open / delay(self.close, 1) - 1
        alpha_dict['low_turnover'] = -turnover
        alpha_dict['abnormal_volume'] = self.volume / self.volume.rolling(63).mean()

        tr = compute_ATR(self.high, self.low, self.close)
        alpha_dict['ATR_trend'] = tr.rolling(14).mean()
        alpha_dict['entry'] = ts_max(self.close, 30) - 2 * tr
        alpha_dict['price_shock'] = self.returns.rolling(14).min()

        if 'SPY' in self.returns.columns:
            vix_proxy = rolling_volatility(self.returns['SPY'], 21)
            alpha_dict['sentiment_style'] = (vix_proxy < vix_proxy.rolling(126).mean()).astype(float).values.reshape(-1, 1) * mom_12

        print("  Computing vol/skew/kurtosis/entropy (21d, 63d)...")
        for w in [21, 63]:
            alpha_dict[f"vol_{w}"] = rolling_volatility(self.returns, w)
            alpha_dict[f"skewness_{w}"] = rolling_skewness(self.returns, w)
            alpha_dict[f"kurtosis_{w}"] = rolling_kurtosis(self.returns, w)
            alpha_dict[f"entropy_{w}"] = rolling_entropy(self.returns, w)

        print("  Computing lagged close/ret/ATR (5d, 10d, 30d)...")
        ret = np.log(self.close / self.close.shift(1)).replace([np.inf, -np.inf], np.nan)
        for lag in [5, 10, 30]:
            alpha_dict[f"close_{lag}"] = delay(self.close, lag)
            alpha_dict[f"ret_{lag}"] = delay(ret, lag)
            alpha_dict[f"close_mean_{lag}"] = sma(self.close, lag)
            alpha_dict[f"close_std_{lag}"] = rolling_volatility(self.close, lag)
            alpha_dict[f"close_max_{lag}"] = ts_max(self.close, lag)
            alpha_dict[f"close_min_{lag}"] = ts_min(self.close, lag)
            alpha_dict[f"ret_mean_{lag}"] = sma(ret, lag)
            alpha_dict[f"ret_std_{lag}"] = rolling_volatility(ret, lag)
            alpha_dict[f"ret_max_{lag}"] = ts_max(ret, lag)
            alpha_dict[f"ret_min_{lag}"] = ts_min(ret, lag)
            alpha_dict[f"ATR_{lag}"] = compute_ATR(self.high, self.low, self.close, lag)

        print("  Computing cross-asset beta & correlation (21d, 63d) for key tickers...")
        ret = np.log(self.close / self.close.shift(1)).replace([np.inf, -np.inf], np.nan)
        if 'SPY' in ret.columns:
            spy_ret = ret['SPY']
            for w in [21, 63]:
                beta_df = pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
                corr_df = pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
                for ticker in KEY_TICKERS:
                    if ticker == 'SPY':
                        continue
                    tmp = pd.concat([ret[ticker], spy_ret], axis=1, join='inner').dropna()
                    if tmp.shape[0] < w:
                        continue
                    y = tmp.iloc[:, 0]; x = tmp.iloc[:, 1]
                    beta_df.loc[:, ticker] = rolling_beta(y, x, window=w)
                    corr_df.loc[:, ticker] = rolling_corr(y, x, window=w)
                alpha_dict[f'beta_SPY_{w}'] = beta_df
                alpha_dict[f'corr_SPY_{w}'] = corr_df

        if '^VIX' in self.close.columns:
            vix_ret = np.log(self.close['^VIX'] / self.close['^VIX'].shift(1)).replace([np.inf, -np.inf], np.nan)
            for w in [21, 63]:
                beta_vix = pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
                corr_vix = pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=float)
                for ticker in KEY_TICKERS:
                    if ticker == '^VIX':
                        continue
                    tmp = pd.concat([ret[ticker], vix_ret], axis=1, join='inner').dropna()
                    if tmp.shape[0] < w:
                        continue
                    y = tmp.iloc[:, 0]; x = tmp.iloc[:, 1]
                    beta_vix.loc[:, ticker] = rolling_beta(y, x, window=w)
                    corr_vix.loc[:, ticker] = rolling_corr(y, x, window=w)
                alpha_dict[f'beta_VIX_{w}'] = beta_vix
                alpha_dict[f'corr_VIX_{w}'] = corr_vix

        print(f"  Total alphas computed: {len(alpha_dict)}")
        return alpha_dict


# ======================================================================
# DONCHIAN & KALMAN HELPERS
# ======================================================================
def kalman_trend(log_prices):
    y = log_prices.values; n = len(y)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([1.0, 0.0])
    Q = np.array([[1e-6, 0], [0, 3e-7]])
    valid = y[~np.isnan(y)]
    R = np.nanvar(np.diff(valid[:252])) if len(valid) > 252 else 1e-5
    x = np.array([valid[0], 0.0])
    P = np.eye(2) * 1e-4
    levels = np.full(n, np.nan); slopes = np.full(n, np.nan)
    for t in range(n):
        if np.isnan(y[t]):
            x = F @ x; P = F @ P @ F.T + Q
        else:
            xp = F @ x; Pp = F @ P @ F.T + Q
            inn = y[t] - H @ xp; S = H @ Pp @ H + R
            K = (Pp @ H) / S
            x = xp + K * inn; P = Pp - np.outer(K, H) @ Pp
        levels[t] = x[0]; slopes[t] = x[1]
    return (pd.Series(levels, index=log_prices.index),
            pd.Series(slopes, index=log_prices.index))

def donchian_stateful(px, n_entry, n_exit):
    he = px.rolling(n_entry).max().shift(1).values
    le = px.rolling(n_entry).min().shift(1).values
    hx = px.rolling(n_exit).max().shift(1).values
    lx = px.rolling(n_exit).min().shift(1).values
    p = px.values; pos = np.zeros(len(p)); cur = 0
    for i in range(n_entry, len(p)):
        if cur == 0:
            if   p[i] > he[i] and not np.isnan(he[i]): cur =  1
            elif p[i] < le[i] and not np.isnan(le[i]): cur = -1
        elif cur == 1:
            if p[i] < lx[i] and not np.isnan(lx[i]):
                cur = 0
                if p[i] < le[i] and not np.isnan(le[i]): cur = -1
        elif cur == -1:
            if p[i] > hx[i] and not np.isnan(hx[i]):
                cur = 0
                if p[i] > he[i] and not np.isnan(he[i]): cur = 1
        pos[i] = cur
    return pd.Series(pos, index=px.index)


# ======================================================================
# 1. LOAD & PREPARE DATA
# ======================================================================
print("=" * 70)
print("  STEP 1: Load & Prepare Data")
print("=" * 70)

parquet_path = f"{BASE}/claude/copper_alpha_features.parquet"

if USE_PARQUET and os.path.exists(parquet_path):
    # ---- FAST PATH: Load from precomputed parquet ----
    print(f"Loading from parquet: {parquet_path}")
    pq = pd.read_parquet(parquet_path)
    print(f"Loaded: {pq.shape[0]} rows x {pq.shape[1]} cols")
    print(f"Date range: {pq.index.min().date()} to {pq.index.max().date()}")

else:
    # ---- FULL BUILD: Load CSVs + compute AlphaLib features ----
    print("Loading CSVs...")
    market = pd.read_csv(f"{BASE}/market_data_2005_2025.csv", parse_dates=["datetime"])
    commod = pd.read_csv(f"{BASE}/commodities_data_1990_2025.csv", parse_dates=["datetime"])
    compan = pd.read_csv(f"{BASE}/companies_data_1990_2025.csv", parse_dates=["datetime"])

    df_all = pd.concat([market, commod, compan], ignore_index=True)
    df_all = df_all[df_all["datetime"] >= "2005-01-01"]
    df_all = df_all.sort_values("datetime").drop_duplicates(subset=["datetime", "ticker"], keep="last")

    print(f"Combined: {df_all.shape[0]} rows, {df_all['ticker'].nunique()} tickers")
    print(f"Date range: {df_all['datetime'].min().date()} to {df_all['datetime'].max().date()}")

    df_all.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close_raw", "Adj Close": "close", "Volume": "volume",
        "datetime": "trade_date"
    }, inplace=True)

    if df_all["close"].isna().all():
        df_all["close"] = df_all["close_raw"]

    daily_info = {}
    daily_info['open']   = df_all.pivot(index='trade_date', columns='ticker', values='open')
    daily_info['close']  = df_all.pivot(index='trade_date', columns='ticker', values='close')
    daily_info['high']   = df_all.pivot(index='trade_date', columns='ticker', values='high')
    daily_info['low']    = df_all.pivot(index='trade_date', columns='ticker', values='low')
    daily_info['volume'] = df_all.pivot(index='trade_date', columns='ticker', values='volume')

    for k in daily_info:
        daily_info[k] = daily_info[k].sort_index().ffill()

    daily_info['amount']  = daily_info['volume'] * daily_info['close']
    daily_info['vwap']    = (daily_info['amount'] * 1000) / (daily_info['volume'] * 100 + 1)
    daily_info['returns'] = np.log(daily_info['close'] / daily_info['close'].shift(1))

    print(f"Pivoted shape: {daily_info['close'].shape}")

    print("\nComputing alphas (this may take several minutes)...")
    alpha_lib = AlphaLib(daily_info)
    alpha_dict = alpha_lib.calcu_alpha()

    print(f"\nExtracting {COPPER} features from alpha_dict...")
    copper_features = {}
    for key, val in alpha_dict.items():
        if isinstance(val, pd.DataFrame):
            if COPPER in val.columns:
                copper_features[key] = val[COPPER]
        elif isinstance(val, pd.Series):
            copper_features[key] = val

    cross_asset_keys = [k for k in alpha_dict if k.startswith(('beta_', 'corr_'))]
    for key in cross_asset_keys:
        df_ca = alpha_dict[key]
        if isinstance(df_ca, pd.DataFrame) and COPPER in df_ca.columns:
            for ticker in df_ca.columns:
                if ticker != COPPER:
                    copper_features[f"{key}_{ticker}"] = df_ca[ticker]

    print(f"Extracted {len(copper_features)} features for {COPPER}")

    pq = pd.DataFrame(copper_features)
    pq.index.name = "trade_date"

    close_tmp = daily_info['close'][COPPER].reindex(pq.index)
    ret_tmp = close_tmp.pct_change()
    pq["close"] = close_tmp

    # Compute log return targets + z-score versions
    for name, h in ALL_HORIZONS.items():
        raw_ret = np.log(close_tmp.shift(-h) / close_tmp)
        pq[name] = raw_ret
        roll_mean = raw_ret.rolling(ZSCORE_WINDOW, min_periods=60).mean()
        roll_std  = raw_ret.rolling(ZSCORE_WINDOW, min_periods=60).std().replace(0, np.nan)
        pq[f"{name}_zscore"] = (raw_ret - roll_mean) / roll_std

    # Recompute signals
    print("\nRecomputing signals...")
    vix_tmp   = daily_info['close'].get('^VIX', pd.Series(dtype=float)).reindex(pq.index).ffill()
    dxy_tmp   = daily_info['close'].get('DX-Y.NYB', pd.Series(dtype=float)).reindex(pq.index).ffill()

    ewm_vol_tmp    = ret_tmp.ewm(span=21, adjust=False).std() * np.sqrt(252)
    ewm_vol_tmp    = ewm_vol_tmp.replace(0, np.nan).ffill()
    vix_smooth_tmp = vix_tmp.ewm(span=5).mean()
    dxy_mom_tmp    = dxy_tmp.pct_change(40)

    kf_level, kf_slope_tmp = kalman_trend(np.log(close_tmp.dropna()))
    kf_slope_tmp = kf_slope_tmp.reindex(pq.index)
    pq["kf_slope"] = kf_slope_tmp

    from hmmlearn.hmm import GaussianHMM
    def fit_hmm_regimes(returns_series, split_date, n_regimes=2):
        r_clean = returns_series.dropna()
        r_train = r_clean[r_clean.index < split_date]
        X_train = r_train.values.reshape(-1, 1) * 100
        model = GaussianHMM(n_components=n_regimes, covariance_type="full",
                            n_iter=200, random_state=42)
        model.fit(X_train)
        X_full = r_clean.values.reshape(-1, 1) * 100
        probs = model.predict_proba(X_full)
        vars_ = [model.covars_[i][0, 0] for i in range(n_regimes)]
        trend_regime = np.argmin(vars_)
        p_trend_s = pd.Series(probs[:, trend_regime], index=r_clean.index)
        return p_trend_s.reindex(returns_series.index).ffill().fillna(0.5)

    p_trend_tmp = fit_hmm_regimes(ret_tmp, SPLIT_TEST)
    pq["p_trend"] = p_trend_tmp

    close_clean_tmp = close_tmp.dropna()
    raw_pos_tmp = donchian_stateful(close_clean_tmp, 30, 10).reindex(pq.index).fillna(0)
    pq["raw_pos_donchian"] = raw_pos_tmp
    pq["ewm_vol"] = ewm_vol_tmp
    pq["vix_smooth"] = vix_smooth_tmp
    pq["dxy_mom"] = dxy_mom_tmp

    # Dual TF Donchian
    raw3_short = donchian_stateful(close_clean_tmp, 20, 7).reindex(pq.index).fillna(0)
    raw3_long  = donchian_stateful(close_clean_tmp, 60, 20).reindex(pq.index).fillna(0)
    slope_s = np.sign(kf_slope_tmp)
    raw3_tmp = pd.Series(
        np.where((raw3_short > 0) & ((raw3_long > 0) | (slope_s > 0)), 1,
        np.where((raw3_short < 0) & ((raw3_long < 0) | (slope_s < 0)), -1,
        np.where((raw3_short != 0) & (raw3_long == raw3_short), raw3_short, 0))),
        index=pq.index, dtype=float)
    pq["raw_pos_dualtf"] = raw3_tmp

    print(f"Total feature DataFrame: {pq.shape}")

    # Save parquet
    pq.to_parquet(parquet_path)
    print(f"Saved: {parquet_path}")

# Verify key columns
target_cols_check = list(ALL_HORIZONS.keys()) + [f"{h}_zscore" for h in ALL_HORIZONS]
missing_targets = [c for c in target_cols_check if c not in pq.columns]
if missing_targets:
    raise ValueError(f"Missing target columns: {missing_targets}")

signal_cols_needed = ["kf_slope", "p_trend", "raw_pos_donchian", "raw_pos_dualtf",
                      "ewm_vol", "vix_smooth", "dxy_mom", "close"]
for c in signal_cols_needed:
    assert c in pq.columns, f"Missing signal column: {c}"

print(f"Targets: {list(HORIZONS.keys())}")
print(f"Z-score targets: {[f'{h}_zscore' for h in HORIZONS]}")

# Extract key series
close = pq["close"]
ret = close.pct_change()
ewm_vol = pq["ewm_vol"]
vix_smooth = pq["vix_smooth"]
dxy_mom = pq["dxy_mom"]
kf_slope = pq["kf_slope"]
p_trend = pq["p_trend"]
raw_pos = pq["raw_pos_donchian"]
raw3 = pq["raw_pos_dualtf"]

# Split periods
train_mask = pq.index < SPLIT_VAL
val_mask   = (pq.index >= SPLIT_VAL) & (pq.index < SPLIT_TEST)
test_mask  = pq.index >= SPLIT_TEST
print(f"Split: Train {train_mask.sum()} | Val {val_mask.sum()} | Test {test_mask.sum()}")

# ======================================================================
# 2. RETURN DISTRIBUTION ANALYSIS
# ======================================================================
print("\n" + "=" * 70)
print("  STEP 2: Return Distribution Analysis")
print("=" * 70)

dist_fig, dist_axes = plt.subplots(len(ALL_HORIZONS), 3, figsize=(18, 4 * len(ALL_HORIZONS)))
dist_fig.suptitle("Return Distribution Analysis", fontsize=14, fontweight="bold", y=1.01)

summary_rows = []
for i, (name, h) in enumerate(ALL_HORIZONS.items()):
    y = pq[name].dropna()

    jb_stat, jb_p = stats.jarque_bera(y)
    adf_result = adfuller(y, maxlag=20)
    adf_stat, adf_p = adf_result[0], adf_result[1]
    acf_vals = acf(y, nlags=20, fft=True)

    summary_rows.append({
        "Horizon": name, "N": len(y),
        "Mean": f"{y.mean():.6f}", "Std": f"{y.std():.4f}",
        "Skew": f"{y.skew():.3f}", "Kurt": f"{y.kurt():.3f}",
        "JB_stat": f"{jb_stat:.1f}", "JB_p": f"{jb_p:.4f}",
        "ADF_stat": f"{adf_stat:.3f}", "ADF_p": f"{adf_p:.4f}",
        "ACF_1": f"{acf_vals[1]:.3f}",
    })

    ax = dist_axes[i, 0]
    ax.hist(y, bins=80, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_title(f"{name}  (JB p={jb_p:.4f})", fontsize=9)
    ax.axvline(0, color="red", lw=0.8, ls="--")

    ax = dist_axes[i, 1]
    qqplot(y, line="45", ax=ax, markersize=1, alpha=0.5)
    ax.set_title(f"{name} Q-Q", fontsize=9)

    ax = dist_axes[i, 2]
    lags = np.arange(1, 21)
    ax.bar(lags, acf_vals[1:21], color="teal", edgecolor="white", width=0.6)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(1.96 / np.sqrt(len(y)), color="red", ls="--", lw=0.7)
    ax.axhline(-1.96 / np.sqrt(len(y)), color="red", ls="--", lw=0.7)
    ax.set_title(f"{name} ACF", fontsize=9)
    ax.set_xlabel("Lag")

dist_fig.tight_layout()
dist_fig.savefig(f"{BASE}/claude/dist_analysis.png", dpi=120, bbox_inches="tight")
plt.close(dist_fig)

print("\nDistribution Summary:")
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

# ======================================================================
# 3. FEATURE-RETURN CORRELATIONS
# ======================================================================
print("\n" + "=" * 70)
print("  STEP 3: Feature-Return Correlations")
print("=" * 70)

target_and_zscore = list(ALL_HORIZONS.keys()) + [f"{n}_zscore" for n in ALL_HORIZONS.keys()] + ["close"]
feature_cols = [c for c in pq.columns if c not in target_and_zscore
                and pq[c].dtype in [np.float64, np.float32, np.int64]]
print(f"Total feature candidates: {len(feature_cols)}")

train_data = pq[train_mask]
nan_pct = train_data[feature_cols].isna().mean()
keep_cols = nan_pct[nan_pct <= MISSING_THRESH].index.tolist()
dropped = len(feature_cols) - len(keep_cols)
print(f"Dropped {dropped} features with >{MISSING_THRESH*100:.0f}% missing -> {len(keep_cols)} remaining")
feature_cols = keep_cols

train_for_corr = train_data[feature_cols + list(HORIZONS.keys())].dropna(subset=list(HORIZONS.keys()))
corr_results = {}
selected_features = {}

for target_name in HORIZONS.keys():
    target_vals = train_for_corr[target_name]
    corrs = {}
    for feat in feature_cols:
        feat_vals = train_for_corr[feat]
        valid = feat_vals.notna() & target_vals.notna()
        if valid.sum() > 100:
            c, _ = stats.pearsonr(feat_vals[valid], target_vals[valid])
            corrs[feat] = c
    corr_series = pd.Series(corrs)
    corr_results[target_name] = corr_series

    selected = corr_series[corr_series.abs() > CORR_THRESH].sort_values(key=abs, ascending=False)
    selected_features[target_name] = selected
    print(f"\n  {target_name}: {len(selected)} features pass |corr| > {CORR_THRESH}")
    if len(selected) > 0:
        print(f"    Top 10: {list(selected.head(10).index)}")
        print(f"    Top corrs: {selected.head(10).round(3).to_dict()}")

# Correlation heatmap
all_top_feats = set()
for v in selected_features.values():
    all_top_feats.update(v.head(15).index.tolist())
all_top_feats = sorted(all_top_feats)[:30]

if len(all_top_feats) > 3:
    heatmap_data = pd.DataFrame({tgt: corr_results[tgt].reindex(all_top_feats)
                                  for tgt in HORIZONS.keys()})
    fig_heat, ax_heat = plt.subplots(figsize=(10, max(8, len(all_top_feats) * 0.35)))
    im = ax_heat.imshow(heatmap_data.values, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax_heat.set_xticks(range(len(HORIZONS)))
    ax_heat.set_xticklabels(list(HORIZONS.keys()), fontsize=9)
    ax_heat.set_yticks(range(len(all_top_feats)))
    ax_heat.set_yticklabels(all_top_feats, fontsize=7)
    for i in range(len(all_top_feats)):
        for j in range(len(HORIZONS)):
            v = heatmap_data.values[i, j]
            if not np.isnan(v):
                ax_heat.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
    fig_heat.colorbar(im, ax=ax_heat, shrink=0.8)
    ax_heat.set_title("Feature-Return Correlations (Train Set)", fontsize=11, fontweight="bold")
    fig_heat.tight_layout()
    fig_heat.savefig(f"{BASE}/claude/correlation_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig_heat)
    print(f"\nCorrelation heatmap saved.")

# ======================================================================
# 4. FEATURE SELECTION SUMMARY
# ======================================================================
print("\n" + "=" * 70)
print("  STEP 4: Feature Selection Summary")
print("=" * 70)

def categorize_feature(name):
    if name.startswith("alpha"): return "Alpha"
    if "mom" in name.lower() or "trend" in name.lower(): return "Momentum"
    if name.startswith(("beta_", "corr_")): return "CrossAsset"
    if any(m in name for m in ["CPI", "PPI", "FED", "IND_PROD", "TREASURY", "OVERNIGHT", "M2"]): return "Macro"
    if any(v in name.lower() for v in ["vol_", "skew", "kurt", "entropy", "ATR"]): return "VolStat"
    if name in ["kf_slope", "p_trend", "raw_pos_donchian", "raw_pos_dualtf",
                 "ewm_vol", "vix_smooth", "dxy_mom"]: return "Signal"
    return "Other"

for target_name in HORIZONS.keys():
    sel = selected_features[target_name]
    if len(sel) == 0:
        print(f"\n  {target_name}: No features selected")
        continue
    cats = pd.Series([categorize_feature(f) for f in sel.index])
    print(f"\n  {target_name}: {len(sel)} features selected")
    for cat, count in cats.value_counts().items():
        feats_in_cat = [f for f, c in zip(sel.index, cats) if c == cat]
        print(f"    {cat:12s}: {count:3d}  ({', '.join(feats_in_cat[:5])}{'...' if len(feats_in_cat) > 5 else ''})")

# ======================================================================
# 4.5 ALPHA SIGNAL EVALUATION
# ======================================================================
print("\n" + "=" * 70)
print("  STEP 4.5: Alpha Signal Evaluation")
print("=" * 70)

def compute_rolling_ic(alpha_arr, return_arr, window=252):
    """Rolling Spearman IC via rank-then-Pearson on numpy arrays."""
    n = len(alpha_arr)
    ics = np.full(n, np.nan)
    for i in range(window, n):
        a = alpha_arr[i-window:i]
        r = return_arr[i-window:i]
        mask = np.isfinite(a) & np.isfinite(r)
        if mask.sum() > 50:
            ra = stats.rankdata(a[mask]).astype(float)
            rr = stats.rankdata(r[mask]).astype(float)
            ra -= ra.mean(); rr -= rr.mean()
            d = np.sqrt(np.dot(ra, ra) * np.dot(rr, rr))
            ics[i] = np.dot(ra, rr) / d if d > 0 else 0
    return ics

# --- 4.5a: Per-alpha metrics on TRAIN set ---
print("\n--- 4.5a: Per-alpha signal metrics (Train set) ---")

alpha_signal_metrics = {}

for target_name, h in HORIZONS.items():
    print(f"\n  Computing metrics for {target_name} (horizon={h})...")
    target_s = train_for_corr[target_name]

    horizon_metrics = {}
    for feat in feature_cols:
        feat_s = train_for_corr[feat]
        valid = feat_s.notna() & target_s.notna()
        if valid.sum() < 252:
            continue
        fv = feat_s[valid].values
        tv = target_s[valid].values

        pearson_val = corr_results[target_name].get(feat, np.nan)
        ic_val, _ = stats.spearmanr(fv, tv)
        dir_acc = np.mean(np.sign(fv) == np.sign(tv)) * 100
        turnover_val = np.mean(np.abs(np.diff(fv))) if len(fv) > 1 else np.nan

        horizon_metrics[feat] = dict(
            pearson=pearson_val, ic=ic_val,
            dir_acc=dir_acc, turnover=turnover_val)

    # Rolling IC for features with |IC| > 0.02 (pre-filter for speed)
    ic_cands = [f for f, m in horizon_metrics.items() if abs(m['ic']) > 0.02]
    print(f"    {len(ic_cands)}/{len(horizon_metrics)} pass |IC|>0.02 for rolling IC")

    target_arr = train_for_corr[target_name].values
    for feat in ic_cands:
        feat_arr = train_for_corr[feat].fillna(0).values
        rics = compute_rolling_ic(feat_arr, target_arr, 252)
        v_ics = rics[np.isfinite(rics)]
        if len(v_ics) > 10:
            m_ic = np.mean(v_ics)
            s_ic = np.std(v_ics)
            ir = m_ic / s_ic if s_ic > 0 else 0
        else:
            m_ic = horizon_metrics[feat]['ic']
            s_ic = np.nan
            ir = 0
        horizon_metrics[feat].update(
            mean_rolling_ic=m_ic, std_rolling_ic=s_ic, ir=ir)

    # Defaults for features that didn't pass pre-filter
    for feat in horizon_metrics:
        if 'ir' not in horizon_metrics[feat]:
            horizon_metrics[feat].update(
                mean_rolling_ic=horizon_metrics[feat]['ic'],
                std_rolling_ic=np.nan, ir=0)

    alpha_signal_metrics[target_name] = horizon_metrics

    # Print top 30 by |IR|
    ranked = sorted(horizon_metrics.items(),
                    key=lambda x: abs(x[1]['ir']), reverse=True)
    print(f"\n  Top 30 alphas by |IR| for {target_name}:")
    print(f"    {'Feature':<35} {'Pearson':>8} {'IC':>8} {'MeanIC':>8} "
          f"{'IR':>8} {'DirAcc':>8}")
    print(f"    {'-'*78}")
    for feat, m in ranked[:30]:
        print(f"    {feat:<35} {m['pearson']:>8.4f} {m['ic']:>8.4f} "
              f"{m.get('mean_rolling_ic',0):>8.4f} {m['ir']:>8.4f} "
              f"{m['dir_acc']:>7.1f}%")

# --- 4.5b: Select top alpha signals ---
print("\n--- 4.5b: Select top alpha signals ---")

selected_alphas_per_hz = {}
all_selected_alphas = set()

for target_name in HORIZONS.keys():
    hm = alpha_signal_metrics[target_name]
    cands = [(f, m) for f, m in hm.items()
             if abs(m['ir']) > 0.10
             and abs(m.get('mean_rolling_ic', 0)) > 0.03
             and m['dir_acc'] > 52]
    cands.sort(key=lambda x: abs(x[1]['ir']), reverse=True)
    sel = cands[:10]
    selected_alphas_per_hz[target_name] = sel
    for f, _ in sel:
        all_selected_alphas.add(f)
    print(f"  {target_name}: {len(cands)} pass filters, top {len(sel)} selected")
    for f, m in sel:
        print(f"    {f:<35} IR={m['ir']:+.4f}  IC={m['ic']:+.4f}  "
              f"DA={m['dir_acc']:.1f}%")

if len(all_selected_alphas) == 0:
    print("  Relaxing: |IR|>0.05, |IC|>0.02, DirAcc>50%")
    for target_name in HORIZONS.keys():
        hm = alpha_signal_metrics[target_name]
        cands = [(f, m) for f, m in hm.items()
                 if abs(m['ir']) > 0.05
                 and abs(m.get('mean_rolling_ic', 0)) > 0.02
                 and m['dir_acc'] > 50]
        cands.sort(key=lambda x: abs(x[1]['ir']), reverse=True)
        sel = cands[:10]
        selected_alphas_per_hz[target_name] = sel
        for f, _ in sel:
            all_selected_alphas.add(f)
        print(f"  {target_name}: {len(sel)} selected (relaxed)")

if len(all_selected_alphas) == 0:
    print("  Further relaxing: top 5 by |IR| per horizon (no threshold)")
    for target_name in HORIZONS.keys():
        hm = alpha_signal_metrics[target_name]
        ranked_all = sorted(hm.items(),
                            key=lambda x: abs(x[1]['ir']), reverse=True)
        sel = ranked_all[:5]
        selected_alphas_per_hz[target_name] = sel
        for f, _ in sel:
            all_selected_alphas.add(f)
        print(f"  {target_name}: top 5 selected (no threshold)")

all_selected_alphas = sorted(all_selected_alphas)
print(f"\n  Unique alpha signals: {len(all_selected_alphas)}")

# --- 4.5c: Build alpha-based positions ---
print("\n--- 4.5c: Build alpha-based positions ---")

vix_mult_alpha = pd.Series(
    np.where(vix_smooth > 35, 0.30,
    np.where(vix_smooth > 25, 0.60, 1.00)), index=close.index)

alpha_positions = {}
for feat in all_selected_alphas:
    raw_alpha = pq[feat].copy()
    roll_m = raw_alpha.rolling(ZSCORE_WINDOW, min_periods=60).mean()
    roll_s = raw_alpha.rolling(ZSCORE_WINDOW, min_periods=60).std().replace(0, np.nan)
    z_alpha = ((raw_alpha - roll_m) / roll_s).clip(-5, 5).fillna(0)

    # Signal direction from highest-|IR| horizon
    best_ir = 0; best_sign = 1.0
    for tn in HORIZONS:
        if feat in alpha_signal_metrics[tn]:
            ir_v = alpha_signal_metrics[tn][feat]['ir']
            if abs(ir_v) > abs(best_ir):
                best_ir = ir_v
                s = np.sign(alpha_signal_metrics[tn][feat]['ic'])
                best_sign = float(s) if s != 0 else 1.0

    raw_pos_a = np.tanh(z_alpha * best_sign)
    pos_a = (raw_pos_a * (TARGET_VOL / ewm_vol)).clip(-3, 3)
    dm_a = pd.Series(np.where((raw_pos_a > 0) & (dxy_mom > 0.04), 0.65, 1.00),
                     index=close.index)
    pos_a = pos_a * vix_mult_alpha * dm_a
    pos_a = pos_a.shift(1).fillna(0)
    alpha_positions[feat] = pos_a
    print(f"  Alpha_{feat[:30]:<30s}: mean={pos_a.mean():+.4f}  "
          f"std={pos_a.std():.3f}  sign={best_sign:+.0f}")

# Top 5 by max |IR| for composite
alpha_ir_scores = {}
for feat in all_selected_alphas:
    max_ir = max((abs(alpha_signal_metrics[tn][feat]['ir'])
                  for tn in HORIZONS if feat in alpha_signal_metrics[tn]),
                 default=0)
    alpha_ir_scores[feat] = max_ir
top5_alphas = sorted(alpha_ir_scores, key=alpha_ir_scores.get, reverse=True)[:5]
print(f"\n  Top 5 for composite: {top5_alphas}")

if len(top5_alphas) > 0:
    alpha_composite_pos = sum(alpha_positions[f] for f in top5_alphas) / len(top5_alphas)
else:
    alpha_composite_pos = pd.Series(0.0, index=close.index)
print(f"  Alpha Composite: mean={alpha_composite_pos.mean():+.4f}  "
      f"std={alpha_composite_pos.std():.3f}")

# --- 4.5d: Test-set preview ---
print("\n--- 4.5d: Alpha strategy test-set preview ---")
print(f"  {'Signal':<35} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>10} {'Hit':>8}")
print(f"  {'-'*75}")
alpha_preview_list = list(alpha_positions.keys())[:10] + ['_composite_']
for feat in alpha_preview_list:
    if feat == '_composite_':
        p_ev = alpha_composite_pos; lbl = "Alpha_Composite"
    else:
        p_ev = alpha_positions[feat]; lbl = f"Alpha_{feat[:25]}"
    r_ev = (p_ev * ret).dropna()
    r_te = r_ev[r_ev.index >= SPLIT_TEST]
    if len(r_te) > 21 and r_te.std() > 0:
        sh = (r_te.mean() * 252) / (r_te.std() * np.sqrt(252))
        ar = r_te.mean() * 252
        cum_e = (1 + r_te).cumprod()
        dd_e = ((cum_e - cum_e.cummax()) / cum_e.cummax()).min()
        hit_e = (r_te > 0).mean() * 100
    else:
        sh = 0; ar = 0; dd_e = 0; hit_e = 50
    print(f"  {lbl:<35} {sh:>+8.2f} {ar*100:>+9.1f}% "
          f"{dd_e*100:>+9.1f}% {hit_e:>7.1f}%")

# ======================================================================
# 5. LINEAR REGRESSION BASELINE (per horizon)
# ======================================================================
print("\n" + "=" * 70)
print("  STEP 5: Linear Regression Baseline (per horizon)")
print("=" * 70)

def eval_metrics(pred, actual, label, verbose=True):
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    dir_acc = np.mean(np.sign(pred) == np.sign(actual)) * 100
    ic, _ = stats.spearmanr(pred, actual)
    if verbose:
        print(f"  {label:<16s}  MSE={mse:.6f}  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  DirAcc={dir_acc:.1f}%  IC={ic:.3f}")
    return dict(mse=mse, rmse=rmse, mae=mae, r2=r2, dir_acc=dir_acc, ic=ic)

lr_models = {}
lr_predictions = {}
lr_metrics = {}
lr_feature_sets = {}

for target_name in HORIZONS.keys():
    print(f"\n--- {target_name} ---")

    horizon_feats = [f for f in selected_features[target_name].index.tolist()
                     if not f.startswith("log_ret")]
    if len(horizon_feats) < 3:
        all_sel = set()
        for v in selected_features.values():
            all_sel.update(v.index.tolist())
        horizon_feats = sorted([f for f in all_sel if not f.startswith("log_ret")])
    print(f"  Features: {len(horizon_feats)}")
    lr_feature_sets[target_name] = horizon_feats

    df_lr = pq[horizon_feats + [target_name]].dropna(subset=[target_name])
    df_lr_feats = df_lr[horizon_feats].fillna(0)
    df_lr_target = df_lr[target_name]

    tr_mask = df_lr.index < SPLIT_VAL
    va_mask = (df_lr.index >= SPLIT_VAL) & (df_lr.index < SPLIT_TEST)
    te_mask = df_lr.index >= SPLIT_TEST

    X_train = df_lr_feats[tr_mask].values
    y_train = df_lr_target[tr_mask].values
    X_val = df_lr_feats[va_mask].values
    y_val = df_lr_target[va_mask].values
    X_test = df_lr_feats[te_mask].values
    y_test = df_lr_target[te_mask].values

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_models[target_name] = lr_model

    pred_tr = lr_model.predict(X_train)
    pred_va = lr_model.predict(X_val)
    pred_te = lr_model.predict(X_test)

    lr_predictions[target_name] = {
        "train": pd.Series(pred_tr, index=df_lr.index[tr_mask]),
        "val":   pd.Series(pred_va, index=df_lr.index[va_mask]),
        "test":  pd.Series(pred_te, index=df_lr.index[te_mask]),
    }

    lr_metrics[target_name] = {}
    lr_metrics[target_name]["train"] = eval_metrics(pred_tr, y_train, "Train")
    lr_metrics[target_name]["val"]   = eval_metrics(pred_va, y_val, "Val")
    lr_metrics[target_name]["test"]  = eval_metrics(pred_te, y_test, "Test")

# ======================================================================
# 6. LSTM MODEL TRAINING (per horizon)
# ======================================================================
print("\n" + "=" * 70)
print("  STEP 6: LSTM Model Training (per horizon)")
print("=" * 70)

class CopperDataset(Dataset):
    def __init__(self, features, targets, seq_len):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.features) - self.seq_len
    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return x, y

class CopperLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)

def predict_dataset(model, dataloader, device):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            preds.extend(pred.cpu().numpy())
            actuals.extend(y_batch.numpy())
    return np.array(preds), np.array(actuals)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

lstm_models = {}
lstm_predictions = {}
lstm_metrics = {}
lstm_training_curves = {}
lstm_feature_sets = {}
lstm_norm_stats = {}

for target_name in HORIZONS.keys():
    print(f"\n{'='*50}")
    print(f"  Training LSTM for {target_name}")
    print(f"{'='*50}")

    target_z = f"{target_name}_zscore"

    lstm_features = lr_feature_sets[target_name]
    lstm_feature_sets[target_name] = lstm_features
    print(f"  Features: {len(lstm_features)}")

    df_model = pq[lstm_features + [target_name, target_z]].copy()
    df_model = df_model.dropna(subset=[target_z])

    train_end_idx = df_model.index < SPLIT_VAL
    train_feat = df_model.loc[train_end_idx, lstm_features]
    feat_mean = train_feat.mean()
    feat_std = train_feat.std().replace(0, 1)
    lstm_norm_stats[target_name] = (feat_mean, feat_std)

    df_norm = df_model.copy()
    df_norm[lstm_features] = (df_model[lstm_features] - feat_mean) / feat_std
    df_norm[lstm_features] = df_norm[lstm_features].fillna(0).clip(-5, 5)

    dates = df_norm.index
    train_idx = dates < SPLIT_VAL
    val_idx = (dates >= SPLIT_VAL) & (dates < SPLIT_TEST)
    test_idx = dates >= SPLIT_TEST

    X_all = df_norm[lstm_features].values
    y_all_z = df_norm[target_z].values
    y_all_raw = df_norm[target_name].values

    train_ds = CopperDataset(X_all[train_idx], y_all_z[train_idx], SEQ_LEN)
    val_ds = CopperDataset(X_all[val_idx], y_all_z[val_idx], SEQ_LEN)
    test_ds = CopperDataset(X_all[test_idx], y_all_z[test_idx], SEQ_LEN)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} sequences")

    torch.manual_seed(42)
    model = CopperLSTM(len(lstm_features), HIDDEN, N_LAYERS, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    print(f"  Training (max {MAX_EPOCHS} epochs, patience {PATIENCE})...")
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0; n_batches = 0
        for X_batch, y_batch in train_dl:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0; n_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item(); n_val += 1
        val_loss = val_loss / max(n_val, 1)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss; patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            print(f"    Epoch {epoch+1:3d}  train={train_loss:.6f}  val={val_loss:.6f}  {'*' if patience_counter == 0 else ''}")

        if patience_counter >= PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  Best val loss: {best_val_loss:.6f} ({len(train_losses)} epochs)")

    lstm_models[target_name] = model
    lstm_training_curves[target_name] = (train_losses, val_losses)

    pred_train_z, actual_train_z = predict_dataset(model, DataLoader(train_ds, batch_size=256, shuffle=False), device)
    pred_val_z, actual_val_z = predict_dataset(model, DataLoader(val_ds, batch_size=256, shuffle=False), device)
    pred_test_z, actual_test_z = predict_dataset(model, DataLoader(test_ds, batch_size=256, shuffle=False), device)

    train_dates = dates[train_idx][SEQ_LEN:]
    val_dates = dates[val_idx][SEQ_LEN:]
    test_dates = dates[test_idx][SEQ_LEN:]

    lstm_predictions[target_name] = {
        "train_z": pd.Series(pred_train_z, index=train_dates[:len(pred_train_z)]),
        "val_z":   pd.Series(pred_val_z, index=val_dates[:len(pred_val_z)]),
        "test_z":  pd.Series(pred_test_z, index=test_dates[:len(pred_test_z)]),
        "actual_train_z": actual_train_z,
        "actual_val_z":   actual_val_z,
        "actual_test_z":  actual_test_z,
        "raw_train": y_all_raw[train_idx][SEQ_LEN:],
        "raw_val":   y_all_raw[val_idx][SEQ_LEN:],
        "raw_test":  y_all_raw[test_idx][SEQ_LEN:],
    }

# ======================================================================
# 7. MODEL EVALUATION -- LR vs LSTM COMPARISON
# ======================================================================
print("\n" + "=" * 70)
print("  STEP 7: Model Evaluation -- LR vs LSTM Comparison")
print("=" * 70)

lstm_eval_metrics = {}

for target_name in HORIZONS.keys():
    print(f"\n--- {target_name} ---")
    preds = lstm_predictions[target_name]

    print("  [LSTM - Z-score domain]")
    lstm_eval_metrics[target_name] = {}
    lstm_eval_metrics[target_name]["train_z"] = eval_metrics(
        preds["train_z"].values, preds["actual_train_z"], "Train(z)")
    lstm_eval_metrics[target_name]["val_z"] = eval_metrics(
        preds["val_z"].values, preds["actual_val_z"], "Val(z)")
    lstm_eval_metrics[target_name]["test_z"] = eval_metrics(
        preds["test_z"].values, preds["actual_test_z"], "Test(z)")

    print("  [LSTM - Direction vs raw returns]")
    for label, prd, raw_act in [("Train", preds["train_z"].values, preds["raw_train"]),
                                 ("Val", preds["val_z"].values, preds["raw_val"]),
                                 ("Test", preds["test_z"].values, preds["raw_test"])]:
        n = min(len(prd), len(raw_act))
        da = np.mean(np.sign(prd[:n]) == np.sign(raw_act[:n])) * 100
        ic_raw, _ = stats.spearmanr(prd[:n], raw_act[:n])
        print(f"  {label:<16s}  DirAcc(raw)={da:.1f}%  IC(raw)={ic_raw:.3f}")

# Comparison table
print("\n" + "=" * 50)
print("  LR vs LSTM Comparison (Test Set)")
print("=" * 50)
print(f"  {'Horizon':<12} {'Model':<6} {'MSE':>10} {'R2':>8} {'DirAcc':>8} {'IC':>8}")
print(f"  {'-'*52}")

for target_name in HORIZONS.keys():
    lr_m = lr_metrics[target_name]["test"]
    print(f"  {target_name:<12} {'LR':<6} {lr_m['mse']:>10.6f} {lr_m['r2']:>8.4f} {lr_m['dir_acc']:>7.1f}% {lr_m['ic']:>8.3f}")
    lstm_m = lstm_eval_metrics[target_name]["test_z"]
    print(f"  {'':<12} {'LSTM':<6} {lstm_m['mse']:>10.6f} {lstm_m['r2']:>8.4f} {lstm_m['dir_acc']:>7.1f}% {lstm_m['ic']:>8.3f}")

# ======================================================================
# 8. TRADING STRATEGIES (per horizon + combined)
# ======================================================================
print("\n" + "=" * 70)
print("  STEP 8: Trading Strategies (per-horizon LSTM + combined)")
print("=" * 70)

vix_mult = pd.Series(np.where(vix_smooth > 35, 0.30,
                     np.where(vix_smooth > 25, 0.60, 1.00)), index=close.index)

def apply_filters_sig(pos, direction):
    dm_f = pd.Series(np.where((direction > 0) & (dxy_mom > 0.04), 0.65, 1.00), index=close.index)
    return pos * vix_mult * dm_f

def vol_target_sig(direction, shift=True):
    pos = (direction * (TARGET_VOL / ewm_vol)).clip(-3, 3)
    pos = apply_filters_sig(pos, direction)
    if shift:
        pos = pos.shift(1).fillna(0)
    return pos

# --- Per-horizon LSTM positions ---
lstm_positions = {}

for target_name in HORIZONS.keys():
    preds = lstm_predictions[target_name]
    full_pred = pd.concat([preds["train_z"], preds["val_z"], preds["test_z"]]).sort_index()
    full_pred = full_pred.reindex(close.index)

    pred_std = full_pred.dropna().std()
    lstm_raw_pos = np.tanh(full_pred / (pred_std + 1e-10))

    lstm_pos = (lstm_raw_pos * (TARGET_VOL / ewm_vol)).clip(-3, 3)
    dm = pd.Series(np.where((lstm_raw_pos > 0) & (dxy_mom > 0.04), 0.65, 1.00), index=close.index)
    lstm_pos = lstm_pos * vix_mult * dm
    lstm_pos = lstm_pos.shift(1).fillna(0)

    lstm_positions[target_name] = lstm_pos
    print(f"  {target_name} LSTM position: mean={lstm_pos.mean():.3f}, std={lstm_pos.std():.3f}")

# Combined LSTM: average of all 3 horizon signals
lstm_combined_raw = sum(lstm_positions.values()) / len(lstm_positions)
lstm_positions["combined_3h"] = lstm_combined_raw
print(f"  Combined 3-horizon LSTM: mean={lstm_combined_raw.mean():.3f}, std={lstm_combined_raw.std():.3f}")

# --- Donchian signals ---
pos_s0 = vol_target_sig(raw_pos, shift=False)

slope_sign = np.sign(kf_slope)
kf_confirm = pd.Series(
    np.where((raw_pos > 0) & (slope_sign > 0), raw_pos,
    np.where((raw_pos < 0) & (slope_sign < 0), raw_pos,
    np.where(raw_pos == 0, 0, raw_pos * 0.3))), index=close.index)
pos_s1 = vol_target_sig(kf_confirm, shift=False)

regime_mult = pd.Series(
    np.where(p_trend > 0.6, 1.0, np.where(p_trend > 0.4, 0.7, 0.35)), index=close.index)
pos_s2 = vol_target_sig(raw_pos * regime_mult, shift=False)

pos_s3 = vol_target_sig(raw3, shift=False)

close_clean = close.dropna()
n_entry_adapt = (p_trend * 20 + (1 - p_trend) * 50).clip(15, 50).astype(int)
channel_widths = [(15, 5), (20, 7), (25, 8), (30, 10), (35, 12), (40, 13), (45, 15), (50, 17)]
channel_positions = {(ne, nx): donchian_stateful(close_clean, ne, nx).reindex(pq.index).fillna(0)
                     for ne, nx in channel_widths}
raw4 = pd.Series(0.0, index=close.index)
for i in range(len(close)):
    ne = int(n_entry_adapt.iloc[i])
    best = min(channel_widths, key=lambda w: abs(w[0] - ne))
    raw4.iloc[i] = channel_positions[best].iloc[i]
pos_s4 = vol_target_sig(raw4, shift=False)

pos_s5 = pos_s0  # Cross-asset (simplified)

all_donchian = {
    "S0_Donchian": pos_s0, "S1_KF_Donchian": pos_s1, "S2_HMM_Donchian": pos_s2,
    "S3_DualTF": pos_s3, "S4_AdaptChannel": pos_s4, "S5_CrossAsset": pos_s5,
}
pos_master = sum(all_donchian.values()) / len(all_donchian)

best_horizon = "log_ret20"

strategies = {
    "S0_Donchian": pos_s0,
    "S3_DualTF": pos_s3,
    "Master_6sig": pos_master,
}

for target_name in HORIZONS.keys():
    strategies[f"LSTM_{target_name}"] = lstm_positions[target_name]

strategies["LSTM_Combined_3h"] = lstm_positions["combined_3h"]
strategies["S3+LSTM_best"] = 0.5 * pos_s3 + 0.5 * lstm_positions[best_horizon]
strategies["Master+LSTM_7sig"] = (sum(all_donchian.values()) + lstm_positions[best_horizon]) / 7
strategies["Master+LSTM_comb"] = (sum(all_donchian.values()) + lstm_positions["combined_3h"]) / 7

# Alpha strategies
for feat in top5_alphas[:3]:
    strategies[f"Alpha_{feat[:20]}"] = alpha_positions[feat]
if len(top5_alphas) > 0:
    strategies["Alpha_Composite"] = alpha_composite_pos
    strategies["Alpha+LSTM"] = 0.5 * alpha_composite_pos + 0.5 * lstm_positions[best_horizon]
    strategies["Master+Alpha+LSTM"] = (
        sum(all_donchian.values()) + alpha_composite_pos + lstm_positions[best_horizon]) / 8

print(f"\nStrategies: {list(strategies.keys())}")

# ======================================================================
# 9. RESULTS & PLOTS
# ======================================================================
print("\n" + "=" * 70)
print("  STEP 9: Results & Plots")
print("=" * 70)

def compute_strategy_metrics(pos, ret_series, period_mask_fn, label):
    r = (pos * ret_series).dropna()
    r = r[period_mask_fn(r.index)]
    if len(r) < 21 or r.std() == 0:
        return dict(sharpe=0, ann_ret=0, ann_vol=0, max_dd=0, hit=0,
                    cum=pd.Series(dtype=float), dd=pd.Series(dtype=float), daily_ret=pd.Series(dtype=float))
    ar = r.mean() * 252; av = r.std() * np.sqrt(252)
    sh = ar / av if av > 0 else 0
    cum = (1 + r).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    md = dd.min(); hit = (r > 0).mean() * 100
    return dict(sharpe=sh, ann_ret=ar, ann_vol=av, max_dd=md, hit=hit, cum=cum, dd=dd, daily_ret=r)

# --- Table 1: LR vs LSTM ---
print("\n--- Table 1: LR vs LSTM Metrics (Test Set) ---")
print(f"  {'Horizon':<12} {'Model':<6} {'MSE':>10} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'DirAcc':>8} {'IC':>8}")
print(f"  {'-'*62}")
for target_name in HORIZONS.keys():
    lr_m = lr_metrics[target_name]["test"]
    lstm_m = lstm_eval_metrics[target_name]["test_z"]
    print(f"  {target_name:<12} {'LR':<6} {lr_m['mse']:>10.6f} {lr_m['rmse']:>8.4f} {lr_m['mae']:>8.4f} {lr_m['r2']:>8.4f} {lr_m['dir_acc']:>7.1f}% {lr_m['ic']:>8.3f}")
    print(f"  {'':<12} {'LSTM':<6} {lstm_m['mse']:>10.6f} {lstm_m['rmse']:>8.4f} {lstm_m['mae']:>8.4f} {lstm_m['r2']:>8.4f} {lstm_m['dir_acc']:>7.1f}% {lstm_m['ic']:>8.3f}")

# --- Table 2: Strategy performance ---
print(f"\n--- Table 2: Strategy Performance ---")
print(f"  {'Strategy':<25} {'Train Sh':>10} {'Test Sh':>10} {'Test Ret':>10} {'Test DD':>10} {'Test Hit':>10}")
print(f"  {'-'*75}")

results_all = {}
train_period = lambda idx: idx < SPLIT_TEST
test_period = lambda idx: idx >= SPLIT_TEST

for name, pos in strategies.items():
    m_tr = compute_strategy_metrics(pos, ret, train_period, name)
    m_te = compute_strategy_metrics(pos, ret, test_period, name)
    results_all[name] = {"train": m_tr, "test": m_te}
    print(f"  {name:<23} {m_tr['sharpe']:>+10.2f} {m_te['sharpe']:>+10.2f} "
          f"{m_te['ann_ret']*100:>+9.1f}% {m_te['max_dd']*100:>+9.1f}% {m_te['hit']:>9.1f}%")

bh_tr = compute_strategy_metrics(pd.Series(1.0, index=close.index), ret, train_period, "BuyHold")
bh_te = compute_strategy_metrics(pd.Series(1.0, index=close.index), ret, test_period, "BuyHold")
print(f"  {'BuyHold':<23} {bh_tr['sharpe']:>+10.2f} {bh_te['sharpe']:>+10.2f} "
      f"{bh_te['ann_ret']*100:>+9.1f}% {bh_te['max_dd']*100:>+9.1f}% {bh_te['hit']:>9.1f}%")

# --- Annual Sharpe ---
key_strats_annual = ["S3_DualTF", "LSTM_log_ret20", "LSTM_Combined_3h", "S3+LSTM_best",
                     "Master+LSTM_7sig", "Alpha_Composite", "Master+Alpha+LSTM"]
key_strats_annual = [k for k in key_strats_annual if k in strategies]
print("\n--- Annual Sharpe (Test) ---")
print(f"  {'Year':>4}", end="")
for name in key_strats_annual:
    print(f"  {name[:18]:>18}", end="")
print()

for yr in range(2021, 2026):
    print(f"  {yr:>4}", end="")
    for name in key_strats_annual:
        pos = strategies[name]
        r = (pos * ret).dropna()
        yr_r = r[(r.index.year == yr) & (r.index >= SPLIT_TEST)]
        if len(yr_r) > 20 and yr_r.std() > 0:
            ysh = (yr_r.mean() * 252) / (yr_r.std() * np.sqrt(252))
        else:
            ysh = 0
        print(f"  {ysh:>+18.2f}", end="")
    print()

# --- Signal correlations ---
print("\n--- Signal Correlation (Test, daily returns) ---")
sig_returns = {}
sig_list_corr = ["S0_Donchian", "S3_DualTF", "Master_6sig",
                 "LSTM_log_ret20", "LSTM_log_ret30", "LSTM_log_ret60", "LSTM_Combined_3h",
                 "Alpha_Composite", "Alpha+LSTM", "Master+Alpha+LSTM"]
sig_list_corr = [s for s in sig_list_corr if s in strategies]
for name in sig_list_corr:
    if name in strategies:
        r = (strategies[name] * ret).dropna()
        sig_returns[name] = r[r.index >= SPLIT_TEST]
corr_df = pd.DataFrame(sig_returns).dropna().corr()
print(corr_df.round(3).to_string())

# ======================================================================
# MASTER FIGURE
# ======================================================================
print("\nGenerating plots...")

fig = plt.figure(figsize=(24, 37))
fig.suptitle(
    "Copper Multi-Horizon Prediction: LR Baseline + LSTM + Alpha Signals\n"
    f"Horizons: {list(HORIZONS.keys())}  |  LSTM: {N_LAYERS}x{HIDDEN}, seq={SEQ_LEN}  |  "
    f"Train 2005-2018 | Val 2019-2020 | Test 2021-2025",
    fontsize=13, fontweight="bold", y=0.995)

gs = gridspec.GridSpec(7, 3, figure=fig, hspace=0.45, wspace=0.30)

colors_strat = {
    "S0_Donchian": "steelblue", "S3_DualTF": "crimson", "Master_6sig": "grey",
    "LSTM_log_ret20": "darkorange", "LSTM_log_ret30": "forestgreen", "LSTM_log_ret60": "purple",
    "LSTM_Combined_3h": "gold", "S3+LSTM_best": "deeppink", "Master+LSTM_7sig": "black",
    "Master+LSTM_comb": "teal",
    "Alpha_Composite": "limegreen", "Alpha+LSTM": "magenta", "Master+Alpha+LSTM": "darkblue",
}
# Add colors for individual alpha signals
for _i, _f in enumerate(top5_alphas[:3]):
    colors_strat[f"Alpha_{_f[:20]}"] = ["#2ca02c", "#d62728", "#9467bd"][_i % 3]

# Row 0: Training curves (one per horizon)
for j, target_name in enumerate(HORIZONS.keys()):
    ax = fig.add_subplot(gs[0, j])
    tl, vl = lstm_training_curves[target_name]
    ax.plot(tl, label="Train", color="steelblue", lw=1.2)
    ax.plot(vl, label="Val", color="darkorange", lw=1.2)
    ax.set_title(f"LSTM Training: {target_name}", fontweight="bold", fontsize=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.legend(fontsize=8); ax.set_yscale("log")

# Row 1: LR vs LSTM scatter (test set)
for j, target_name in enumerate(HORIZONS.keys()):
    ax = fig.add_subplot(gs[1, j])
    lr_pred_te = lr_predictions[target_name]["test"].values
    lr_act_te = pq.loc[lr_predictions[target_name]["test"].index, target_name].values
    ax.scatter(lr_act_te, lr_pred_te, alpha=0.2, s=4, color="steelblue", label="LR")
    lstm_pred_te = lstm_predictions[target_name]["test_z"].values
    lstm_act_te = lstm_predictions[target_name]["actual_test_z"]
    ax.scatter(lstm_act_te, lstm_pred_te, alpha=0.2, s=4, color="darkorange", label="LSTM(z)")
    all_vals = np.concatenate([lr_act_te, lstm_act_te])
    lo, hi = np.nanpercentile(all_vals, [1, 99])
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    lr_r2 = lr_metrics[target_name]["test"]["r2"]
    lstm_r2 = lstm_eval_metrics[target_name]["test_z"]["r2"]
    ax.set_title(f"{target_name} Pred vs Act\nLR R2={lr_r2:.4f} | LSTM R2={lstm_r2:.4f}",
                 fontweight="bold", fontsize=9)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.legend(fontsize=7, markerscale=3)

# Row 2: Cumulative returns (train+val, test, per-horizon)
plot_strats = ["S3_DualTF", "Master_6sig", "LSTM_log_ret20", "LSTM_Combined_3h",
               "S3+LSTM_best", "Master+LSTM_7sig", "Alpha_Composite", "Master+Alpha+LSTM"]
plot_strats = [s for s in plot_strats if s in strategies]
for col, (period, prd_name) in enumerate([(train_period, "Train+Val"), (test_period, "Test")]):
    ax = fig.add_subplot(gs[2, col])
    prd_key = "train" if col == 0 else "test"
    for name in plot_strats:
        if name not in results_all:
            continue
        m = results_all[name][prd_key]
        if len(m.get("cum", [])) > 0:
            (m["cum"] - 1).plot(ax=ax, color=colors_strat.get(name, "black"),
                                lw=2 if "LSTM" in name else 1, label=f"{name} ({m['sharpe']:.2f})")
    ax.set_title(f"{prd_name}: Cumulative Return", fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.legend(fontsize=6, loc="upper left"); ax.set_xlabel("")

ax = fig.add_subplot(gs[2, 2])
for target_name in HORIZONS.keys():
    sname = f"LSTM_{target_name}"
    m = results_all[sname]["test"]
    if len(m.get("cum", [])) > 0:
        (m["cum"] - 1).plot(ax=ax, color=colors_strat.get(sname, "black"),
                            lw=1.5, label=f"{sname} ({m['sharpe']:.2f})")
ax.set_title("Per-Horizon LSTM (Test)", fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
ax.legend(fontsize=7); ax.set_xlabel("")

# Row 3: Drawdowns + predictions vs actual + signal correlations
ax = fig.add_subplot(gs[3, 0])
for name in ["Master_6sig", "S3+LSTM_best", "Master+LSTM_7sig", "LSTM_Combined_3h",
             "Alpha_Composite", "Master+Alpha+LSTM"]:
    if name not in results_all:
        continue
    m = results_all[name]["test"]
    if len(m.get("dd", [])) > 0:
        m["dd"].plot(ax=ax, color=colors_strat.get(name, "black"), lw=1.2, label=name)
ax.set_title("Test: Drawdowns", fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
ax.legend(fontsize=7); ax.set_xlabel("")

ax = fig.add_subplot(gs[3, 1])
preds_best = lstm_predictions[best_horizon]
test_actual_s = pd.Series(preds_best["actual_test_z"],
                           index=preds_best["test_z"].index[:len(preds_best["actual_test_z"])])
test_actual_s.rolling(20).mean().plot(ax=ax, color="steelblue", lw=1, label="Actual (20d MA)")
preds_best["test_z"].rolling(20).mean().plot(ax=ax, color="darkorange", lw=1, label="Predicted (20d MA)")
ax.axhline(0, color="grey", lw=0.5, ls="--")
ax.set_title(f"LSTM {best_horizon} Pred vs Act (Test, 20d MA)", fontweight="bold")
ax.legend(fontsize=8); ax.set_xlabel("")

ax = fig.add_subplot(gs[3, 2])
corr_display = corr_df
im = ax.imshow(corr_display.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr_display)))
ax.set_yticks(range(len(corr_display)))
ax.set_xticklabels([s[:15] for s in corr_display.columns], fontsize=6, rotation=45, ha="right")
ax.set_yticklabels([s[:15] for s in corr_display.index], fontsize=6)
for i in range(len(corr_display)):
    for j in range(len(corr_display)):
        ax.text(j, i, f"{corr_display.values[i,j]:.2f}", ha="center", va="center", fontsize=5)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Signal Return Correlations (Test)", fontweight="bold", fontsize=9)

# Row 4: Annual Sharpe bars + feature importance
ax = fig.add_subplot(gs[4, 0])
bar_strats = ["S3_DualTF", "LSTM_log_ret20", "LSTM_Combined_3h", "S3+LSTM_best",
              "Master+LSTM_7sig", "Alpha_Composite", "Master+Alpha+LSTM"]
bar_strats = [s for s in bar_strats if s in strategies]
years = list(range(2021, 2026))
bar_w = 0.8 / max(len(bar_strats), 1)
for j, name in enumerate(bar_strats):
    sharpes_yr = []
    for yr in years:
        pos = strategies[name]
        r = (pos * ret).dropna()
        yr_r = r[(r.index.year == yr) & (r.index >= SPLIT_TEST)]
        if len(yr_r) > 20 and yr_r.std() > 0:
            sharpes_yr.append((yr_r.mean() * 252) / (yr_r.std() * np.sqrt(252)))
        else:
            sharpes_yr.append(0)
    x_pos = np.arange(len(years)) + j * bar_w
    ax.bar(x_pos, sharpes_yr, width=bar_w, label=name[:18], color=colors_strat.get(name, "grey"))
ax.set_xticks(np.arange(len(years)) + len(bar_strats) * bar_w / 2)
ax.set_xticklabels(years)
ax.axhline(0, color="black", lw=0.8); ax.axhline(2, color="gold", ls="--", lw=1)
ax.set_title("Annual Sharpe (Test)", fontweight="bold"); ax.legend(fontsize=5); ax.set_xlabel("")

for j, target_name in enumerate(HORIZONS.keys()):
    ax = fig.add_subplot(gs[4, j + 1]) if j < 2 else fig.add_subplot(gs[6, 0])
    top_corrs = selected_features[target_name].head(15)
    if len(top_corrs) > 0:
        colors_bar = ["darkgreen" if v > 0 else "crimson" for v in top_corrs.values]
        ax.barh(range(len(top_corrs)), top_corrs.values, color=colors_bar, edgecolor="white")
        ax.set_yticks(range(len(top_corrs)))
        ax.set_yticklabels(top_corrs.index, fontsize=5)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"Top Features -> {target_name}", fontweight="bold", fontsize=9)
    ax.set_xlabel("Pearson Corr (Train)")

# Row 5: Alpha Signal Analysis
# Panel 5,0: Alpha IC/IR bar chart
ax = fig.add_subplot(gs[5, 0])
# Gather top alphas by |IR| across all horizons
alpha_ir_display = {}
for tn in HORIZONS:
    for feat, m in alpha_signal_metrics[tn].items():
        key = feat
        if key not in alpha_ir_display or abs(m['ir']) > abs(alpha_ir_display[key]['ir']):
            alpha_ir_display[key] = {'ic': m['ic'], 'ir': m['ir'], 'horizon': tn}
alpha_ir_sorted = sorted(alpha_ir_display.items(), key=lambda x: abs(x[1]['ir']), reverse=True)[:20]
if len(alpha_ir_sorted) > 0:
    a_names = [f[:25] for f, _ in alpha_ir_sorted]
    a_ics = [m['ic'] for _, m in alpha_ir_sorted]
    a_irs = [m['ir'] for _, m in alpha_ir_sorted]
    y_pos = np.arange(len(a_names))
    ax.barh(y_pos - 0.15, a_ics, height=0.3, color="steelblue", label="IC")
    ax.barh(y_pos + 0.15, a_irs, height=0.3, color="darkorange", label="IR")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(a_names, fontsize=5)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.5)
    ax.legend(fontsize=7)
ax.set_title("Top 20 Alpha Signals: IC & IR", fontweight="bold", fontsize=9)

# Panel 5,1: Alpha strategy cumulative returns (test)
ax = fig.add_subplot(gs[5, 1])
alpha_strat_names = ["Alpha_Composite", "Alpha+LSTM", "Master+Alpha+LSTM"]
alpha_strat_names = [s for s in alpha_strat_names if s in results_all]
# Also add top individual alpha signals
for feat in top5_alphas[:3]:
    sname = f"Alpha_{feat[:20]}"
    if sname in results_all:
        alpha_strat_names.append(sname)
for name in alpha_strat_names:
    m = results_all[name]["test"]
    if len(m.get("cum", [])) > 0:
        (m["cum"] - 1).plot(ax=ax, color=colors_strat.get(name, "grey"),
                            lw=2 if "Composite" in name else 1.2,
                            label=f"{name[:20]} ({m['sharpe']:.2f})")
# Add S3_DualTF and BuyHold as baselines
for bname in ["S3_DualTF"]:
    if bname in results_all:
        m = results_all[bname]["test"]
        if len(m.get("cum", [])) > 0:
            (m["cum"] - 1).plot(ax=ax, color=colors_strat.get(bname, "grey"),
                                lw=1, ls="--", label=f"{bname} ({m['sharpe']:.2f})")
ax.set_title("Alpha Strategies (Test)", fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
ax.legend(fontsize=5, loc="upper left"); ax.set_xlabel("")

# Panel 5,2: Alpha summary text
ax = fig.add_subplot(gs[5, 2])
ax.axis("off")
alpha_text = f"Alpha Signal Summary\n{'='*30}"
alpha_text += f"\nSelected: {len(all_selected_alphas)} signals"
alpha_text += f"\nComposite: top {len(top5_alphas)} by |IR|"
for f in top5_alphas:
    best_h = max(HORIZONS, key=lambda tn: abs(alpha_signal_metrics[tn].get(f, {}).get('ir', 0)))
    m = alpha_signal_metrics[best_h].get(f, {})
    alpha_text += f"\n  {f[:22]:22s} IR={m.get('ir',0):+.3f}"
alpha_text += f"\n\nAlpha Strategy Sharpe (Test):"
for name in ["Alpha_Composite", "Alpha+LSTM", "Master+Alpha+LSTM"]:
    if name in results_all:
        m = results_all[name]["test"]
        alpha_text += f"\n  {name:22s}: {m['sharpe']:+.2f}"
ax.text(0.05, 0.95, alpha_text, transform=ax.transAxes,
        fontsize=7, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="honeydew", alpha=0.8))

# Row 6: Model + strategy summaries
ax = fig.add_subplot(gs[6, 1])
ax.axis("off")
summary_text = "LR vs LSTM Model Summary\n" + "=" * 35
for target_name in HORIZONS.keys():
    lr_m = lr_metrics[target_name]["test"]
    lstm_m = lstm_eval_metrics[target_name]["test_z"]
    n_feats = len(lr_feature_sets[target_name])
    n_epochs = len(lstm_training_curves[target_name][0])
    summary_text += (
        f"\n\n{target_name} ({n_feats} features, {n_epochs} epochs):\n"
        f"  LR:   R2={lr_m['r2']:.4f}  DA={lr_m['dir_acc']:.1f}%  IC={lr_m['ic']:.3f}\n"
        f"  LSTM: R2={lstm_m['r2']:.4f}  DA={lstm_m['dir_acc']:.1f}%  IC={lstm_m['ic']:.3f}"
    )
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=8, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

ax = fig.add_subplot(gs[6, 2])
ax.axis("off")
strat_text = "Strategy Sharpe (Test)\n" + "=" * 30
for name in ["S3_DualTF", "LSTM_log_ret20", "LSTM_log_ret30", "LSTM_log_ret60",
             "LSTM_Combined_3h", "S3+LSTM_best", "Master+LSTM_7sig", "Master+LSTM_comb",
             "Alpha_Composite", "Alpha+LSTM", "Master+Alpha+LSTM"]:
    if name in results_all:
        m = results_all[name]["test"]
        strat_text += f"\n  {name:22s}: {m['sharpe']:+.2f}"
strat_text += f"\n\n  {'BuyHold':22s}: {bh_te['sharpe']:+.2f}"
strat_text += f"\n\nBest horizon: {best_horizon}"
strat_text += f"\nArchitecture: {N_LAYERS}L LSTM, {HIDDEN}h"
strat_text += f"\nSeq len: {SEQ_LEN}, Dropout: {DROPOUT}"
strat_text += f"\nAlpha signals: {len(all_selected_alphas)}"
ax.text(0.05, 0.95, strat_text, transform=ax.transAxes,
        fontsize=7, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8))

out_path = f"{BASE}/claude/model_results.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved: {out_path}")
print(f"Distribution analysis saved: {BASE}/claude/dist_analysis.png")
print(f"Correlation heatmap saved: {BASE}/claude/correlation_heatmap.png")

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)
