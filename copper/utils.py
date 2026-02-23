import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tools.tools import add_constant


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
