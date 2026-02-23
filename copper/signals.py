import numpy as np
import pandas as pd


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

def fit_hmm_regimes(returns_series, split_date, n_regimes=2):
    from hmmlearn.hmm import GaussianHMM
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
