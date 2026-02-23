import numpy as np
import pandas as pd
from config import COPPER
from utils import (sma, ts_sum, ts_min, ts_median, ts_max, delay, rank,
                   rolling_volatility, rolling_skewness, rolling_kurtosis,
                   rolling_entropy, hurst_exponent, rolling_hurst_one_ticker,
                   rolling_beta, rolling_corr, compute_ATR)


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
