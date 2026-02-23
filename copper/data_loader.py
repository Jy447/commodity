import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from config import (BASE, COPPER, ALL_HORIZONS, HORIZONS, ZSCORE_WINDOW,
                    SPLIT_VAL, SPLIT_TEST, USE_PARQUET)
from signals import kalman_trend, donchian_stateful, fit_hmm_regimes
from alpha_lib import AlphaLib


@dataclass
class PipelineData:
    pq: pd.DataFrame
    close: pd.Series
    ret: pd.Series
    ewm_vol: pd.Series
    vix_smooth: pd.Series
    dxy_mom: pd.Series
    kf_slope: pd.Series
    p_trend: pd.Series
    raw_pos: pd.Series
    raw3: pd.Series
    train_mask: pd.Series
    val_mask: pd.Series
    test_mask: pd.Series


def load_data() -> PipelineData:
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

    return PipelineData(
        pq=pq, close=close, ret=ret, ewm_vol=ewm_vol,
        vix_smooth=vix_smooth, dxy_mom=dxy_mom, kf_slope=kf_slope,
        p_trend=p_trend, raw_pos=raw_pos, raw3=raw3,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
