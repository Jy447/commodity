import numpy as np
import pandas as pd
from dataclasses import dataclass
from config import HORIZONS, TARGET_VOL
from signals import donchian_stateful


@dataclass
class StrategyResults:
    strategies: dict
    all_donchian: dict
    lstm_positions: dict
    best_horizon: str


def build_strategies(close, ret, ewm_vol, vix_smooth, dxy_mom, kf_slope,
                      p_trend, raw_pos, raw3,
                      lstm_predictions, alpha_positions, alpha_composite_pos,
                      top5_alphas):
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
    channel_positions = {(ne, nx): donchian_stateful(close_clean, ne, nx).reindex(close.index).fillna(0)
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

    return StrategyResults(
        strategies=strategies,
        all_donchian=all_donchian,
        lstm_positions=lstm_positions,
        best_horizon=best_horizon)
