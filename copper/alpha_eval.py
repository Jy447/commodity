import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats
from config import HORIZONS, ZSCORE_WINDOW, TARGET_VOL, SPLIT_TEST


@dataclass
class AlphaEvalResults:
    alpha_signal_metrics: dict
    all_selected_alphas: list
    alpha_positions: dict
    alpha_composite_pos: pd.Series
    top5_alphas: list


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


def run_alpha_evaluation(pq, feature_cols, corr_results, train_for_corr,
                          close, ewm_vol, vix_smooth, dxy_mom, ret):
    print("\n" + "=" * 70)
    print("  STEP 4.5: Alpha Signal Evaluation")
    print("=" * 70)

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

    return AlphaEvalResults(
        alpha_signal_metrics=alpha_signal_metrics,
        all_selected_alphas=all_selected_alphas,
        alpha_positions=alpha_positions,
        alpha_composite_pos=alpha_composite_pos,
        top5_alphas=top5_alphas)
