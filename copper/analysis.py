import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.gofplots import qqplot
from config import BASE, ALL_HORIZONS, HORIZONS, CORR_THRESH, MISSING_THRESH


@dataclass
class AnalysisResults:
    feature_cols: list
    selected_features: dict
    corr_results: dict
    train_for_corr: pd.DataFrame


def run_distribution_analysis(pq):
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


def run_correlation_analysis(pq, train_mask):
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

    return AnalysisResults(
        feature_cols=feature_cols,
        selected_features=selected_features,
        corr_results=corr_results,
        train_for_corr=train_for_corr)
