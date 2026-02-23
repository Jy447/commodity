import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import (BASE, HORIZONS, SPLIT_TEST, N_LAYERS, HIDDEN, SEQ_LEN, DROPOUT)


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


def generate_results(strats, data, analysis, alpha, lr, lstm):
    """
    strats: StrategyResults
    data: PipelineData
    analysis: AnalysisResults
    alpha: AlphaEvalResults
    lr: LRResults
    lstm: LSTMResults
    """
    print("\n" + "=" * 70)
    print("  STEP 9: Results & Plots")
    print("=" * 70)

    strategies = strats.strategies
    best_horizon = strats.best_horizon
    ret = data.ret
    close = data.close
    pq = data.pq
    lr_metrics = lr.lr_metrics
    lstm_eval_metrics = lstm.lstm_eval_metrics
    lr_predictions = lr.lr_predictions
    lstm_predictions = lstm.lstm_predictions
    lstm_training_curves = lstm.lstm_training_curves
    selected_features = analysis.selected_features
    alpha_signal_metrics = alpha.alpha_signal_metrics
    all_selected_alphas = alpha.all_selected_alphas
    top5_alphas = alpha.top5_alphas
    lr_feature_sets = lr.lr_feature_sets

    train_period = lambda idx: idx < SPLIT_TEST
    test_period = lambda idx: idx >= SPLIT_TEST

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
    # Add S3_DualTF as baseline
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
