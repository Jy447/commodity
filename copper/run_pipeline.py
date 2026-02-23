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

import config  # noqa: F401 -- side effects: matplotlib backend, warnings, seeds
from data_loader import load_data
from analysis import run_distribution_analysis, run_correlation_analysis
from alpha_eval import run_alpha_evaluation
from models import train_linear_regression, train_lstm, print_model_comparison
from strategies import build_strategies
from results import generate_results


def main():
    # Step 1
    data = load_data()

    # Step 2
    run_distribution_analysis(data.pq)

    # Steps 3-4
    analysis = run_correlation_analysis(data.pq, data.train_mask)

    # Step 4.5
    alpha = run_alpha_evaluation(
        data.pq, analysis.feature_cols, analysis.corr_results,
        analysis.train_for_corr, data.close, data.ewm_vol,
        data.vix_smooth, data.dxy_mom, data.ret)

    # Step 5
    lr = train_linear_regression(data.pq, analysis.selected_features)

    # Steps 6-7
    lstm = train_lstm(data.pq, lr.lr_feature_sets)

    # Step 7 comparison table
    print_model_comparison(lr.lr_metrics, lstm.lstm_eval_metrics)

    # Step 8
    strats = build_strategies(
        data.close, data.ret, data.ewm_vol, data.vix_smooth,
        data.dxy_mom, data.kf_slope, data.p_trend,
        data.raw_pos, data.raw3,
        lstm.lstm_predictions,
        alpha.alpha_positions, alpha.alpha_composite_pos,
        alpha.top5_alphas)

    # Step 9
    generate_results(strats, data, analysis, alpha, lr, lstm)


if __name__ == "__main__":
    main()
