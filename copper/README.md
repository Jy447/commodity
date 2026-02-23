================================================================================
  COPPER ALPHA SIGNALS & LSTM MODEL PIPELINE
================================================================================

  Files:
    1. features.py    - Volatility-based signals for CL=F (26 signals + composites)
    2. copper.ipynb    - Exploratory analysis notebook
    3. test.py         - 6-signal Donchian ensemble with KF + HMM (Sharpe 2.38 OOS)
    4. model.py        - LSTM model with AlphaLib features + strategy (monolithic)
    requirements.txt   - Python dependencies

  Modular Pipeline (split from 4. model.py):
    config.py          - Constants, hyperparameters, paths, random seeds
    utils.py           - Rolling/stat helpers (sma, ts_sum, rolling_vol, ATR, etc.)
    alpha_lib.py       - AlphaLib class (~100 cross-sectional alpha features)
    signals.py         - Kalman filter, Donchian stateful, HMM regime detection
    data_loader.py     - Step 1: Load parquet/CSVs, prepare features -> PipelineData
    analysis.py        - Steps 2-4: Distribution analysis, correlations, feature selection
    alpha_eval.py      - Step 4.5: Alpha signal IC/IR evaluation, position building
    models.py          - Steps 5-7: Linear Regression baseline, LSTM training, evaluation
    strategies.py      - Step 8: Build all trading strategies (Donchian + LSTM + Alpha)
    results.py         - Step 9: Performance tables + 7x3 master figure
    run_pipeline.py    - Main orchestrator (~50 lines, calls all modules in sequence)

================================================================================
  USAGE
================================================================================

  Run the full pipeline:

    python -u copper/run_pipeline.py

  This executes Steps 1-9 in sequence and produces:
    - claude/model_results.png       (7x3 master figure)
    - claude/dist_analysis.png       (return distribution plots)
    - claude/correlation_heatmap.png  (feature-return correlation matrix)

  On first run with USE_PARQUET=False, CSVs are loaded and AlphaLib features
  are computed (takes several minutes). Results are cached to
  claude/copper_alpha_features.parquet. Subsequent runs with USE_PARQUET=True
  (default) load the parquet in seconds.

  To modify individual components:
    - Add new alpha features     -> alpha_lib.py (AlphaLib.calcu_alpha)
    - Add new signals            -> signals.py
    - Change model architecture  -> models.py (CopperLSTM class)
    - Add/remove strategies      -> strategies.py (build_strategies)
    - Change plots/tables        -> results.py (generate_results)
    - Tune hyperparameters       -> config.py

  Data flow (no globals, explicit dataclass inputs/outputs):

    load_data()                  -> PipelineData
    run_distribution_analysis()  -> (plots only)
    run_correlation_analysis()   -> AnalysisResults
    run_alpha_evaluation()       -> AlphaEvalResults
    train_linear_regression()    -> LRResults
    train_lstm()                 -> LSTMResults
    build_strategies()           -> StrategyResults
    generate_results()           -> (tables + plots)

================================================================================
  THE CORE THESIS
================================================================================

  Copper is a trend-following market. Mine supply cycles take 5-10 years,
  Chinese infrastructure programs play out over quarters, and EV demand ramps
  over years. These slow-moving fundamentals create persistent price trends.

  The Kalman filter and HMM are NOT used as standalone alpha generators --
  they are FILTERS and ADAPTERS that improve the core trend-following signal.
  The LSTM model adds a return-prediction layer that diversifies the ensemble.

================================================================================
  SHARED INFRASTRUCTURE
================================================================================

  Volatility Targeting (20% annual)
    Every signal scaled by 0.20 / realized_vol. Low vol -> higher leverage,
    high vol -> lower leverage. Ensures equal daily risk contribution.

  VIX Filter
    VIX > 25 -> 60% size. VIX > 35 -> 30% size. Captures "risk-off" regimes.

  DXY (Dollar) Filter
    USD rises >4% over 40 days AND signal is long -> reduce to 65%.
    Copper priced in USD, strong dollar = headwind for longs.

  Look-Ahead Prevention
    All Donchian channels .shift(1). Positions shifted 1 day. HMM/LSTM
    fitted on train data only. Feature correlations computed on train only.

================================================================================
  THE 6 DONCHIAN SIGNALS (from 3. test.py)
================================================================================

  S0: Donchian Breakout (Baseline) -- Test Sharpe ~2.0
    Entry: Price breaks 30-day high -> long. 30-day low -> short.
    Exit: 10-day reversal channel (tight stop).
    Stateful: holds position until exit triggers.

  S1: Kalman-Filtered Donchian -- Test Sharpe ~2.15
    Same entries, but 0.3x when Kalman slope disagrees.
    Local Linear Trend model on log(price), tiny slope noise (3e-7).

  S2: HMM Regime-Scaled Donchian -- Test Sharpe ~1.85
    2-state Gaussian HMM: "trending" (low var) vs "choppy" (high var).
    P(trending) > 0.6 -> 1.0x, 0.4-0.6 -> 0.7x, < 0.4 -> 0.35x.

  S3: Dual-Timeframe + Kalman Tiebreaker -- Test Sharpe ~2.5 (BEST)
    Fast channel (20d/7d) + Slow channel (60d/20d).
    Take fast breakout only when slow agrees OR Kalman slope confirms.

  S4: Adaptive Channel Breakout -- Test Sharpe ~2.6
    HMM regime probability adjusts channel width (20d-50d dynamically).
    8 pre-computed Donchian channels, nearest match selected each day.

  S5: Cross-Asset Trend Filter -- Test Sharpe ~1.8
    Donchian scaled by crude oil + SPY momentum confirmation.
    Both confirm: 1.0x | Neutral: 0.6x | Disagree: 0.3x.

  Master Ensemble: Equal-weight average of all 6 signals.

================================================================================
  ALPHA FEATURES (from 4. model.py AlphaLib)
================================================================================

  The AlphaLib class computes ~100+ cross-sectional features from raw OHLCV
  data (market_data + commodities + companies CSVs). Features are computed
  for all tickers cross-sectionally, then copper (HG=F) column extracted.

  --- Price-Volume Alphas (alpha01-alpha32) ---
  alpha01  : Close deviation from 10d SMA, weighted by dollar volume share
  alpha02  : 5d average dollar volume share
  alpha06  : 5d smoothed close-deviation * volume share
  alpha07  : 15d max of (close/15d SMA - 1) * volume
  alpha08  : 10d min of (close/60d max drawdown) * volume (crash signal)
  alpha09  : 50d max of (close/60d min recovery) * volume (recovery signal)
  alpha10  : 15d max of (close/20d min recovery) * volume
  alpha12  : 20d max of (30d high-low range) * volume
  alpha13  : 30d max range / 30d min (open+close) - range expansion vs price
  alpha14  : 5d sum of volume * (close-open) - directional volume
  alpha15  : 15d max of volume * (high-low) - max volatility * volume
  alpha16  : 5d sum of volume * (close-vwap) - volume-weighted price deviation
  alpha17  : 15d min of volume * (low-vwap) - extreme negative VWAP deviations
  alpha18  : 15d min of volume * (open-vwap) - opening price vs VWAP
  alpha19  : 15d min of volume * (open-low) - downside gap from open
  alpha20  : 10d max of volume * (close-low) - closing strength
  alpha21  : Median volume / sum volume over 15d - volume distribution
  alpha23  : 15d max volume / 10d min volume - volume range ratio
  alpha24  : 5d sum of volume share
  alpha28  : (5d max close / lag5 close) * (5d min close / close) - mean reversion
  alpha29  : 6d max/min of |close-open|/(high-low) - body/range ratio expansion
  alpha30  : Body/range ratio momentum (vs 4d ago)
  alpha31  : 4d max/min of |low-open|/(close-low) - lower shadow expansion
  alpha32  : 2d max/min of |high-open|/(close-low) - upper shadow expansion

  --- Momentum Alphas ---
  mom20    : 20-day price momentum (close/close_20 - 1)
  mom60    : 60-day price momentum
  mom120   : 120-day price momentum
  mom_12   : 12-month (252d) momentum
  sharpe_mom20 : 20d momentum / 20d volatility (risk-adjusted momentum)

  --- Reversal Alphas (01-11) ---
  01-04    : Negative lagged returns (1d, 5d, 10d, 20d) - short-term reversal
  05-07    : Price change ratios at 20d, 60d, 120d horizons
  08-11    : Moving average returns at 5d, 10d, 20d, 60d

  --- Volatility & Statistical Alphas ---
  vol_21, vol_63         : 21d and 63d realized volatility of returns
  vol_12                 : 252d realized volatility
  60                     : Volatility of volatility (rolling vol of 21d vol)
  skewness_21, skewness_63 : Rolling return skewness (tail asymmetry)
  kurtosis_21, kurtosis_63 : Rolling return kurtosis (tail fatness)
  entropy_21, entropy_63   : Rolling return entropy (randomness measure)
  vol_skew_proxy         : Skewness * volatility interaction

  --- Microstructure & Volume Alphas ---
  os_ratior              : Volume turnover ratio (vol / 21d avg vol)
  os_ratio_chg           : Turnover change vs 126d average
  abnormal_volume        : Volume / 63d average volume
  abnormal               : Ranked abnormal volume on positive-return days
  low_turnover           : Negative turnover (low activity signal)
  dtc_proxy              : Days-to-cover proxy (21d avg vol / current vol)
  overnight_ret          : Overnight return (open / prev close - 1)

  --- Trend & Factor Alphas ---
  trend_10               : 210d trend filter * risk-adjusted 12m momentum
  pair_spread_spy        : Price ratio vs SPY (relative value)
  style_mom_proxy        : Ranked 12m momentum (cross-sectional rank)
  etf_stat               : Ranked abs z-score vs 63d mean (mean reversion)
  long_short             : Ranked long momentum * ranked short reaction
  cycl                   : Ranked negative 30d cumulative returns
  size                   : Ranked negative dollar volume (size factor)
  seasonality_12m        : Lagged 252d return (same-month-last-year)

  --- Risk & Recovery Alphas ---
  mom_stoploss           : 12m momentum zeroed when drawdown > 5%
  falling_knife          : Ranked extreme drawdown * ranked stability
  post_gap_drift         : Gap * lagged momentum (gap continuation)
  max_daily_ret_21       : 21d max daily return (tail risk measure)
  price_shock            : 14d min return (downside shock)
  entry                  : 30d max close - 2*ATR (support level proxy)

  --- Technical Alphas ---
  hurst                  : Hurst exponent based trend signal (H>0.55 = trending)
  range                  : (High-Close)/(Close-Low) - upper vs lower wick
  skew+proxy             : (High-Low)/Low - intraday range normalized
  intraday_range         : (High-Low)/Close - normalized range
  alpha_w_005            : Rank(open - 10d VWAP avg) * -|rank(close-vwap)|
  sentiment_style        : Low-VIX regime * 12m momentum interaction

  --- Cross-Sectional Alphas ---
  residual_momentum      : 12m momentum minus beta-adjusted SPY momentum
  comomentum             : Rolling correlation of returns with market mean
  long_rev_mom           : Long-term (5yr) reversal * positive 12m momentum

  --- Lagged Features (5d, 10d, 30d) ---
  close_{n}, ret_{n}     : Lagged close and returns
  close_mean_{n}         : Rolling mean of close
  close_std_{n}          : Rolling std of close
  close_max_{n}, close_min_{n} : Rolling max/min of close
  ret_mean_{n}, ret_std_{n}    : Rolling mean/std of returns
  ret_max_{n}, ret_min_{n}     : Rolling max/min of returns
  ATR_{n}                : Average True Range at lag n

  --- Cross-Asset Beta & Correlation ---
  beta_SPY_{w}           : Rolling beta vs SPY (w=21d, 63d) per ticker
  corr_SPY_{w}           : Rolling correlation vs SPY per ticker
  beta_VIX_{w}           : Rolling beta vs VIX per ticker
  corr_VIX_{w}           : Rolling correlation vs VIX per ticker
  (All extracted as cross-features: e.g., beta_SPY_21_CL=F for copper)

================================================================================
  4. model.py -- STEP-BY-STEP DESCRIPTION
================================================================================

  STEP 1: Load CSVs & Compute AlphaLib Features
    - Load market_data_2005_2025.csv (12 macro/ETF tickers)
    - Load commodities_data_1990_2025.csv (18 commodity tickers incl HG=F)
    - Load companies_data_1990_2025.csv (mining stocks)
    - Concatenate, filter >= 2005, deduplicate
    - Pivot into daily_info format (DataFrames: open/high/low/close/volume)
    - Compute VWAP, returns, dollar amount
    - Run AlphaLib.calcu_alpha() -> dict of ~100+ alpha DataFrames
    - Extract copper (HG=F) column from each alpha
    - Cross-asset features: extract beta/corr of other tickers as copper features
    - Recompute signals from 3. test.py (Kalman, HMM, Donchian positions)
    - Split: Train 2005-2018 | Val 2019-2020 | Test 2021-2025

  STEP 2: Return Distribution Analysis
    - Compute log returns at 5 horizons: 1d, 5d, 20d, 30d, 60d
    - Compute z-scored returns (rolling 252d window) for LSTM targets
    - Histogram, Q-Q plot, ACF for each horizon
    - Jarque-Bera test (normality), ADF test (stationarity)
    - Save: claude/dist_analysis.png

  STEP 3: Feature-Return Correlations
    - Compute Pearson correlation of each feature vs each return horizon
    - Only on TRAIN set (no look-ahead)
    - Drop features with >30% missing values
    - Select features with |correlation| > 0.10
    - Run OLS regression for top features -> log_ret5
    - Save: claude/correlation_heatmap.png

  STEP 4: Feature Selection Summary
    - Group selected features by category (Alpha, Momentum, CrossAsset,
      Macro, VolStat, Signal, Other)
    - Print count and examples per category per target horizon

  STEP 5: LSTM Model Training
    - Architecture: 2-layer LSTM, 64 hidden units, dropout=0.2, dense(1)
    - Input: 20-day lookback sequences of selected features
    - Target: z-scored log_ret5 (standardized for better gradient flow)
    - Normalize features with train-period mean/std (no look-ahead)
    - PyTorch DataLoader with batch_size=64
    - Adam optimizer, lr=0.001, gradient clipping at 1.0
    - Early stopping on validation loss (patience=10)
    - Save best model state

  STEP 6: Model Evaluation
    - Metrics in z-score domain: MSE, RMSE, MAE, R2
    - Directional accuracy (% correct sign prediction)
    - Information Coefficient (IC) = Spearman rank correlation
    - Evaluate against both z-scored and raw returns

  STEP 7: LSTM Signal -> Trading Strategy
    - Convert z-score predictions to positions: tanh(pred / pred_std)
    - Vol-target to 20% annual (same as Donchian signals)
    - Apply VIX + DXY macro filters
    - Shift positions by 1 day (no look-ahead)

  STEP 8: Combine with Donchian Signals
    - Recompute all 6 Donchian signals (S0-S5) from 3. test.py
    - Strategy variants:
      A) LSTM-only: pure LSTM signal
      B) Combined: 50% S3_DualTF + 50% LSTM
      C) Master+LSTM: LSTM as 7th signal in ensemble (equal weight)

  STEP 9: Results & Plots
    - Summary table: Sharpe, return, max DD, hit rate for all strategies
    - Annual Sharpe breakdown (2021-2025)
    - Signal correlation matrix (diversity check)
    - Master figure (5x2 grid):
      Row 0: Training curves + Predicted vs Actual scatter
      Row 1: Cumulative returns (Train + Test)
      Row 2: Drawdowns + LSTM predictions vs actual (20d MA)
      Row 3: Annual Sharpe bars + Position correlations heatmap
      Row 4: Top feature correlations + Model summary text
    - Save: claude/model_results.png

================================================================================
  KEY RESULTS (from initial run)
================================================================================

  Master+LSTM (7-signal) achieves smallest max drawdown (-4.1%) while
  maintaining Sharpe > 2.0. The LSTM signal has near-zero correlation
  (-0.05 to -0.10) with all Donchian signals, providing genuine
  diversification benefit even though LSTM-only Sharpe is modest.

  The LSTM's value is in RISK REDUCTION, not return generation --
  consistent with the thesis that copper alpha comes from trend-following,
  and ML models are best used as filters/adapters rather than standalone
  predictors.

================================================================================
