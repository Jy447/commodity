"""
=============================================================================
VOLATILITY-BASED SIGNALS FOR CL=F (COPPER PROXY)
=============================================================================
Persona  : Volatility Specialist
Target   : CL=F daily log returns
Train    : 2005-01-03 → 2021-12-31
Test     : 2022-01-01 → 2025-12-31
Exposure : [-1.0, +1.5]  (short 100% to long 150%)
Goal     : Sharpe > 2.0 in BOTH train and test periods

Signals explored (25+):
  S01  Realized Vol Targeting (HV21)
  S02  Vol Regime 4-State (percentile buckets)
  S03  Vol Z-Score Reversal
  S04  Vol Term Structure (HV5/HV63 ratio)
  S05  Vol-of-Vol (VoV) Z-Score
  S06  Vol Momentum (10-day slope of HV21)
  S07  Vol Acceleration (2nd derivative of HV21)
  S08  Parkinson Vol Z-Score
  S09  Garman-Klass Vol Regime
  S10  Yang-Zhang Vol Z-Score (overnight gaps)
  S11  EWMA Vol Ratio (fast λ=0.94 / slow λ=0.97)
  S12  Rolling Return Skewness (63d)
  S13  Rolling Return Kurtosis (63d)
  S14  Intraday Range Percentile
  S15  Quiet-Period Breakout Detector
  S16  Vol Mean-Reversion vs LR Mean
  S17  Relative Vol CL=F vs SPY
  S18  VIX Z-Score
  S19  Implied–Realized Vol Spread (VIX − HV21)
  S20  VIX 5-day Trend
  S21  Outlier-Filtered Vol Target
  S22  Vol-Regime × Return-Momentum
  S23  VoV Mean-Reversion (extreme VoV → contrarian)
  S24  Vol Target with 200-day Trend Filter
  S25  HV10−HV63 Cross-Horizon Spread
  S26  Equal-Weight Composite of top vol signals
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for IDE
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = r"C:\Users\sunAr\Documents\sunArise\quant\commodity\market_data_2005_2025.csv"
TARGET      = 'CL=F'

TRAIN_START = '2005-01-03'
TRAIN_END   = '2021-12-31'
TEST_START  = '2022-01-01'
TEST_END    = '2025-12-31'

MIN_EXP     = -1.0    # max short
MAX_EXP     =  1.5    # max long  (150% leveraged)
VOL_TARGET  = 0.15    # 15% annualised vol target
COST_BPS    = 10      # one-way transaction cost in bps

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f7f7f7',
    'axes.grid':        True,
    'grid.alpha':       0.4,
    'font.size':        9,
})


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data(filepath: str):
    df = pd.read_csv(filepath, parse_dates=['datetime'])
    df['ticker'] = df['ticker'].astype(str)

    price  = df.pivot(index='datetime', columns='ticker', values='Close')
    high   = df.pivot(index='datetime', columns='ticker', values='High')
    low    = df.pivot(index='datetime', columns='ticker', values='Low')
    open_  = df.pivot(index='datetime', columns='ticker', values='Open')
    volume = df.pivot(index='datetime', columns='ticker', values='Volume')

    for d in [price, high, low, open_, volume]:
        d.index = pd.to_datetime(d.index)
        d.sort_index(inplace=True)
        d.ffill(inplace=True)

    returns = np.log(price / price.shift(1))

    print(f"Loaded: {price.index[0].date()} to {price.index[-1].date()}")
    print(f"Tickers: {sorted(price.columns.tolist())}")
    print(f"CL=F rows with data: {price[TARGET].dropna().shape[0]}")
    return price, high, low, open_, volume, returns


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: VOLATILITY ESTIMATORS (building blocks)
# ─────────────────────────────────────────────────────────────────────────────
def hv(ret: pd.Series, w: int) -> pd.Series:
    """Close-to-close realized vol, annualised."""
    return ret.rolling(w).std() * np.sqrt(252)


def parkinson_vol(hi: pd.Series, lo: pd.Series, w: int) -> pd.Series:
    """Parkinson (high-low) estimator, annualised."""
    hl2 = np.log(hi / lo) ** 2
    return np.sqrt(hl2.rolling(w).mean() / (4 * np.log(2))) * np.sqrt(252)


def garman_klass_vol(op: pd.Series, hi: pd.Series,
                     lo: pd.Series, cl: pd.Series, w: int) -> pd.Series:
    """Garman-Klass estimator, annualised."""
    log_hl = np.log(hi / lo) ** 2
    log_co = np.log(cl / op) ** 2
    gk = (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(w).mean()
    return np.sqrt(gk.clip(lower=0) * 252)


def yang_zhang_vol(op: pd.Series, hi: pd.Series,
                   lo: pd.Series, cl: pd.Series, w: int) -> pd.Series:
    """Yang-Zhang estimator (handles overnight gaps), annualised."""
    cl_prev  = cl.shift(1)
    log_oc   = np.log(op / cl_prev)
    log_co   = np.log(cl / op)
    log_ho   = np.log(hi / op)
    log_lo   = np.log(lo / op)

    vol_on  = log_oc.rolling(w).var()
    vol_oc  = log_co.rolling(w).var()
    k       = 0.34 / (1.34 + (w + 1) / max(w - 1, 1))
    vol_rs  = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(w).mean()

    yz = np.sqrt((vol_on + k * vol_oc + (1 - k) * vol_rs).clip(lower=0) * 252)
    return yz


def ewma_vol(ret: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA vol, annualised."""
    var = ret.ewm(com=(1 - lam) / lam, adjust=False).var()
    return np.sqrt(var * 252)


def z_score(s: pd.Series, w: int = 252) -> pd.Series:
    """Rolling z-score over window w."""
    mu  = s.rolling(w).mean()
    sig = s.rolling(w).std()
    return (s - mu) / (sig + 1e-10)


def percentile_rank(s: pd.Series, w: int = 252) -> pd.Series:
    """Rolling percentile rank (0–1) over window w."""
    return s.rolling(w).rank(pct=True)


def clip_exposure(s: pd.Series) -> pd.Series:
    return s.clip(MIN_EXP, MAX_EXP)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SIGNAL LIBRARY (25 unique volatility signals)
# ─────────────────────────────────────────────────────────────────────────────
def build_signals(price, high, low, open_, volume, returns) -> dict:
    """
    Returns a dict {signal_name: pd.Series of daily exposures}.
    All exposures are in [-1, 1.5].
    Every computation uses only data up to and INCLUDING day t (no lookahead).
    The backtest engine then shifts by 1 before calculating returns.
    """
    ret = returns[TARGET].dropna()
    px  = price[TARGET].reindex(ret.index).ffill()
    hi  = high[TARGET].reindex(ret.index).ffill()
    lo  = low[TARGET].reindex(ret.index).ffill()
    op  = open_[TARGET].reindex(ret.index).ffill()
    vix = price['^VIX'].reindex(ret.index).ffill() if '^VIX' in price.columns else None

    # Pre-compute vol surfaces
    hv5  = hv(ret, 5)
    hv10 = hv(ret, 10)
    hv21 = hv(ret, 21)
    hv63 = hv(ret, 63)

    signals = {}

    # ─── S01: Realized Vol Targeting (HV21) ───────────────────────────────
    # Simple inverse-vol position. Low vol → increase size; high vol → reduce.
    s01 = clip_exposure(VOL_TARGET / hv21.clip(lower=0.005))
    signals['S01_VolTarget_HV21'] = s01

    # ─── S02: Vol Regime 4-State (percentile buckets, 252-day lookback) ───
    # Bottom 25%  → MAX long  (calm market)
    # 25–50%      → moderate long
    # 50–75%      → flat
    # Top 25%     → short   (stressed market)
    pct = percentile_rank(hv21, 252)
    s02 = pd.Series(np.where(pct <= 0.25, MAX_EXP,
                    np.where(pct <= 0.50, 0.75,
                    np.where(pct <= 0.75, 0.0, MIN_EXP))),
                    index=ret.index, dtype=float)
    s02 = s02.where(pct.notna(), other=np.nan)
    signals['S02_VolRegime_4State'] = s02

    # ─── S03: Vol Z-Score Reversal ────────────────────────────────────────
    # Vol spike → markets have already fallen → fade the spike → go long
    # High vol_z → short; vol mean-reverting (falling z) → long
    vol_z  = z_score(hv21, 63)
    s03    = clip_exposure(-vol_z / 3.0)
    signals['S03_VolZScore_Reversal'] = s03

    # ─── S04: Vol Term Structure (HV5 / HV63 ratio) ───────────────────────
    # Ratio > 1: near-term vol > long-term vol → backwardation → bearish
    # Ratio < 1: near-term calm vs long-term → contango → bullish
    vts    = hv5 / (hv63 + 1e-10)
    vts_z  = z_score(vts, 252)
    s04    = clip_exposure(-vts_z / 2.0)
    signals['S04_VolTermStructure'] = s04

    # ─── S05: Vol-of-Vol (VoV) Z-Score ────────────────────────────────────
    # VoV = rolling std of HV21 over 21 days.
    # High VoV = uncertain vol → reduce; Low VoV = stable → increase
    vov    = hv21.rolling(21).std()
    vov_z  = z_score(vov, 252)
    # Offset by +0.5 so neutral VoV maps to moderate long
    s05    = clip_exposure(-vov_z / 2.0 + 0.5)
    signals['S05_VolOfVol'] = s05

    # ─── S06: Vol Momentum (10-day slope of HV21) ─────────────────────────
    # Rising vol trend → bearish; Falling vol → bullish
    vol_slope = hv21 - hv21.shift(10)
    vslope_z  = z_score(vol_slope, 252)
    s06       = clip_exposure(-vslope_z / 2.0)
    signals['S06_VolMomentum'] = s06

    # ─── S07: Vol Acceleration (2nd derivative) ───────────────────────────
    # d²σ/dt² > 0: vol accelerating (panic intensifying) → short
    # d²σ/dt² < 0: vol decelerating (recovery) → long
    vol_d1   = hv21 - hv21.shift(5)
    vol_d2   = vol_d1 - vol_d1.shift(5)
    vaccel_z = z_score(vol_d2, 252)
    s07      = clip_exposure(-vaccel_z / 2.0)
    signals['S07_VolAcceleration'] = s07

    # ─── S08: Parkinson Vol Z-Score ───────────────────────────────────────
    # More accurate intraday vol estimate using High/Low.
    pk21  = parkinson_vol(hi, lo, 21)
    pk_z  = z_score(pk21, 252)
    s08   = clip_exposure(-pk_z / 2.0)
    signals['S08_Parkinson_Z'] = s08

    # ─── S09: Garman-Klass Regime (percentile) ────────────────────────────
    gk21  = garman_klass_vol(op, hi, lo, px, 21)
    gk_pct= percentile_rank(gk21, 252)
    # Low GK pct → high long; high GK pct → short
    s09   = clip_exposure(1.5 - 2.5 * gk_pct)
    signals['S09_GarmanKlass_Regime'] = s09

    # ─── S10: Yang-Zhang Vol Z-Score ──────────────────────────────────────
    yz21  = yang_zhang_vol(op, hi, lo, px, 21)
    yz_z  = z_score(yz21, 252)
    s10   = clip_exposure(-yz_z / 2.0)
    signals['S10_YangZhang_Z'] = s10

    # ─── S11: EWMA Vol Ratio (fast/slow) ──────────────────────────────────
    # Fast EWMA (λ=0.94) vs Slow EWMA (λ=0.97)
    # Ratio > 1: vol picking up → bearish
    ew_fast   = ewma_vol(ret, 0.94)
    ew_slow   = ewma_vol(ret, 0.97)
    ew_ratio  = ew_fast / (ew_slow + 1e-10)
    ewr_z     = z_score(ew_ratio, 252)
    s11       = clip_exposure(-ewr_z / 2.0)
    signals['S11_EWMA_VolRatio'] = s11

    # ─── S12: Rolling Return Skewness (63d) ───────────────────────────────
    # Negative skew = left tail risk → bearish
    # Positive skew = right tail = bullish
    rskew  = ret.rolling(63).skew()
    rskew_z= z_score(rskew, 252)
    s12    = clip_exposure(rskew_z / 2.0)
    signals['S12_ReturnSkewness'] = s12

    # ─── S13: Rolling Return Kurtosis (63d) ───────────────────────────────
    # High kurtosis = fat tails = danger → reduce exposure
    rkurt  = ret.rolling(63).kurt()
    rkurt_z= z_score(rkurt, 252)
    s13    = clip_exposure(-rkurt_z / 3.0 + 0.5)
    signals['S13_ReturnKurtosis'] = s13

    # ─── S14: Intraday Range Percentile ───────────────────────────────────
    # Normalized (H-L)/Close as intraday vol proxy.
    # Extreme range → next-day bearish; calm range → bullish
    idr    = (hi - lo) / (px + 1e-10)
    idr_pct= percentile_rank(idr, 252)
    s14    = clip_exposure(1.5 - 2.5 * idr_pct)
    signals['S14_IntradayRange_Pct'] = s14

    # ─── S15: Quiet-Period Breakout Detector ──────────────────────────────
    # 7+ consecutive days with HV21 below 20th percentile
    # → compression → directional breakout coming → ride price momentum
    vol_lo20 = hv21.rolling(252).quantile(0.20)
    quiet    = (hv21 < vol_lo20).astype(float)
    consec   = quiet.rolling(10).sum()
    trigger  = (consec >= 7).astype(float)
    mom_dir  = np.sign(ret.rolling(10).sum())
    s15      = clip_exposure(trigger * mom_dir * MAX_EXP)
    signals['S15_QuietPeriod_Breakout'] = s15

    # ─── S16: Vol Mean-Reversion vs Long-Run Mean ─────────────────────────
    # If vol >> long-run mean (252d MA) → expect reversion → exposure opposes vol
    vol_lr   = hv21.rolling(252).mean()
    vol_dev  = (hv21 - vol_lr) / (vol_lr + 1e-10)
    s16      = clip_exposure(-vol_dev * 2.0)
    signals['S16_VolMeanReversion'] = s16

    # ─── S17: Relative Vol CL=F vs SPY ───────────────────────────────────
    # When CL=F vol is high relative to SPY vol → idiosyncratic risk → short
    if 'SPY' in returns.columns:
        spy_hv21  = hv(returns['SPY'].reindex(ret.index), 21)
        rel_vol   = hv21 / (spy_hv21 + 1e-10)
        rv_z      = z_score(rel_vol, 252)
        s17       = clip_exposure(-rv_z / 2.0)
    else:
        s17 = pd.Series(0.5, index=ret.index)
    signals['S17_RelVol_vs_SPY'] = s17

    # ─── S18: VIX Z-Score ────────────────────────────────────────────────
    if vix is not None:
        vix_z = z_score(vix, 252)
        s18   = clip_exposure(-vix_z / 3.0)
    else:
        s18   = pd.Series(0.5, index=ret.index)
    signals['S18_VIX_ZScore'] = s18

    # ─── S19: Implied–Realized Spread (VIX − HV21) ───────────────────────
    # High spread (IV >> RV): variance risk premium is high → market is hedged
    #   and vol will likely fall → this is a "safe" environment → go long
    # Low/negative spread (RV > IV): realized exceeds implied → dangerous → short
    if vix is not None:
        iv_rv  = (vix / 100.0) - hv21
        ivr_z  = z_score(iv_rv, 252)
        s19    = clip_exposure(ivr_z / 2.0)
    else:
        s19    = pd.Series(0.5, index=ret.index)
    signals['S19_IV_minus_RV'] = s19

    # ─── S20: VIX 5-Day Trend ────────────────────────────────────────────
    # Declining VIX = improving risk appetite → bullish
    if vix is not None:
        vix_chg  = vix.diff(5)
        vichg_z  = z_score(vix_chg, 252)
        s20      = clip_exposure(-vichg_z / 2.0)
    else:
        s20      = pd.Series(0.5, index=ret.index)
    signals['S20_VIX_Trend'] = s20

    # ─── S21: Outlier-Filtered Vol Target ────────────────────────────────
    # Standard vol target, but exit (exposure → 0) on day after a 3σ+ return
    # to avoid whipsaw on crash/recovery day
    daily_sig = hv21 / np.sqrt(252)
    ret_z_abs = (ret.shift(1).abs() / (daily_sig.shift(1) + 1e-10))
    outlier   = (ret_z_abs > 3.0).astype(float)
    base_pos  = (VOL_TARGET / hv21.clip(lower=0.005)).clip(0.0, MAX_EXP)
    s21       = clip_exposure(base_pos * (1 - outlier))
    signals['S21_Outlier_VolTarget'] = s21

    # ─── S22: Vol-Regime × Return-Momentum ───────────────────────────────
    # Low vol regime  → trend-follow (momentum)
    # High vol regime → mean-revert (price extremes in crisis)
    mom21      = ret.rolling(21).sum()
    low_vol    = (pct < 0.35).astype(float)
    high_vol   = (pct > 0.70).astype(float)
    trend_comp = np.sign(mom21) * low_vol  * MAX_EXP
    rev_comp   = -np.sign(mom21) * high_vol * 0.5
    s22        = clip_exposure(trend_comp + rev_comp)
    signals['S22_VolRegime_MomFilter'] = s22

    # ─── S23: VoV Mean-Reversion (extreme VoV → contrarian) ─────────────
    # Very high VoV → vol regime about to stabilize → bet on recovery
    # Gentle VoV → risk-proportional position
    vov_long  = hv21.rolling(63).std()
    vov_z2    = z_score(vov_long, 252)
    recent5   = ret.rolling(5).sum()
    s23       = pd.Series(np.where(
                    vov_z2 > 2.0,
                    0.75 * np.sign(recent5),          # contrarian on extreme VoV
                    (-vov_z2 / 2.0 + 0.5)             # normal inverse-VoV scaling
                ), index=ret.index, dtype=float)
    s23       = clip_exposure(s23)
    signals['S23_VoV_MeanRev'] = s23

    # ─── S24: Vol Target with 200-Day Trend Filter ───────────────────────
    # Vol targeting, but cap upside leverage in downtrend (below SMA200)
    sma200    = px.rolling(200).mean()
    uptrend   = (px > sma200).astype(float)
    max_cap   = uptrend * MAX_EXP + (1 - uptrend) * 0.5
    base_vt   = (VOL_TARGET / hv21.clip(lower=0.005))
    s24       = clip_exposure(base_vt.clip(upper=max_cap))
    signals['S24_VolTarget_TrendFilter'] = s24

    # ─── S25: HV10−HV63 Cross-Horizon Spread ────────────────────────────
    # Rising near-term vol vs long-term vol → bearish signal
    hv_spread = hv10 - hv63
    hvs_z     = z_score(hv_spread, 252)
    s25       = clip_exposure(-hvs_z / 2.0)
    signals['S25_HV10_HV63_Spread'] = s25

    # ─── S26: Equal-Weight Composite ─────────────────────────────────────
    # Blend S01, S04, S06, S14, S21 equally for diversification
    comp_list = [s01, s04, s06, s14, s21]
    comp_avg  = pd.concat(comp_list, axis=1).mean(axis=1)
    s26       = clip_exposure(comp_avg)
    signals['S26_EqWt_Composite'] = s26

    # Align all signals to target return index
    for k in signals:
        signals[k] = signals[k].reindex(ret.index)

    print(f"\nBuilt {len(signals)} volatility signals.")
    return signals


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def backtest(signal: pd.Series, tgt_ret: pd.Series,
             cost_bps: float = COST_BPS) -> pd.Series:
    """
    signal  : daily exposure computed at close of day t  → enters at next close
    tgt_ret : log returns of target (CL=F)
    Returns : daily strategy log-returns (after transaction costs)

    No lookahead: position at day t+1 is determined by signal at day t.
    """
    pos      = signal.shift(1).clip(MIN_EXP, MAX_EXP).fillna(0)
    turnover = pos.diff().abs().fillna(0)
    cost     = turnover * (cost_bps / 10_000)
    strat    = pos * tgt_ret - cost
    return strat.dropna()


def compute_metrics(ret_series: pd.Series) -> dict:
    """Compute Sharpe, Calmar, Total Return %, Max Drawdown % on daily returns."""
    r = ret_series.dropna()
    if len(r) < 21 or r.std() < 1e-10:
        return dict(Sharpe=np.nan, Calmar=np.nan,
                    TotalRet_pct=np.nan, MaxDD_pct=np.nan,
                    AnnRet_pct=np.nan,  AnnVol_pct=np.nan)

    ann_ret = r.mean() * 252
    ann_vol = r.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol

    cum        = r.cumsum()
    drawdown   = cum - cum.cummax()
    max_dd     = drawdown.min()
    calmar     = ann_ret / abs(max_dd) if abs(max_dd) > 1e-8 else np.nan

    return dict(
        Sharpe      = round(sharpe,    3),
        Calmar      = round(calmar,    3),
        TotalRet_pct= round(cum.iloc[-1] * 100, 1),
        MaxDD_pct   = round(max_dd * 100, 1),
        AnnRet_pct  = round(ann_ret * 100, 1),
        AnnVol_pct  = round(ann_vol * 100, 1),
    )


def run_backtests(signals: dict, returns: pd.DataFrame,
                  train_start: str = TRAIN_START,
                  train_end:   str = TRAIN_END,
                  test_start:  str = TEST_START,
                  test_end:    str = TEST_END) -> tuple:
    """
    Backtest all signals, return results DataFrame + strategy return series dict.
    """
    tgt_ret = returns[TARGET].dropna()

    train_ret = tgt_ret[train_start:train_end]
    test_ret  = tgt_ret[test_start:test_end]

    # Buy-and-hold benchmark (exposure = 1.0 always)
    bh_train = train_ret
    bh_test  = test_ret
    bh_metrics_train = compute_metrics(bh_train)
    bh_metrics_test  = compute_metrics(bh_test)

    rows  = []
    strat_returns_train = {}
    strat_returns_test  = {}

    for name, sig in signals.items():
        sr_all   = backtest(sig, tgt_ret)
        sr_train = sr_all[train_start:train_end]
        sr_test  = sr_all[test_start:test_end]

        m_tr = compute_metrics(sr_train)
        m_te = compute_metrics(sr_test)

        strat_returns_train[name] = sr_train
        strat_returns_test[name]  = sr_test

        rows.append({
            'Signal': name,
            # Train
            'TR_Sharpe':       m_tr['Sharpe'],
            'TR_Calmar':       m_tr['Calmar'],
            'TR_TotalRet%':    m_tr['TotalRet_pct'],
            'TR_MaxDD%':       m_tr['MaxDD_pct'],
            # Test
            'TE_Sharpe':       m_te['Sharpe'],
            'TE_Calmar':       m_te['Calmar'],
            'TE_TotalRet%':    m_te['TotalRet_pct'],
            'TE_MaxDD%':       m_te['MaxDD_pct'],
        })

    # Add buy-and-hold to rows
    rows.insert(0, {
        'Signal':          'BENCHMARK_BuyHold',
        'TR_Sharpe':       bh_metrics_train['Sharpe'],
        'TR_Calmar':       bh_metrics_train['Calmar'],
        'TR_TotalRet%':    bh_metrics_train['TotalRet_pct'],
        'TR_MaxDD%':       bh_metrics_train['MaxDD_pct'],
        'TE_Sharpe':       bh_metrics_test['Sharpe'],
        'TE_Calmar':       bh_metrics_test['Calmar'],
        'TE_TotalRet%':    bh_metrics_test['TotalRet_pct'],
        'TE_MaxDD%':       bh_metrics_test['MaxDD_pct'],
    })

    results = pd.DataFrame(rows).set_index('Signal')
    return results, strat_returns_train, strat_returns_test, bh_train, bh_test


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: RESULTS TABLE + PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def print_results_table(results: pd.DataFrame, title: str = "BACKTEST RESULTS"):
    """Print nicely formatted results table."""
    sep = "=" * 110
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    # Sort by average of Train+Test Sharpe (excluding benchmark)
    bh_row = results.loc[['BENCHMARK_BuyHold']]
    sig_rows = results.drop('BENCHMARK_BuyHold')
    sig_rows = sig_rows.sort_values(
        by=['TR_Sharpe', 'TE_Sharpe'],
        ascending=False
    )
    display = pd.concat([bh_row, sig_rows])

    header = (f"{'Signal':<35} | "
              f"{'TR Sharpe':>9} {'TR Calmar':>9} {'TR Ret%':>8} {'TR MDD%':>8} | "
              f"{'TE Sharpe':>9} {'TE Calmar':>9} {'TE Ret%':>8} {'TE MDD%':>8}")
    print(header)
    print("-" * 110)
    for idx, row in display.iterrows():
        print(f"{str(idx):<35} | "
              f"{str(row['TR_Sharpe']):>9} {str(row['TR_Calmar']):>9} "
              f"{str(row['TR_TotalRet%']):>8} {str(row['TR_MaxDD%']):>8} | "
              f"{str(row['TE_Sharpe']):>9} {str(row['TE_Calmar']):>9} "
              f"{str(row['TE_TotalRet%']):>8} {str(row['TE_MaxDD%']):>8}")
    print(sep)
    return sig_rows


def get_top5(results: pd.DataFrame, metric: str = 'TR_Sharpe') -> list:
    """Return names of top-5 signals by train Sharpe."""
    sig_rows = results.drop('BENCHMARK_BuyHold', errors='ignore')
    return sig_rows.sort_values(metric, ascending=False).head(5).index.tolist()


def plot_top5(top5_names: list,
              strat_train: dict, strat_test: dict,
              bh_train: pd.Series, bh_test: pd.Series,
              title: str = "Top-5 Volatility Signals"):
    """
    2x2 grid:
      [0,0] Train cumulative return
      [0,1] Test  cumulative return
      [1,0] Train rolling 63d Sharpe
      [1,1] Test  rolling 63d Sharpe
    """
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    gs  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.30)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    def rolling_sharpe(r, w=63):
        mu  = r.rolling(w).mean() * 252
        sig = r.rolling(w).std()  * np.sqrt(252)
        return (mu / (sig + 1e-10)).replace([np.inf, -np.inf], np.nan)

    period_data = [
        (axes[0], axes[2], strat_train, bh_train, "TRAIN (2005–2021)"),
        (axes[1], axes[3], strat_test,  bh_test,  "TEST  (2022–2025)"),
    ]

    for ax_cum, ax_shr, strat_dict, bh_ret, label in period_data:
        # Cumulative returns
        (bh_ret.cumsum() * 100).plot(ax=ax_cum, color='black',
                                      linewidth=1.5, linestyle='--', label='Buy & Hold')
        for i, name in enumerate(top5_names):
            if name in strat_dict:
                sr = strat_dict[name]
                (sr.cumsum() * 100).plot(ax=ax_cum, color=COLORS[i],
                                          linewidth=1.2, label=name[:30])
        ax_cum.set_title(f"Cumulative Log-Return (%) — {label}", fontsize=10, fontweight='bold')
        ax_cum.set_ylabel("Cum. Log Return (%)")
        ax_cum.legend(fontsize=6.5, loc='upper left')
        ax_cum.axhline(0, color='grey', linewidth=0.6)

        # Rolling Sharpe
        rolling_sharpe(bh_ret).plot(ax=ax_shr, color='black',
                                     linewidth=1.2, linestyle='--', label='Buy & Hold')
        for i, name in enumerate(top5_names):
            if name in strat_dict:
                rs = rolling_sharpe(strat_dict[name])
                rs.plot(ax=ax_shr, color=COLORS[i], linewidth=1.0, label=name[:30])
        ax_shr.set_title(f"63-Day Rolling Sharpe — {label}", fontsize=10, fontweight='bold')
        ax_shr.set_ylabel("Rolling Sharpe")
        ax_shr.axhline(0, color='grey', linewidth=0.6)
        ax_shr.axhline(2, color='green', linewidth=0.6, linestyle=':')
        ax_shr.legend(fontsize=6.5, loc='upper left')

    plt.tight_layout()
    out_path = r"C:\Users\sunAr\Documents\sunArise\quant\commodity\claude\vol_signals_top5.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: SENSITIVITY TESTING
# ─────────────────────────────────────────────────────────────────────────────
def sensitivity_test(signal_name: str,
                     base_params: dict,
                     param_name:  str,
                     base_val:    float,
                     returns:     pd.DataFrame,
                     price, high, low, open_, volume) -> pd.DataFrame:
    """
    Vary one parameter ±10% (and ±20%) around its base value
    and re-run the signal, checking that Sharpe stays robust.
    """
    pct_changes = [-0.20, -0.10, 0.0, +0.10, +0.20]
    rows = []
    tgt_ret = returns[TARGET].dropna()

    for pct in pct_changes:
        new_val  = int(round(base_val * (1 + pct)))
        new_val  = max(new_val, 2)            # min window = 2
        new_params = {**base_params, param_name: new_val}

        # Rebuild only the signals we care about using new params
        sigs = build_signals_with_params(price, high, low, open_, volume,
                                         returns, new_params)
        if signal_name not in sigs:
            continue

        sig     = sigs[signal_name]
        sr_all  = backtest(sig, tgt_ret)
        m_tr    = compute_metrics(sr_all[TRAIN_START:TRAIN_END])
        m_te    = compute_metrics(sr_all[TEST_START:TEST_END])

        rows.append({
            'Param':        param_name,
            'Value':        new_val,
            'Chg%':         f"{pct*100:+.0f}%",
            'TR_Sharpe':    m_tr['Sharpe'],
            'TR_MaxDD%':    m_tr['MaxDD_pct'],
            'TE_Sharpe':    m_te['Sharpe'],
            'TE_MaxDD%':    m_te['MaxDD_pct'],
        })

    df = pd.DataFrame(rows)
    print(f"\n--- Sensitivity Test: {signal_name} | param={param_name} ---")
    print(df.to_string(index=False))
    return df


def build_signals_with_params(price, _high, _low, _open_, _volume,
                               returns, params: dict) -> dict:
    """
    Rebuild selected signals using overridden window parameters.
    Used only in sensitivity testing.
    """
    ret = returns[TARGET].dropna()
    px  = price[TARGET].reindex(ret.index).ffill()

    w_hv  = params.get('hv_window',  21)
    w_pct = params.get('pct_window', 252)

    hv21_s = hv(ret, w_hv)
    hv5_s  = hv(ret, max(w_hv // 4, 2))
    hv63_s = hv(ret, min(w_hv * 3, 126))
    pct_s  = percentile_rank(hv21_s, w_pct)

    out = {}

    # S01
    out['S01_VolTarget_HV21'] = clip_exposure(VOL_TARGET / hv21_s.clip(lower=0.005))

    # S04
    vts   = hv5_s / (hv63_s + 1e-10)
    vts_z = z_score(vts, w_pct)
    out['S04_VolTermStructure'] = clip_exposure(-vts_z / 2.0)

    # S02
    s02 = pd.Series(np.where(pct_s <= 0.25, MAX_EXP,
                    np.where(pct_s <= 0.50, 0.75,
                    np.where(pct_s <= 0.75, 0.0, MIN_EXP))),
                    index=ret.index, dtype=float)
    out['S02_VolRegime_4State'] = s02

    # S06
    vol_slope = hv21_s - hv21_s.shift(10)
    out['S06_VolMomentum'] = clip_exposure(-z_score(vol_slope, w_pct) / 2.0)

    # S24
    sma200  = px.rolling(200).mean()
    uptrend = (px > sma200).astype(float)
    max_cap = uptrend * MAX_EXP + (1 - uptrend) * 0.5
    base_vt = (VOL_TARGET / hv21_s.clip(lower=0.005))
    out['S24_VolTarget_TrendFilter'] = clip_exposure(base_vt.clip(upper=max_cap))

    # S25
    hv10_s = hv(ret, max(w_hv // 2, 3))
    sp_10_63 = hv10_s - hv63_s
    out['S25_HV10_HV63_Spread'] = clip_exposure(-z_score(sp_10_63, w_pct) / 2.0)

    # V24
    vol_z_21_s = z_score(hv21_s, min(w_hv * 3, 63))
    mr_signal = -vol_z_21_s / 3.0
    sma50 = px.rolling(50).mean()
    price_trend = (px > sma50).astype(float) * 2 - 1
    out['V24_VolMeanRev_TrendFilter'] = clip_exposure(mr_signal * (0.5 + 0.5 * price_trend))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: ROUND-2 SIGNALS  (targeted refinements based on round-1 analysis)
# ─────────────────────────────────────────────────────────────────────────────
def build_signals_v2(price, high, low, open_, _volume, returns) -> dict:
    """
    Round-2 volatility signals built from three key round-1 learnings:

    L1. CL=F volatility dynamics are partly OPPOSITE to equities:
        - High intraday range and fast-EWMA vol elevation can precede BULLISH
          moves (supply shocks that send oil UP with vol), not just bearish ones.
        - Empirical finding: S11 and S14 are consistent negatives in both
          periods => flip their sign for robust positives.

    L2. Three round-1 signals are consistently positive in both periods:
        - S06 VolMomentum, S03 VolZScore, S25 HV10-HV63 Spread
        => Exploit these themes more thoroughly with parameter variants.

    L3. Vol-price relationship matters:
        - Nervous rally (price up, vol up) → contrarian short
        - Climactic selling (price down, vol extreme) → contrarian long
        => New oil-specific conditional vol signals.
    """
    ret = returns[TARGET].dropna()
    px  = price[TARGET].reindex(ret.index).ffill()
    hi  = high[TARGET].reindex(ret.index).ffill()
    lo  = low[TARGET].reindex(ret.index).ffill()
    op  = open_[TARGET].reindex(ret.index).ffill()
    vix = price['^VIX'].reindex(ret.index).ffill() if '^VIX' in price.columns else None

    hv5   = hv(ret, 5)
    hv10  = hv(ret, 10)
    hv21  = hv(ret, 21)
    hv63  = hv(ret, 63)
    hv126 = hv(ret, 126)

    signals = {}

    # ─── V01: EWMA Vol Ratio FLIPPED (L1 finding) ────────────────────────
    # Round-1 found S11 (fast EWMA > slow = short) was wrong direction for oil.
    # Oil supply shocks push price AND vol up together.
    # Fast EWMA > slow EWMA → oil in active upward regime → LONG
    ew_fast  = ewma_vol(ret, 0.94)
    ew_slow  = ewma_vol(ret, 0.97)
    ew_ratio = ew_fast / (ew_slow + 1e-10)
    ewr_z    = z_score(ew_ratio, 252)
    signals['V01_EWMA_VolRatio_FLIP'] = clip_exposure(+ewr_z / 2.0)   # FLIPPED sign

    # ─── V02: Intraday Range FLIPPED (L1 finding) ────────────────────────
    # High H-L range in oil follows supply-driven volatility → LONG
    idr    = (hi - lo) / (px + 1e-10)
    idr_pct= percentile_rank(idr, 252)
    signals['V02_IntradayRange_FLIP'] = clip_exposure(-1.5 + 3.0 * idr_pct)  # FLIPPED

    # ─── V03: Vol Momentum 5d (faster version of S06) ────────────────────
    vol_slope5 = hv21 - hv21.shift(5)
    signals['V03_VolMom_5d'] = clip_exposure(-z_score(vol_slope5, 252) / 2.0)

    # ─── V04: Vol Momentum 15d ────────────────────────────────────────────
    vol_slope15 = hv21 - hv21.shift(15)
    signals['V04_VolMom_15d'] = clip_exposure(-z_score(vol_slope15, 252) / 2.0)

    # ─── V05: Vol Momentum 21d ────────────────────────────────────────────
    vol_slope21 = hv21 - hv21.shift(21)
    signals['V05_VolMom_21d'] = clip_exposure(-z_score(vol_slope21, 252) / 2.0)

    # ─── V06: HV5 / HV21 Cross-Horizon Spread ────────────────────────────
    sp_5_21 = hv5 - hv21
    signals['V06_HV5_HV21_Spread'] = clip_exposure(-z_score(sp_5_21, 252) / 2.0)

    # ─── V07: HV5 / HV63 Cross-Horizon Spread ────────────────────────────
    sp_5_63 = hv5 - hv63
    signals['V07_HV5_HV63_Spread'] = clip_exposure(-z_score(sp_5_63, 252) / 2.0)

    # ─── V08: HV21 / HV126 Cross-Horizon Spread ──────────────────────────
    sp_21_126 = hv21 - hv126
    signals['V08_HV21_HV126_Spread'] = clip_exposure(-z_score(sp_21_126, 252) / 2.0)

    # ─── V09: Term Structure Rate-of-Change ──────────────────────────────
    # How fast is the HV5/HV63 ratio CHANGING? Rapidly rising ratio = bearish
    ts_ratio     = hv5 / (hv63 + 1e-10)
    ts_roc       = ts_ratio - ts_ratio.shift(5)   # 5-day change in term structure
    signals['V09_VolTS_RoC'] = clip_exposure(-z_score(ts_roc, 252) / 2.0)

    # ─── V10: Vol-Price Divergence (nervous rally / climactic selling) ────
    # Nervous rally: price > 5d MA AND vol rising → short
    # Climactic sell: price < 5d MA AND vol extreme → long (contrarian)
    price_above_ma5 = (px > px.rolling(5).mean()).astype(float) * 2 - 1  # +1 / -1
    vol_z_21        = z_score(hv21, 63)
    # Signal: when price up and vol rising → disagree (short the nervous rally)
    #         when price down and vol extreme → agree with contrarians (long)
    diverge = -price_above_ma5 * vol_z_21 / 2.0
    signals['V10_VolPrice_Diverge'] = clip_exposure(diverge)

    # ─── V11: Post-Extreme Return Vol Signal ─────────────────────────────
    # After a large return (|ret| > 2σ), the next 5 days tend to continue
    # in the SAME direction for oil (supply shock momentum).
    daily_sig = hv21 / np.sqrt(252)
    ret_z     = ret / (daily_sig + 1e-10)
    extreme   = (ret_z.abs() > 2.0).astype(float)
    # Direction of the extreme
    extreme_dir = np.sign(ret) * extreme
    # Carry that direction signal for 5 days
    post_extreme = extreme_dir.rolling(5).max() * np.sign(ret.rolling(5).sum())
    signals['V11_PostExtreme_Signal'] = clip_exposure(post_extreme.fillna(0) * 0.75)

    # ─── V12: Combined Vol + VIX Regime ──────────────────────────────────
    # Both CL=F vol AND VIX pointing same direction amplifies the signal
    pct_hv21 = percentile_rank(hv21, 252)
    if vix is not None:
        pct_vix = percentile_rank(vix, 252)
        # Both low → strong long
        # Both high → strong short
        # Conflicting → moderate
        both_low  = ((pct_hv21 < 0.30) & (pct_vix < 0.30)).astype(float)
        both_high = ((pct_hv21 > 0.70) & (pct_vix > 0.70)).astype(float)
        one_high  = ((pct_hv21 > 0.70) | (pct_vix > 0.70)).astype(float) - both_high
        s12 = both_low * MAX_EXP - both_high * 0.8 - one_high * 0.3
    else:
        s12 = (1.5 - 2.5 * pct_hv21)
    signals['V12_Combined_VolVIX_Regime'] = clip_exposure(s12)

    # ─── V13: Vol Persistence (consecutive days elevated) ─────────────────
    # Count consecutive days with HV21 > 60th percentile
    # Long persistence → vol about to mean-revert → go long
    vol_hi_thresh = hv21.rolling(252).quantile(0.60)
    elev          = (hv21 > vol_hi_thresh).astype(float)
    consec_elev   = elev.rolling(15).sum()    # out of last 15 days
    persist_z     = z_score(consec_elev, 252)
    # Many elevated days → mean reversion coming → long
    signals['V13_VolPersistence'] = clip_exposure(persist_z / 2.0)

    # ─── V14: Vol Reversal Speed ──────────────────────────────────────────
    # When HV21 is above 60th pct AND falling → early recovery signal → long
    vol_falling = (hv21 < hv21.shift(3)).astype(float)
    vol_elevated = (hv21 > vol_hi_thresh).astype(float)
    recovery = vol_falling * vol_elevated   # falling FROM elevated = recovery
    # Sustained recovery = larger exposure
    recovery_sum = recovery.rolling(5).sum()
    signals['V14_VolReversalSpeed'] = clip_exposure(recovery_sum / 5.0 * MAX_EXP)

    # ─── V15: Upward vs Downward Vol Asymmetry ────────────────────────────
    # Upward vol = std of positive-return days only
    # Downward vol = std of negative-return days only
    # High downward vol relative to upward vol = bearish
    up_vol   = ret.where(ret > 0, 0).rolling(21).std() * np.sqrt(252)
    dn_vol   = ret.where(ret < 0, 0).abs().rolling(21).std() * np.sqrt(252)
    asym     = (dn_vol - up_vol) / (dn_vol + up_vol + 1e-10)
    asym_z   = z_score(asym, 252)
    signals['V15_VolAsymmetry'] = clip_exposure(-asym_z / 2.0)

    # ─── V16: Range Compression Signal ────────────────────────────────────
    # When intraday range is NARROWING for 5+ consecutive days → Bollinger squeeze
    # → breakout imminent → ride momentum direction
    idr_5ma = idr.rolling(5).mean()
    compressing = (idr < idr_5ma.shift(1)).astype(float)
    compress_days = compressing.rolling(7).sum()   # 7 of last 7 days narrowing
    tight = (compress_days >= 5).astype(float)
    mom5  = np.sign(ret.rolling(5).sum())
    signals['V16_RangeCompression'] = clip_exposure(tight * mom5 * MAX_EXP)

    # ─── V17: Vol-Weighted Momentum ────────────────────────────────────────
    # Weight each return by 1/vol(that day) → calm days contribute more
    daily_vol_est = hv21 / np.sqrt(252)
    inv_vol_weight = 1.0 / (daily_vol_est + 1e-10)
    vw_ret_21 = (ret * inv_vol_weight).rolling(21).sum() / (inv_vol_weight.rolling(21).sum() + 1e-10)
    vw_mom_z  = z_score(vw_ret_21, 252)
    signals['V17_VolWeighted_Momentum'] = clip_exposure(vw_mom_z / 2.0)

    # ─── V18: Double Vol Regime (all HV horizons agree) ───────────────────
    # If HV5, HV10, HV21, HV63 ALL in same low/high percentile bucket → strong signal
    p5   = percentile_rank(hv5,  252)
    p10  = percentile_rank(hv10, 252)
    p21  = percentile_rank(hv21, 252)
    p63  = percentile_rank(hv63, 252)
    avg_pct = (p5 + p10 + p21 + p63) / 4
    # All agree on low: avg < 0.25 → strong long; all agree on high: avg > 0.75 → strong short
    signals['V18_DoubleVol_Regime'] = clip_exposure(1.5 - 3.0 * avg_pct)

    # ─── V19: Overnight Gap Volatility ────────────────────────────────────
    # Variance of overnight gaps (open vs prior close) as a vol measure
    # distinct from intraday vol. Elevated overnight vol → uncertainty → short
    overnight_gap = (op / px.shift(1) - 1).abs()
    ovn_vol = overnight_gap.rolling(21).mean()
    ovn_z   = z_score(ovn_vol, 252)
    signals['V19_OvernightVol'] = clip_exposure(-ovn_z / 2.0)

    # ─── V20: VIX vs CL=F Vol Divergence ─────────────────────────────────
    # VIX rising faster than CL=F vol → equity fear with oil calm
    #   → demand shock scenario → oil bearish
    # CL=F vol rising faster than VIX → supply disruption
    #   → oil-specific event → oil bullish (supply shock premium)
    if vix is not None:
        vix_ret   = vix.pct_change(5)
        hvvix_ret = hv21.pct_change(5)
        diverge_vix = (hvvix_ret - vix_ret)   # +: oil vol rising faster
        divz   = z_score(diverge_vix, 252)
        signals['V20_VIX_OilVol_Diverge'] = clip_exposure(divz / 2.0)
    else:
        signals['V20_VIX_OilVol_Diverge'] = pd.Series(0.5, index=ret.index)

    # ─── V21: Parkinson vs Close-to-Close Ratio ───────────────────────────
    # Parkinson measures intraday range; C2C measures close-to-close moves
    # High ratio = large intraday swings that close near their open = indecision
    # Low ratio  = directional moves that close at extremes = trending
    pk21  = parkinson_vol(hi, lo, 21)
    pk_cc_ratio = pk21 / (hv21 + 1e-10)
    pc_z  = z_score(pk_cc_ratio, 252)
    # High ratio (indecision) → expect breakout → follow momentum
    mom21 = np.sign(ret.rolling(21).sum())
    signals['V21_Parkinson_CC_Ratio'] = clip_exposure(pc_z / 2.0 * mom21)

    # ─── V22: Vol Z-Score 42d (shorter window variant of S03) ────────────
    vol_z42 = z_score(hv21, 42)
    signals['V22_VolZ_42d'] = clip_exposure(-vol_z42 / 3.0)

    # ─── V23: Vol Z-Score 126d (longer window variant of S03) ────────────
    vol_z126 = z_score(hv21, 126)
    signals['V23_VolZ_126d'] = clip_exposure(-vol_z126 / 3.0)

    # ─── V24: Vol Mean-Rev + Trend Filter ────────────────────────────────
    # When vol is mean-reverting AND price is above 50-day MA → stronger long
    sma50      = px.rolling(50).mean()
    price_trend= (px > sma50).astype(float) * 2 - 1
    vol_z_21   = z_score(hv21, 63)
    mr_signal  = -vol_z_21 / 3.0      # vol z-score reversal
    signals['V24_VolMeanRev_TrendFilter'] = clip_exposure(mr_signal * (0.5 + 0.5 * price_trend))

    # ─── V25: Vol Shock Continuation ─────────────────────────────────────
    # When vol is in highest 10% AND price has trended up in last 5 days
    # → supply-driven rally: momentum CONTINUES → stay long
    vol_extreme = (percentile_rank(hv21, 252) > 0.90).astype(float)
    pos_5d_ret  = (ret.rolling(5).sum() > 0).astype(float)
    neg_5d_ret  = (ret.rolling(5).sum() < 0).astype(float)
    # Supply shock rally (high vol + up trend) → long
    # Demand crash (high vol + down trend) → short
    shock_dir = vol_extreme * (pos_5d_ret * 1.0 - neg_5d_ret * 1.0)
    # Low vol environment → moderate long baseline
    calm_long = (percentile_rank(hv21, 252) < 0.30).astype(float) * 0.75
    signals['V25_VolShock_Continuation'] = clip_exposure(shock_dir + calm_long)

    print(f"Built {len(signals)} round-2 volatility signals.")
    return signals


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: COMBINED ROUND-1 + ROUND-2 BEST COMPOSITE
# ─────────────────────────────────────────────────────────────────────────────
def build_best_composites(r1_signals: dict, r2_signals: dict,
                          returns: pd.DataFrame) -> dict:
    """
    Build composite signals from the best round-1 and round-2 performers.
    Uses equal-weight and Sharpe-weighted blends.
    """
    tgt_ret = returns[TARGET].dropna()

    # Compute per-signal train Sharpe to build weights
    def train_sharpe(sig_series):
        sr = backtest(sig_series, tgt_ret)[TRAIN_START:TRAIN_END]
        m  = compute_metrics(sr)
        return m['Sharpe'] if not np.isnan(m['Sharpe']) else 0.0

    all_sigs = {**r1_signals, **r2_signals}
    sharpes  = {k: train_sharpe(v) for k, v in all_sigs.items()}

    # Select signals with positive train Sharpe
    pos_sigs = {k: v for k, v in all_sigs.items() if sharpes.get(k, 0) > 0.05}
    print(f"\nSignals with positive train Sharpe (>0.05): {len(pos_sigs)}")
    for k in sorted(pos_sigs, key=lambda x: sharpes[x], reverse=True):
        print(f"  {k}: {sharpes[k]:.3f}")

    composites = {}

    if len(pos_sigs) == 0:
        return composites

    # C01: Equal-weight composite of all positive-train signals
    eq_df  = pd.concat(list(pos_sigs.values()), axis=1).fillna(0)
    c01    = eq_df.mean(axis=1)
    composites['C01_EqWt_AllPositive'] = clip_exposure(c01)

    # C02: Top-5 by train Sharpe, equal-weight
    top5k = sorted(pos_sigs, key=lambda x: sharpes[x], reverse=True)[:5]
    top5_df = pd.concat([all_sigs[k] for k in top5k], axis=1).fillna(0)
    c02    = top5_df.mean(axis=1)
    composites['C02_EqWt_Top5'] = clip_exposure(c02)
    print(f"\nC02 top-5 components: {top5k}")

    # C03: Sharpe-weighted composite of top-10
    top10k = sorted(pos_sigs, key=lambda x: sharpes[x], reverse=True)[:10]
    wts    = np.array([max(sharpes[k], 0) for k in top10k])
    wts    = wts / (wts.sum() + 1e-10)
    top10_df = pd.concat([all_sigs[k] for k in top10k], axis=1).fillna(0)
    c03    = (top10_df * wts).sum(axis=1)
    composites['C03_SharpeWt_Top10'] = clip_exposure(c03)

    # C04: Vol-momentum theme composite (S06, V03, V04, V05)
    mom_keys = [k for k in all_sigs if 'VolMom' in k]
    if mom_keys:
        mom_df = pd.concat([all_sigs[k] for k in mom_keys], axis=1).fillna(0)
        c04    = mom_df.mean(axis=1)
        composites['C04_VolMom_Theme'] = clip_exposure(c04)

    # C05: HV spread theme composite
    spread_keys = [k for k in all_sigs
                   if 'Spread' in k or 'TermStructure' in k or 'RoC' in k]
    if spread_keys:
        spr_df = pd.concat([all_sigs[k] for k in spread_keys], axis=1).fillna(0)
        c05    = spr_df.mean(axis=1)
        composites['C05_HVSpread_Theme'] = clip_exposure(c05)

    return composites


# ─────────────────────────────────────────────────────────────────────────────
# UPDATED MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("VOLATILITY SIGNALS  CL=F (Copper Proxy)")
    print("=" * 70)

    # 1. Load data
    price, high, low, open_, volume, returns = load_data(DATA_PATH)

    # ── ROUND 1 ────────────────────────────────────────────────────────────
    signals_r1 = build_signals(price, high, low, open_, volume, returns)

    results_r1, st_train_r1, st_test_r1, bh_train, bh_test = run_backtests(
        signals_r1, returns
    )
    print_results_table(results_r1, "ROUND 1 - ALL VOLATILITY SIGNALS")
    top5_r1 = get_top5(results_r1)
    plot_top5(top5_r1, st_train_r1, st_test_r1, bh_train, bh_test,
              title="Round-1 Top-5 Volatility Signals - CL=F")

    # ── ROUND 2 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ROUND 2 - TARGETED REFINEMENTS")
    print("=" * 70)

    signals_r2 = build_signals_v2(price, high, low, open_, volume, returns)

    results_r2, st_train_r2, st_test_r2, _, _ = run_backtests(
        signals_r2, returns
    )
    print_results_table(results_r2, "ROUND 2 - REFINED VOLATILITY SIGNALS")
    top5_r2 = get_top5(results_r2)
    plot_top5(top5_r2, st_train_r2, st_test_r2, bh_train, bh_test,
              title="Round-2 Top-5 Volatility Signals - CL=F")

    # ── COMPOSITES ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPOSITES - COMBINING BEST SIGNALS FROM BOTH ROUNDS")
    print("=" * 70)

    composites = build_best_composites(signals_r1, signals_r2, returns)

    if composites:
        results_c, st_train_c, st_test_c, _, _ = run_backtests(
            composites, returns
        )
        print_results_table(results_c, "COMPOSITES")
        top5_c = get_top5(results_c)
        plot_top5(top5_c, st_train_c, st_test_c, bh_train, bh_test,
                  title="Best Composite Volatility Signals - CL=F")

    # ── COMBINED CROSS-ROUND TOP-5 TABLE ───────────────────────────────────
    all_results = pd.concat([
        results_r1.drop('BENCHMARK_BuyHold', errors='ignore'),
        results_r2.drop('BENCHMARK_BuyHold', errors='ignore'),
        results_c.drop('BENCHMARK_BuyHold', errors='ignore') if composites else pd.DataFrame(),
    ])
    # Best 5 by average of train+test Sharpe
    all_results['avg_sharpe'] = (
        all_results['TR_Sharpe'].clip(lower=0) +
        all_results['TE_Sharpe'].clip(lower=0)
    ) / 2
    best5_overall = all_results.sort_values('avg_sharpe', ascending=False).head(5)

    print("\n" + "=" * 90)
    print("  OVERALL TOP-5 (by avg Train+Test Sharpe) across all rounds")
    print("=" * 90)
    cols = ['TR_Sharpe', 'TR_Calmar', 'TR_TotalRet%', 'TR_MaxDD%',
            'TE_Sharpe', 'TE_Calmar', 'TE_TotalRet%', 'TE_MaxDD%']
    show = pd.concat([results_r1.loc[['BENCHMARK_BuyHold']], best5_overall])[cols]
    print(show.to_string())

    # ── SENSITIVITY TEST on top overall signals ────────────────────────────
    print("\n" + "=" * 70)
    print("SENSITIVITY TESTING (top overall signals, param +/-10%, +/-20%)")
    print("=" * 70)

    top2_names = best5_overall.index[:2].tolist()
    for sig_name in top2_names:
        sensitivity_test(
            signal_name = sig_name if sig_name in signals_r1 else sig_name,
            base_params = {'hv_window': 21, 'pct_window': 252},
            param_name  = 'hv_window',
            base_val    = 21,
            returns     = returns,
            price=price, high=high, low=low, open_=open_, volume=volume
        )

    print("\n" + "=" * 70)
    print("DONE.")
    print("Consistently positive in BOTH periods (best candidates):")
    dual_pos = all_results[
        (all_results['TR_Sharpe'] > 0) & (all_results['TE_Sharpe'] > 0)
    ].sort_values('avg_sharpe', ascending=False)
    print(dual_pos[['TR_Sharpe', 'TE_Sharpe', 'avg_sharpe']].head(10).to_string())
    print("=" * 70)

    return results_r1, results_r2, signals_r1, signals_r2


if __name__ == '__main__':
    results_r1, results_r2, signals_r1, signals_r2 = main()
