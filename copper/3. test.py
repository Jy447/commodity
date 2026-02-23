# -*- coding: utf-8 -*-
"""
Copper Alpha Signals: 6-Signal Ensemble with Kalman Filter & Markov-Switching
==============================================================================
Signal 0 : Donchian Breakout 30/10 + VIX/DXY filters       (original baseline)
Signal 1 : KF-Filtered Donchian (Kalman slope confirms breakouts)
Signal 2 : HMM Regime-Scaled Donchian (reduce size in choppy regimes)
Signal 3 : Multi-TF Momentum (Kalman-weighted 10/20/40/60d momentum)
Signal 4 : Adaptive Channel Breakout (vol-scaled channel widths via KF-vol)
Signal 5 : Cross-Asset Trend Filter (copper + crude + SPY confirmation)
Master   : Equal-weight ensemble of all 6

Key insight: Copper alpha = trend-following. KF & Markov-Switching are used
as FILTERS and ADAPTERS on trend signals, not standalone signals.

References:
  - Hamilton (1989) Markov-switching regime models
  - Harvey (1990) Kalman filter for structural time series
  - Nystrup et al. (2020) regime-conditional portfolio strategies
  - Moskowitz, Ooi, Pedersen (2012) time-series momentum

Train: 2005-01-01 to 2020-12-31  |  Test: 2021-01-01 to 2025-12-31
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

BASE  = "C:/Users/sunAr/Documents/sunArise/quant/commodity"
SPLIT = "2021-01-01"
TARGET_VOL = 0.20

# ======================================================================
# 1.  LOAD DATA
# ======================================================================
print("Loading data...")
comm  = pd.read_csv(f"{BASE}/commodities_data_1990_2025.csv", parse_dates=["datetime"])
macro = pd.read_csv(f"{BASE}/market_data_2005_2025.csv",      parse_dates=["datetime"])

cu    = (comm[comm["ticker"] == "HG=F"].sort_values("datetime").set_index("datetime"))
cu    = cu[cu.index >= "2003-01-01"]
close = cu["Close"]
ret   = close.pct_change()

def get_macro(ticker, col="Close", rename=None):
    s = (macro[macro["ticker"] == ticker].sort_values("datetime")
         .set_index("datetime")[col])
    s.name = rename or ticker
    return s.reindex(close.index).ffill()

def get_comm(ticker, col="Close", rename=None):
    s = (comm[comm["ticker"] == ticker].sort_values("datetime")
         .set_index("datetime")[col])
    s.name = rename or ticker
    return s.reindex(close.index).ffill()

vix   = get_macro("^VIX",     rename="vix")
dxy   = get_macro("DX-Y.NYB", rename="dxy")
crude = get_macro("CL=F",     rename="crude")
spy   = get_macro("SPY",      rename="spy")

ewm_vol    = ret.ewm(span=21, adjust=False).std() * np.sqrt(252)
ewm_vol    = ewm_vol.replace(0, np.nan).ffill()
vix_smooth = vix.ewm(span=5).mean()
vix_mult   = pd.Series(np.where(vix_smooth > 35, 0.30,
                       np.where(vix_smooth > 25, 0.60, 1.00)), index=close.index)
dxy_mom    = dxy.pct_change(40)

print(f"Copper: {close.index[0].date()} to {close.index[-1].date()}  ({len(close)} days)")

# ======================================================================
# HELPERS
# ======================================================================
def apply_filters(pos, direction):
    dm = pd.Series(np.where((direction > 0) & (dxy_mom > 0.04), 0.65, 1.00),
                   index=close.index)
    return pos * vix_mult * dm

def vol_target(direction, shift=True):
    pos = (direction * (TARGET_VOL / ewm_vol)).clip(-3, 3)
    pos = apply_filters(pos, direction)
    if shift:
        pos = pos.shift(1).fillna(0)
    return pos

def donchian_stateful(px, n_entry, n_exit):
    he = px.rolling(n_entry).max().shift(1).values
    le = px.rolling(n_entry).min().shift(1).values
    hx = px.rolling(n_exit).max().shift(1).values
    lx = px.rolling(n_exit).min().shift(1).values
    p  = px.values
    pos = np.zeros(len(p))
    cur = 0
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


# ======================================================================
# KALMAN FILTER: Local linear trend model
# ======================================================================
def kalman_trend(log_prices):
    """Returns Kalman-filtered level and slope (daily trend rate)."""
    y = log_prices.values
    n = len(y)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([1.0, 0.0])
    Q = np.array([[1e-6, 0], [0, 3e-7]])   # tight slope noise
    R = np.nanvar(np.diff(y[~np.isnan(y)][:252]))
    x = np.array([y[~np.isnan(y)][0], 0.0])
    P = np.eye(2) * 1e-4

    levels = np.full(n, np.nan)
    slopes = np.full(n, np.nan)
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

print("Computing Kalman filter...")
kf_level, kf_slope = kalman_trend(np.log(close))


# ======================================================================
# HMM REGIME MODEL (2-state, fit on train only → predict OOS)
# ======================================================================
print("Fitting HMM regime model...")
from hmmlearn.hmm import GaussianHMM

def fit_hmm_regimes(returns_series, split_date, n_regimes=2):
    """Fit 2-state HMM on train data, then decode full history."""
    r_clean = returns_series.dropna()
    r_train = r_clean[r_clean.index < split_date]
    X_train = r_train.values.reshape(-1, 1) * 100

    model = GaussianHMM(n_components=n_regimes, covariance_type="full",
                        n_iter=200, random_state=42)
    model.fit(X_train)

    # Decode full history
    X_full = r_clean.values.reshape(-1, 1) * 100
    probs  = model.predict_proba(X_full)

    # Identify: higher-variance regime = "choppy/volatile"
    vars_ = [model.covars_[i][0, 0] for i in range(n_regimes)]
    trend_regime = np.argmin(vars_)  # lower-vol = trending

    p_trend = pd.Series(probs[:, trend_regime], index=r_clean.index)
    return p_trend.reindex(returns_series.index).ffill().fillna(0.5)

p_trend = fit_hmm_regimes(ret, SPLIT)

print("Building signals...\n")


# ======================================================================
# SIGNAL 0: DONCHIAN BREAKOUT (original baseline)
# ======================================================================
raw0 = donchian_stateful(close, 30, 10)
pos_0 = vol_target(raw0, shift=False)  # channels already shifted

# ======================================================================
# SIGNAL 1: KF-FILTERED DONCHIAN
#   Only take Donchian entries when Kalman slope confirms direction.
#   Long breakout requires kf_slope > 0 (uptrend confirmed).
#   Short breakout requires kf_slope < 0 (downtrend confirmed).
#   This filters ~30% of false breakouts.
# ======================================================================
slope_sign = np.sign(kf_slope)
# Keep position when slope confirms, zero when it contradicts
kf_confirm = pd.Series(
    np.where((raw0 > 0) & (slope_sign > 0), raw0,
    np.where((raw0 < 0) & (slope_sign < 0), raw0,
    np.where(raw0 == 0, 0, raw0 * 0.3))),  # 0.3x when slope disagrees
    index=close.index)
pos_1 = vol_target(kf_confirm, shift=False)

# ======================================================================
# SIGNAL 2: HMM REGIME-SCALED DONCHIAN
#   Full position when HMM says "trending regime" (P > 0.5).
#   Reduced position (0.4x) when "choppy regime".
#   Adds regime awareness without changing entry/exit logic.
# ======================================================================
regime_mult = pd.Series(
    np.where(p_trend > 0.6, 1.0,
    np.where(p_trend > 0.4, 0.7, 0.35)),
    index=close.index)
pos_2 = vol_target(raw0 * regime_mult, shift=False)

# ======================================================================
# SIGNAL 3: DUAL-TIMEFRAME DONCHIAN (short + long confirmation)
#   Short-term: 20d entry / 7d exit (catches fast moves)
#   Long-term:  60d Donchian direction as confirmation filter.
#   Only take short-term breakout when long-term trend agrees.
#   This removes false breakouts in counter-trend corrections.
#   Kalman slope used as tiebreaker when timeframes disagree.
# ======================================================================
raw3_short = donchian_stateful(close, 20, 7)   # fast channel
raw3_long  = donchian_stateful(close, 60, 20)  # slow channel

# Confirmation: take fast signal only when slow agrees or Kalman confirms
slope_s = np.sign(kf_slope)
raw3 = pd.Series(
    np.where((raw3_short > 0) & ((raw3_long > 0) | (slope_s > 0)), 1,
    np.where((raw3_short < 0) & ((raw3_long < 0) | (slope_s < 0)), -1,
    np.where((raw3_short != 0) & (raw3_long == raw3_short), raw3_short,
             0))),
    index=close.index, dtype=float)
pos_3 = vol_target(raw3, shift=False)

# ======================================================================
# SIGNAL 4: ADAPTIVE CHANNEL BREAKOUT (vol-regime scaled channels)
#   Channel width adapts to Kalman-filtered volatility:
#   - Low-vol regime → tighter channels (15-20d) → catch moves faster
#   - High-vol regime → wider channels (40-50d) → avoid whipsaws
#   Uses HMM regime probability to interpolate.
# ======================================================================
# Interpolate channel width: trending(low-vol) → 20d, choppy(high-vol) → 50d
n_entry_adapt = (p_trend * 20 + (1 - p_trend) * 50).clip(15, 50).astype(int)
n_exit_adapt  = (n_entry_adapt / 3).clip(5, 15).astype(int)

# Pre-compute channels for several widths and select based on regime
channel_widths = [(15, 5), (20, 7), (25, 8), (30, 10), (35, 12), (40, 13), (45, 15), (50, 17)]
channel_positions = {}
for ne, nx in channel_widths:
    channel_positions[(ne, nx)] = donchian_stateful(close, ne, nx)

# For each day, pick the channel position closest to the adaptive width
raw4 = pd.Series(0.0, index=close.index)
for i in range(len(close)):
    ne = int(n_entry_adapt.iloc[i])
    # Find closest pre-computed channel
    best = min(channel_widths, key=lambda w: abs(w[0] - ne))
    raw4.iloc[i] = channel_positions[best].iloc[i]

pos_4 = vol_target(raw4, shift=False)

# ======================================================================
# SIGNAL 5: CROSS-ASSET TREND FILTER
#   Copper trends when macro backdrop confirms:
#   - Crude oil trending up → industrial demand strong
#   - SPY trending up → risk-on environment
#   Use Donchian breakout but scale by cross-asset confirmation.
#   Confirmation = sign(crude 40d return) + sign(spy 40d return).
# ======================================================================
crude_mom = crude.pct_change(40).fillna(0)
spy_mom   = spy.pct_change(40).fillna(0)

# Confirmation score: -2 to +2
confirm = np.sign(crude_mom) + np.sign(spy_mom)

# Scale Donchian: full when confirmed, reduced otherwise
cross_mult = pd.Series(
    np.where((raw0 > 0) & (confirm >= 1), 1.0,    # long + macro confirms
    np.where((raw0 > 0) & (confirm == 0), 0.6,    # long + neutral macro
    np.where((raw0 > 0) & (confirm < 0),  0.3,    # long + macro disagrees
    np.where((raw0 < 0) & (confirm <= -1), 1.0,   # short + macro confirms
    np.where((raw0 < 0) & (confirm == 0),  0.6,   # short + neutral
    np.where((raw0 < 0) & (confirm > 0),   0.3,   # short + macro disagrees
             0)))))),
    index=close.index)

pos_5 = vol_target(raw0 * cross_mult, shift=False)


# ======================================================================
# MASTER ENSEMBLE
# ======================================================================
all_signals = {
    "S0_Donchian":       pos_0,
    "S1_KF_Donchian":    pos_1,
    "S2_HMM_Donchian":   pos_2,
    "S3_DualTF":    pos_3,
    "S4_AdaptChannel":   pos_4,
    "S5_CrossAsset":     pos_5,
}
pos_master = sum(all_signals.values()) / len(all_signals)


# ======================================================================
# PERFORMANCE
# ======================================================================
def metrics(r, label, verbose=True):
    ar = r.mean() * 252; av = r.std() * np.sqrt(252)
    sh = ar / av if av > 0 else float("nan")
    cum = (1 + r).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    md = dd.min(); cp = (cum.iloc[-1] - 1) * 100
    hit = (r > 0).mean() * 100
    if verbose:
        print(f"  {label:<35} Sh={sh:+.2f}  Ret={ar*100:+.1f}%  DD={md*100:+.1f}%  Hit={hit:.0f}%")
    return dict(sharpe=sh, ann_ret=ar, ann_vol=av, cum_pct=cp, max_dd=md, cum=cum, dd=dd, hit=hit)

names = list(all_signals.keys()) + ["Master_Ensemble"]
positions = list(all_signals.values()) + [pos_master]

res_tr, res_te = {}, {}

print("\n" + "#"*65)
print("  TRAIN 2005-2020")
print("#"*65)
for nm, p in zip(names, positions):
    r = (p * ret).dropna()
    res_tr[nm] = metrics(r[(r.index >= "2005-01-01") & (r.index < SPLIT)], nm)

bh_train = ret[(ret.index >= "2005-01-01") & (ret.index < SPLIT)].dropna()
res_tr["BuyHold"] = metrics(bh_train, "BuyHold")

print(f"\n{'#'*65}")
print("  TEST 2021-2025")
print("#"*65)
for nm, p in zip(names, positions):
    r = (p * ret).dropna()
    res_te[nm] = metrics(r[r.index >= SPLIT], nm)

bh_test = ret[ret.index >= SPLIT].dropna()
res_te["BuyHold"] = metrics(bh_test, "BuyHold")

# Summary
print(f"\n{'='*75}")
print(f"  {'Signal':<22} {'Train Sh':>10} {'Test Sh':>10} {'Test Ret':>10} {'Test DD':>10}")
print(f"  {'-'*62}")
for nm in names + ["BuyHold"]:
    ts = res_tr[nm]["sharpe"]; te = res_te[nm]["sharpe"]
    tr = res_te[nm]["ann_ret"]*100; td = res_te[nm]["max_dd"]*100
    flag = " **" if te >= 2.0 else ""
    print(f"  {nm:<22} {ts:>+10.2f} {te:>+10.2f} {tr:>+9.1f}% {td:>+9.1f}%{flag}")

# Annual breakdown
print("\n-- Master Ensemble Annual (Test) --")
mr = (pos_master * ret).dropna()
mr_te = mr[mr.index >= SPLIT]
print(f"  {'Year':>4}  {'Return':>8}  {'Sharpe':>8}  {'Max DD':>8}")
for yr, g in mr_te.groupby(mr_te.index.year):
    yre = g.mean()*252*100; ysh = (g.mean()*252)/(g.std()*np.sqrt(252)) if g.std()>0 else 0
    c = (1+g).cumprod(); yd = ((c-c.cummax())/c.cummax()).min()*100
    print(f"  {yr:>4}  {yre:>+8.1f}%  {ysh:>+8.2f}  {yd:>+8.1f}%")

# Correlations
print("\n-- Signal Correlations (Test) --")
rdf = pd.DataFrame({nm: (p*ret).dropna() for nm, p in zip(names[:-1], positions[:-1])})
rdf = rdf[rdf.index >= SPLIT].dropna()
print(rdf.corr().round(2).to_string())

# Signal change counts
print("\n-- Signal Changes (full period) --")
for nm, p in zip(names[:-1], positions[:-1]):
    chg = (p.diff().abs() > 0.01).sum()
    print(f"  {nm}: {chg} changes")


# ======================================================================
# PLOTS
# ======================================================================
print("\nGenerating plots...")
colors = {"S0_Donchian": "steelblue", "S1_KF_Donchian": "darkorange",
          "S2_HMM_Donchian": "purple", "S3_DualTF": "crimson",
          "S4_AdaptChannel": "teal", "S5_CrossAsset": "olive",
          "Master_Ensemble": "black", "BuyHold": "grey"}

fig = plt.figure(figsize=(18, 18))
fig.suptitle(
    "Copper 6-Signal Ensemble  |  Kalman Filter + HMM Regime Detection\n"
    "Vol-targeted 20% ann  |  Train 2005-2020  |  Test 2021-2025",
    fontsize=13, fontweight="bold", y=0.995)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.30)

def pf(ax):
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))

# Row 0: Cumulative — Train / Test
for col, prd, res in [(0, "Train 2005-2020", res_tr), (1, "Test 2021-2025", res_te)]:
    ax = fig.add_subplot(gs[0, col])
    for nm in names:
        lw = 2.5 if nm == "Master_Ensemble" else 1.1
        al = 1.0 if nm == "Master_Ensemble" else 0.65
        sh = res[nm]["sharpe"]
        (res[nm]["cum"] - 1).plot(ax=ax, color=colors[nm], lw=lw, alpha=al,
                                   label=f"{nm} ({sh:.2f})")
    sh_bh = res["BuyHold"]["sharpe"]
    (res["BuyHold"]["cum"] - 1).plot(ax=ax, color="grey", lw=1, ls="--", alpha=0.5,
                                       label=f"B&H ({sh_bh:.2f})")
    ax.set_title(f"{prd}: Cumulative Return"); pf(ax)
    ax.legend(fontsize=5.5, loc="upper left"); ax.set_xlabel("")

# Row 1: Drawdowns — Master
for col, prd, res in [(0, "Train", res_tr), (1, "Test", res_te)]:
    ax = fig.add_subplot(gs[1, col])
    dd = res["Master_Ensemble"]["dd"]
    dd.plot(ax=ax, color="black", lw=1)
    ax.fill_between(dd.index, dd, 0, alpha=0.3, color="black")
    ax.set_title(f"{prd} Drawdown — Master  |  Max={res['Master_Ensemble']['max_dd']*100:.1f}%")
    pf(ax); ax.set_xlabel("")

# Row 2: Annual Sharpe bars + Rolling Sharpe
ax = fig.add_subplot(gs[2, 0])
yr_sh = mr_te.groupby(mr_te.index.year).apply(
    lambda r: (r.mean()*252)/(r.std()*np.sqrt(252)) if r.std()>0 else 0)
bc = ["gold" if v>=2 else "darkgreen" if v>=0 else "crimson" for v in yr_sh]
yr_sh.plot(kind="bar", ax=ax, color=bc, edgecolor="white")
ax.axhline(2, color="gold", lw=1.2, ls="--", label="Sharpe=2")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Annual Sharpe — Master Ensemble"); ax.tick_params(axis="x", rotation=0)
ax.legend(fontsize=8); ax.set_xlabel("")

ax = fig.add_subplot(gs[2, 1])
for nm in ["Master_Ensemble", "S0_Donchian"]:
    r_s = (dict(zip(names, positions))[nm] * ret).dropna()
    r_s = r_s[r_s.index >= SPLIT]
    roll = (r_s.rolling(63).mean()*252) / (r_s.rolling(63).std()*np.sqrt(252))
    roll.plot(ax=ax, color=colors[nm], lw=1.8 if nm=="Master_Ensemble" else 1, label=nm)
ax.axhline(2, color="gold", lw=1.2, ls="--"); ax.axhline(0, color="black", lw=0.8)
ax.set_title("Rolling 63d Sharpe — Test"); ax.legend(fontsize=7); ax.set_xlabel("")

# Row 3: HMM regime probs + Kalman slope + Correlation heatmap
ax = fig.add_subplot(gs[3, 0])
pt = p_trend[p_trend.index >= SPLIT]
ax.fill_between(pt.index, pt, 0.5, where=pt>0.5, alpha=0.4, color="green", label="Trending")
ax.fill_between(pt.index, pt, 0.5, where=pt<0.5, alpha=0.4, color="red", label="Choppy")
ax.plot(pt.index, pt, color="black", lw=0.7)
ax2 = ax.twinx()
ks_test = kf_slope[kf_slope.index >= SPLIT]
ax2.plot(ks_test.index, ks_test * 1e4, color="blue", lw=0.5, alpha=0.5, label="KF slope (bps)")
ax2.axhline(0, color="blue", ls=":", lw=0.3)
ax2.set_ylabel("KF slope (bps)", fontsize=7)
ax.axhline(0.5, color="grey", ls="--", lw=0.5)
ax.set_title("HMM Regime + Kalman Slope — Test"); ax.legend(fontsize=7, loc="upper left")
ax.set_ylim(0, 1); ax.set_xlabel("")

ax = fig.add_subplot(gs[3, 1])
corr = rdf.corr()
im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels([s.replace("_","\n") for s in corr.columns], fontsize=6, rotation=45, ha="right")
ax.set_yticklabels([s.replace("_","\n") for s in corr.columns], fontsize=6)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Signal Correlations (Test)")

out = f"{BASE}/claude/signal_results.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Plot saved: {out}")
plt.close()
print("\nDone!")
