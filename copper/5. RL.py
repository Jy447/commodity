# -*- coding: utf-8 -*-
"""
Multi-Asset RL Portfolio Agent: PPO + Differential Sharpe Ratio
================================================================
Faithful implementation of arXiv 2602.17098 methodology.

State:  (n+1) x T matrix  [portfolio weights | 60-day log returns per asset]
        Last row (cash):   [cash weight | vol_20 series | vol_20/vol_60 series | VIX series]
Action: Softmax portfolio weights across n assets + cash (long-only)
Reward: Differential Sharpe Ratio (Moody & Saffell 2001)

Assets: HG=F, SPY, QQQ, GLD, TLT, CL=F, GC=F, SI=F, IWM, BZ=F, NG=F, ZC=F + cash
Features: ^VIX, DX-Y.NYB (non-tradeable)

Train: 2005-2018 | Val: 2019-2020 | Test: 2021-2025
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# ======================================================================
# CONSTANTS
# ======================================================================
BASE       = "C:/Users/sunAr/Documents/sunArise/quant/commodity"
SPLIT_VAL  = "2019-01-01"
SPLIT_TEST = "2021-01-01"

# Tradeable assets
TRADE_ASSETS = ["HG=F", "SPY", "QQQ", "GLD", "TLT", "CL=F",
                "GC=F", "SI=F", "IWM", "BZ=F", "NG=F", "ZC=F"]
N_ASSETS = len(TRADE_ASSETS)     # 12
N_POSITIONS = N_ASSETS + 1       # 13 (includes cash)
COPPER_IDX = 0                   # HG=F is first

# Paper hyperparameters (arXiv 2602.17098, Table 1)
T           = 60                 # lookback window (days)
GAMMA       = 0.9                # discount factor
GAE_LAMBDA  = 0.9                # GAE lambda
PPO_CLIP    = 0.25               # clipping epsilon
ENTROPY_COEF = 0.01
VF_COEF     = 0.5
HIDDEN_DIM  = 64                 # [64, 64] MLP
LR_INIT     = 3e-4               # annealed to 1e-5
LR_FINAL    = 1e-5
MAX_GRAD    = 0.5
N_STEPS     = 756                # rollout length per update
PPO_EPOCHS  = 16                 # PPO epochs per rollout
BATCH_SIZE  = 1260
N_EPISODES  = 80                 # training episodes
N_SEEDS     = 5                  # agents per config
TC_BPS      = 5                  # transaction cost (basis points)

np.random.seed(42)
torch.manual_seed(42)

# ======================================================================
# 1. LOAD MULTI-ASSET DATA
# ======================================================================
print("=" * 70)
print("  STEP 1: Load Multi-Asset Data")
print("=" * 70)

market = pd.read_csv(f"{BASE}/market_data_2005_2025.csv", parse_dates=["datetime"])
commod = pd.read_csv(f"{BASE}/commodities_data_1990_2025.csv", parse_dates=["datetime"])
df_all = pd.concat([market, commod], ignore_index=True)
df_all = df_all[df_all["datetime"] >= "2005-01-01"]
df_all = df_all.sort_values("datetime").drop_duplicates(subset=["datetime", "ticker"], keep="last")

# Use Adj Close if available, else Close
col_close = "Adj Close" if not df_all["Adj Close"].isna().all() else "Close"
close_wide = df_all.pivot(index="datetime", columns="ticker", values=col_close)
close_wide = close_wide.sort_index().ffill()

# Extract tradeable + feature tickers
all_needed = TRADE_ASSETS + ["^VIX"]
missing = [t for t in all_needed if t not in close_wide.columns]
if missing:
    print(f"  WARNING: missing tickers: {missing}")

prices = close_wide[TRADE_ASSETS].copy()
vix_series = close_wide["^VIX"].copy() if "^VIX" in close_wide.columns else pd.Series(20.0, index=prices.index)

# Drop rows where copper has no data
prices = prices[prices["HG=F"].notna()]
vix_series = vix_series.reindex(prices.index).ffill().fillna(20.0)

# Forward-fill remaining NaNs (BZ=F starts 2007)
prices = prices.ffill().bfill()

print(f"  Price matrix: {prices.shape[0]} days x {prices.shape[1]} assets")
print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"  Assets: {list(prices.columns)}")

# Log returns
log_ret = np.log(prices / prices.shift(1))
log_ret.iloc[0] = 0.0

# Volatility features (SPY-based, matching paper)
spy_ret = log_ret["SPY"]
vol_20 = spy_ret.rolling(20, min_periods=10).std()
vol_60 = spy_ret.rolling(60, min_periods=30).std()
vol_ratio = (vol_20 / vol_60).replace([np.inf, -np.inf], np.nan).fillna(1.0)

print(f"  Vol features computed (SPY-based vol_20, vol_20/vol_60, VIX)")

# ======================================================================
# 2. EXPANDING-WINDOW STANDARDIZATION
# ======================================================================
print("\n" + "=" * 70)
print("  STEP 2: Expanding-Window Standardization")
print("=" * 70)

def expanding_zscore_series(s, min_periods=252):
    """Z-score using expanding mean/std (no look-ahead)."""
    emean = s.expanding(min_periods=min_periods).mean()
    estd  = s.expanding(min_periods=min_periods).std().replace(0, np.nan)
    return ((s - emean) / estd).clip(-5, 5).fillna(0)

# Standardize log returns per asset
log_ret_z = log_ret.apply(lambda col: expanding_zscore_series(col))

# Standardize vol features
vol_20_z    = expanding_zscore_series(vol_20)
vol_ratio_z = expanding_zscore_series(vol_ratio)
vix_z       = expanding_zscore_series(vix_series)

print(f"  Standardized returns: {log_ret_z.shape}")
print(f"  NaN check: {log_ret_z.isna().sum().sum()}")

# Convert to numpy arrays
dates       = prices.index
ret_arr     = log_ret_z.values.astype(np.float32)       # (T_total, n_assets)
raw_ret_arr = log_ret.values.astype(np.float32)          # raw returns for PnL
vol20_arr   = vol_20_z.values.astype(np.float32)         # (T_total,)
volr_arr    = vol_ratio_z.values.astype(np.float32)
vix_arr     = vix_z.values.astype(np.float32)
T_total     = len(dates)

# Split indices
train_end  = int(np.searchsorted(dates, pd.Timestamp(SPLIT_VAL)))
val_end    = int(np.searchsorted(dates, pd.Timestamp(SPLIT_TEST)))
test_end   = T_total

print(f"  Split: Train [T:{train_end}] | Val [{train_end}:{val_end}] | Test [{val_end}:{test_end}]")
print(f"  Days:  Train {train_end - T} | Val {val_end - train_end} | Test {test_end - val_end}")

# ======================================================================
# 3. DIFFERENTIAL SHARPE RATIO
# ======================================================================
print("\n" + "=" * 70)
print("  STEP 3: Differential Sharpe Ratio (DSR)")
print("=" * 70)

class DifferentialSharpeRatio:
    """
    DSR from Moody & Saffell (2001), as used in arXiv 2602.17098.
    D_t = (B*dA - 0.5*A*dB) / (B - A^2)^1.5
    """
    def __init__(self, eta=1.0/252):
        self.eta = eta
        self.A = 0.0
        self.B = 0.0

    def reset(self):
        self.A = 0.0
        self.B = 0.0

    def step(self, R_t):
        dA = R_t - self.A
        dB = R_t**2 - self.B
        denom = self.B - self.A**2
        if denom <= 1e-12:
            D_t = R_t
        else:
            D_t = (self.B * dA - 0.5 * self.A * dB) / (denom ** 1.5)
        self.A += self.eta * dA
        self.B += self.eta * dB
        return D_t

print("  DSR ready (eta=1/252)")

# ======================================================================
# 4. MULTI-ASSET PORTFOLIO ENVIRONMENT
# ======================================================================
print("\n" + "=" * 70)
print("  STEP 4: MultiAssetPortfolioEnv")
print("=" * 70)

class MultiAssetPortfolioEnv:
    """
    Multi-asset portfolio environment matching arXiv 2602.17098.

    State: (N_POSITIONS x T) matrix, flattened
      - Rows 0..N_ASSETS-1: [weight_i, 59 days of z-scored log returns]
      - Row N_ASSETS (cash):  [weight_cash, 20x vol_20_z, 20x vol_ratio_z, 19x vix_z]
    Action: raw logits -> softmax -> portfolio weights (long-only, sum=1)
    Reward: DSR of portfolio return (net of transaction costs)
    """
    def __init__(self, start_idx, end_idx):
        self.start_idx = max(start_idx, T)
        self.end_idx = end_idx
        self.state_dim = N_POSITIONS * T  # (n_assets+1) * 60
        self.action_dim = N_POSITIONS     # weights including cash
        self.dsr = DifferentialSharpeRatio()
        self.weights = np.zeros(N_POSITIONS, dtype=np.float32)
        self.weights[-1] = 1.0  # start in cash
        self.t = self.start_idx
        self.done = False

    def reset(self):
        self.dsr.reset()
        self.weights = np.zeros(N_POSITIONS, dtype=np.float32)
        self.weights[-1] = 1.0
        self.t = self.start_idx
        self.done = False
        return self._get_state()

    def _get_state(self):
        """Build (N_POSITIONS x T) state matrix."""
        state = np.zeros((N_POSITIONS, T), dtype=np.float32)

        # Asset rows: [weight, 59 days of z-scored returns]
        for i in range(N_ASSETS):
            state[i, 0] = self.weights[i]
            window = ret_arr[self.t - T + 1 : self.t, i]  # 59 days
            state[i, 1:] = window

        # Cash row: [weight_cash, vol features as time series]
        state[N_ASSETS, 0] = self.weights[-1]
        # Fill with interleaved vol features: 20 vol_20, 20 vol_ratio, 19 vix
        v20 = vol20_arr[self.t - 20 : self.t]
        vr  = volr_arr[self.t - 20 : self.t]
        vx  = vix_arr[self.t - 19 : self.t]
        state[N_ASSETS, 1:21]  = v20
        state[N_ASSETS, 21:41] = vr
        state[N_ASSETS, 41:60] = vx

        return state.flatten()

    def step(self, action_weights):
        """
        action_weights: np.array of shape (N_POSITIONS,), already softmax'd
        Returns: (next_state, reward, done, info)
        """
        # Portfolio return using raw (non-standardized) returns
        raw_rets = raw_ret_arr[self.t]  # today's returns for each asset
        port_ret = 0.0
        for i in range(N_ASSETS):
            port_ret += self.weights[i] * raw_rets[i]
        # Cash earns 0

        # Transaction cost
        turnover = np.sum(np.abs(action_weights - self.weights))
        tc = turnover * TC_BPS * 1e-4
        net_ret = port_ret - tc

        # DSR reward
        reward = self.dsr.step(net_ret)

        # Update weights
        self.weights = action_weights.copy()
        self.t += 1
        if self.t >= self.end_idx:
            self.done = True

        next_state = self._get_state() if not self.done else np.zeros(self.state_dim, dtype=np.float32)

        return next_state, reward, self.done, {
            "port_ret": net_ret,
            "raw_ret": port_ret,
            "turnover": turnover,
            "copper_weight": action_weights[COPPER_IDX],
        }

STATE_DIM = N_POSITIONS * T
ACTION_DIM = N_POSITIONS
print(f"  State dim: {N_POSITIONS} x {T} = {STATE_DIM}")
print(f"  Action dim: {ACTION_DIM} (softmax weights)")

# ======================================================================
# 5. PPO NETWORKS (PAPER ARCHITECTURE)
# ======================================================================
print("\n" + "=" * 70)
print("  STEP 5: PPO Networks")
print("=" * 70)

class ActorNetwork(nn.Module):
    """
    Continuous actor for portfolio allocation.
    Outputs raw logits -> softmax externally to get weights.
    Plus learned log_std for exploration.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_dist(self, x):
        mean, std = self.forward(x)
        return Normal(mean, std)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

print(f"  Actor: [{STATE_DIM}]->{HIDDEN_DIM}->tanh->{HIDDEN_DIM}->tanh->{ACTION_DIM}")
print(f"  Critic: [{STATE_DIM}]->{HIDDEN_DIM}->tanh->{HIDDEN_DIM}->tanh->1")

# ======================================================================
# 6. PPO AGENT WITH GAE
# ======================================================================
print("\n" + "=" * 70)
print("  STEP 6: PPO Agent (GAE, softmax actions)")
print("=" * 70)

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def add(self, state, action_logits, log_prob, reward, value, done):
        self.states.append(state)
        self.action_logits.append(action_logits)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, last_value):
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else self.values[t + 1]
            mask = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + GAMMA * next_val * mask - self.values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * mask * last_gae
            advantages[t] = last_gae
        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def get_tensors(self, advantages, returns):
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.action_logits))
        old_lp = torch.FloatTensor(np.array(self.log_probs))
        adv = torch.FloatTensor(advantages)
        ret = torch.FloatTensor(returns)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return states, actions, old_lp, adv, ret

    def clear(self):
        self.states = []
        self.action_logits = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def __len__(self):
        return len(self.states)


def softmax_np(x):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


class PPOAgent:
    def __init__(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.actor = ActorNetwork(STATE_DIM, ACTION_DIM)
        self.critic = CriticNetwork(STATE_DIM)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=LR_INIT)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR_INIT)
        self.training_rewards = []
        self.training_sharpes = []
        self.step_count = 0
        self.total_steps = 0

    def _anneal_lr(self, progress):
        """Linear annealing from LR_INIT to LR_FINAL."""
        lr = LR_INIT + (LR_FINAL - LR_INIT) * progress
        for pg in self.actor_opt.param_groups:
            pg['lr'] = lr
        for pg in self.critic_opt.param_groups:
            pg['lr'] = lr

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            mean, std = self.actor(s)
            if deterministic:
                logits = mean.squeeze(0).numpy()
            else:
                dist = Normal(mean, std)
                sample = dist.sample()
                logits = sample.squeeze(0).numpy()

            # Softmax to get portfolio weights
            weights = softmax_np(logits)

            # Log prob of the sampled logits under the distribution
            dist_eval = Normal(mean, std)
            log_prob = dist_eval.log_prob(torch.FloatTensor(logits).unsqueeze(0)).sum(dim=-1).item()
            value = self.critic(s).item()

        return weights, logits, log_prob, value

    def update(self, buffer):
        """PPO-Clip update with paper's 16 epochs."""
        with torch.no_grad():
            last_state = torch.FloatTensor(buffer.states[-1]).unsqueeze(0)
            last_value = self.critic(last_state).item()

        advantages, returns = buffer.compute_gae(last_value)
        states, actions, old_lp, adv, ret = buffer.get_tensors(advantages, returns)

        n = len(states)
        bs = min(BATCH_SIZE, n)
        total_pg = 0; total_vf = 0; total_ent = 0; n_up = 0

        for _ in range(PPO_EPOCHS):
            indices = np.random.permutation(n)
            for start in range(0, n, bs):
                end = min(start + bs, n)
                idx = indices[start:end]

                s_b = states[idx]
                a_b = actions[idx]
                olp_b = old_lp[idx]
                adv_b = adv[idx]
                ret_b = ret[idx]

                dist = self.actor.get_dist(s_b)
                new_lp = dist.log_prob(a_b).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = (new_lp - olp_b).exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * adv_b
                pg_loss = -torch.min(surr1, surr2).mean()

                values = self.critic(s_b)
                vf_loss = nn.MSELoss()(values, ret_b)

                loss = pg_loss + VF_COEF * vf_loss - ENTROPY_COEF * entropy

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD)
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD)
                self.actor_opt.step()
                self.critic_opt.step()

                total_pg += pg_loss.item()
                total_vf += vf_loss.item()
                total_ent += entropy.item()
                n_up += 1

        return {
            "pg_loss": total_pg / max(n_up, 1),
            "vf_loss": total_vf / max(n_up, 1),
            "entropy": total_ent / max(n_up, 1),
        }

print("  PPO agent ready (softmax action, GAE, LR annealing)")

# ======================================================================
# 7. TRAINING
# ======================================================================
print("\n" + "=" * 70)
print("  STEP 7: Training PPO Agents")
print("=" * 70)

def compute_sharpe(returns_list):
    r = np.array(returns_list)
    r = r[np.isfinite(r)]
    if len(r) < 21 or np.std(r) == 0:
        return 0.0
    return (np.mean(r) * 252) / (np.std(r) * np.sqrt(252))


def train_agent(seed, verbose=True):
    agent = PPOAgent(seed=seed)
    env = MultiAssetPortfolioEnv(T, train_end)
    total_train_steps = N_EPISODES * (train_end - T)

    for ep in range(N_EPISODES):
        state = env.reset()
        buffer = RolloutBuffer()
        ep_returns = []
        ep_copper_w = []

        # Anneal LR based on progress
        progress = ep / N_EPISODES
        agent._anneal_lr(progress)

        while not env.done:
            weights, logits, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(weights)
            buffer.add(state, logits, log_prob, reward, value, done)
            ep_returns.append(info["port_ret"])
            ep_copper_w.append(info["copper_weight"])
            state = next_state

            # Update every N_STEPS or at episode end
            if len(buffer) >= N_STEPS or done:
                metrics = agent.update(buffer)
                buffer.clear()

        ep_sharpe = compute_sharpe(ep_returns)
        ep_ret = np.sum(ep_returns)
        agent.training_rewards.append(ep_ret)
        agent.training_sharpes.append(ep_sharpe)

        if verbose and (ep + 1) % 10 == 0:
            avg_cu_w = np.mean(ep_copper_w)
            print(f"    Seed {seed} | Ep {ep+1:3d}/{N_EPISODES} | "
                  f"Ret={ep_ret:+.4f} | Sharpe={ep_sharpe:+.2f} | "
                  f"Cu_w={avg_cu_w:.3f} | LR={agent.actor_opt.param_groups[0]['lr']:.1e}")

    return agent


print(f"\n  Training {N_SEEDS} seeds x {N_EPISODES} episodes...")
t0 = time.time()
agents = []
for i in range(N_SEEDS):
    seed = 42 + i * 100
    print(f"\n  --- Seed {seed} ---")
    agent = train_agent(seed, verbose=True)
    agents.append(agent)

# Select best by tail Sharpe
tail = max(1, N_EPISODES // 4)
best_idx = 0
best_tail = -999
for i, ag in enumerate(agents):
    avg = np.mean(ag.training_sharpes[-tail:])
    print(f"  Seed {i}: tail avg Sharpe = {avg:.3f}")
    if avg > best_tail:
        best_tail = avg
        best_idx = i

elapsed = time.time() - t0
print(f"\n  Best seed: {best_idx} (tail Sharpe={best_tail:.3f}) | Time: {elapsed:.1f}s")
best_agent = agents[best_idx]

# ======================================================================
# 8. EVALUATION
# ======================================================================
print("\n" + "=" * 70)
print("  STEP 8: Evaluation")
print("=" * 70)

def evaluate_agent(agent, start_idx, end_idx):
    """Run agent deterministically, return weight history and returns."""
    env = MultiAssetPortfolioEnv(start_idx, end_idx)
    state = env.reset()
    all_weights = []
    all_returns = []
    all_dates = []

    while not env.done:
        weights, _, _, _ = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(weights)
        all_weights.append(weights.copy())
        all_returns.append(info["port_ret"])
        all_dates.append(dates[env.t - 1])
        state = next_state

    weight_df = pd.DataFrame(all_weights, index=all_dates,
                              columns=list(TRADE_ASSETS) + ["Cash"])
    ret_series = pd.Series(all_returns, index=all_dates, name="port_ret")
    return weight_df, ret_series


# Evaluate on full period
print("  Evaluating on full period (train+val+test)...")
weight_df, port_ret = evaluate_agent(best_agent, T, test_end)
print(f"  Portfolio return series: {len(port_ret)} days")
print(f"  Mean copper allocation: {weight_df['HG=F'].mean():.3f}")
print(f"  Mean cash allocation: {weight_df['Cash'].mean():.3f}")

# Also evaluate each seed for robustness
all_seed_results = []
for i, ag in enumerate(agents):
    _, sr = evaluate_agent(ag, T, test_end)
    all_seed_results.append(sr)


def compute_strategy_metrics(ret_series, period_mask_fn, label=""):
    r = ret_series.dropna()
    r = r[period_mask_fn(r.index)]
    if len(r) < 21 or r.std() == 0:
        return dict(sharpe=0, ann_ret=0, ann_vol=0, max_dd=0, hit=0,
                    cum=pd.Series(dtype=float), dd=pd.Series(dtype=float), daily_ret=r)
    ar = r.mean() * 252
    av = r.std() * np.sqrt(252)
    sh = ar / av if av > 0 else 0
    cum = (1 + r).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    md = dd.min()
    hit = (r > 0).mean() * 100
    return dict(sharpe=sh, ann_ret=ar, ann_vol=av, max_dd=md, hit=hit, cum=cum, dd=dd, daily_ret=r)


train_period = lambda idx: idx < SPLIT_TEST
test_period  = lambda idx: idx >= SPLIT_TEST

# --- Baselines ---
# 1/N equal weight (rebalanced daily)
eq_ret = raw_ret_arr.mean(axis=1)  # average across all assets
eq_ret_series = pd.Series(eq_ret, index=dates, name="eq_weight")

# Buy & hold copper only
copper_ret = pd.Series(raw_ret_arr[:, COPPER_IDX], index=dates, name="copper_bh")

# SPY buy & hold
spy_idx = TRADE_ASSETS.index("SPY")
spy_ret = pd.Series(raw_ret_arr[:, spy_idx], index=dates, name="spy_bh")

# Build results dict
strategies = {
    "PPO_Portfolio": port_ret,
    "EqualWeight_12": eq_ret_series,
    "Copper_BuyHold": copper_ret,
    "SPY_BuyHold": spy_ret,
}

# Add individual seed PPO results
for i, sr in enumerate(all_seed_results):
    strategies[f"PPO_Seed{i}"] = sr

results_all = {}
print(f"\n  {'Strategy':<20} {'Train Sh':>10} {'Test Sh':>10} {'Test Ret':>10} {'Test DD':>10} {'Test Hit':>10}")
print(f"  {'-'*72}")

display_strats = ["PPO_Portfolio", "EqualWeight_12", "Copper_BuyHold", "SPY_BuyHold"]
for name in display_strats:
    r = strategies[name]
    m_tr = compute_strategy_metrics(r, train_period, name)
    m_te = compute_strategy_metrics(r, test_period, name)
    results_all[name] = {"train": m_tr, "test": m_te}
    print(f"  {name:<20} {m_tr['sharpe']:>+10.2f} {m_te['sharpe']:>+10.2f} "
          f"{m_te['ann_ret']*100:>+9.1f}% {m_te['max_dd']*100:>+9.1f}% {m_te['hit']:>9.1f}%")

# Seed breakdown
print(f"\n  --- Per-Seed Test Sharpe ---")
for i, sr in enumerate(all_seed_results):
    m = compute_strategy_metrics(sr, test_period)
    marker = " <-- best" if i == best_idx else ""
    print(f"    Seed {i}: Sharpe={m['sharpe']:+.2f}, Ret={m['ann_ret']*100:+.1f}%, DD={m['max_dd']*100:+.1f}%{marker}")
    results_all[f"PPO_Seed{i}"] = {"train": compute_strategy_metrics(sr, train_period), "test": m}

# --- Annual Sharpe ---
print(f"\n  --- Annual Sharpe (Test) ---")
print(f"  {'Year':>4}", end="")
for name in display_strats:
    print(f"  {name[:18]:>18}", end="")
print()

for yr in range(2021, 2026):
    print(f"  {yr:>4}", end="")
    for name in display_strats:
        r = strategies[name].dropna()
        yr_r = r[(r.index.year == yr) & (r.index >= SPLIT_TEST)]
        if len(yr_r) > 20 and yr_r.std() > 0:
            ysh = (yr_r.mean() * 252) / (yr_r.std() * np.sqrt(252))
        else:
            ysh = 0
        print(f"  {ysh:>+18.2f}", end="")
    print()

# --- Allocation analysis ---
print(f"\n  --- Average Asset Allocation (Test Period) ---")
test_weights = weight_df[weight_df.index >= SPLIT_TEST]
mean_alloc = test_weights.mean().sort_values(ascending=False)
for asset, w in mean_alloc.items():
    print(f"    {asset:12s}: {w*100:5.1f}%")

# ======================================================================
# 9. MASTER FIGURE
# ======================================================================
print("\n" + "=" * 70)
print("  STEP 9: Master Figure -> claude/rl_results.png")
print("=" * 70)

fig = plt.figure(figsize=(24, 34))
fig.suptitle(
    "Multi-Asset RL Portfolio: PPO + Differential Sharpe Ratio (arXiv 2602.17098)\n"
    f"Assets: {N_ASSETS}+cash | Hidden={HIDDEN_DIM} | Seeds={N_SEEDS} | Eps={N_EPISODES} | "
    f"gamma={GAMMA}, lambda={GAE_LAMBDA}, clip={PPO_CLIP}\n"
    f"Train 2005-2018 | Val 2019-2020 | Test 2021-2025",
    fontsize=12, fontweight="bold", y=0.998)

gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.42, wspace=0.30)

colors = {
    "PPO_Portfolio": "darkorange",
    "EqualWeight_12": "steelblue",
    "Copper_BuyHold": "forestgreen",
    "SPY_BuyHold": "crimson",
}

# --- Row 0: Training curves ---
# (0,0): All seeds - episode return
ax = fig.add_subplot(gs[0, 0])
for i, ag in enumerate(agents):
    lw = 2 if i == best_idx else 0.7
    alpha = 1.0 if i == best_idx else 0.4
    ax.plot(ag.training_rewards, lw=lw, alpha=alpha,
            label=f"Seed {i}" + (" *" if i == best_idx else ""))
ax.set_title("Training: Episode Portfolio Return", fontweight="bold", fontsize=10)
ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative Return")
ax.legend(fontsize=7); ax.axhline(0, color="grey", lw=0.5, ls="--")

# (0,1): All seeds - episode Sharpe
ax = fig.add_subplot(gs[0, 1])
for i, ag in enumerate(agents):
    lw = 2 if i == best_idx else 0.7
    alpha = 1.0 if i == best_idx else 0.4
    ax.plot(ag.training_sharpes, lw=lw, alpha=alpha,
            label=f"Seed {i}" + (" *" if i == best_idx else ""))
ax.set_title("Training: Episode Sharpe Ratio", fontweight="bold", fontsize=10)
ax.set_xlabel("Episode"); ax.set_ylabel("Sharpe")
ax.legend(fontsize=7); ax.axhline(0, color="grey", lw=0.5, ls="--")

# (0,2): Best seed Sharpe with rolling average
ax = fig.add_subplot(gs[0, 2])
sharpes = best_agent.training_sharpes
ax.plot(sharpes, color="steelblue", alpha=0.3, lw=0.8, label="Raw")
if len(sharpes) > 5:
    rolling = pd.Series(sharpes).rolling(5, min_periods=1).mean()
    ax.plot(rolling, color="darkorange", lw=2, label="5-ep MA")
ax.set_title(f"Best Seed (#{best_idx}): Sharpe Progression", fontweight="bold", fontsize=10)
ax.set_xlabel("Episode"); ax.set_ylabel("Sharpe")
ax.legend(fontsize=7); ax.axhline(0, color="grey", lw=0.5, ls="--")

# --- Row 1: Cumulative returns ---
for col, (period_fn, prd_name, prd_key) in enumerate([
    (train_period, "Train+Val (2005-2020)", "train"),
    (test_period, "Test (2021-2025)", "test"),
]):
    ax = fig.add_subplot(gs[1, col])
    for name in display_strats:
        m = results_all[name][prd_key]
        if len(m.get("cum", [])) > 0:
            (m["cum"] - 1).plot(ax=ax, color=colors.get(name, "grey"),
                                lw=2 if "PPO" in name else 1,
                                label=f"{name} ({m['sharpe']:.2f})")
    ax.set_title(f"{prd_name}: Cumulative Return", fontweight="bold", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    ax.legend(fontsize=7, loc="upper left"); ax.set_xlabel("")
    ax.axhline(0, color="grey", lw=0.5, ls="--")

# (1,2): All seeds test cumulative
ax = fig.add_subplot(gs[1, 2])
for i, sr in enumerate(all_seed_results):
    m = compute_strategy_metrics(sr, test_period)
    if len(m.get("cum", [])) > 0:
        lw = 2.5 if i == best_idx else 0.8
        (m["cum"] - 1).plot(ax=ax, lw=lw, alpha=0.9 if i == best_idx else 0.4,
                            label=f"Seed {i} ({m['sharpe']:.2f})" + (" *" if i == best_idx else ""))
# Add baselines
for bname in ["EqualWeight_12", "SPY_BuyHold"]:
    m = results_all[bname]["test"]
    if len(m.get("cum", [])) > 0:
        (m["cum"] - 1).plot(ax=ax, color=colors[bname], lw=1, ls="--", label=f"{bname} ({m['sharpe']:.2f})")
ax.set_title("Test: All Seeds vs Baselines", fontweight="bold", fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
ax.legend(fontsize=6); ax.set_xlabel("")

# --- Row 2: Drawdowns + Asset allocation ---
# (2,0): Drawdowns (test)
ax = fig.add_subplot(gs[2, 0])
for name in display_strats:
    m = results_all[name]["test"]
    if len(m.get("dd", [])) > 0:
        m["dd"].plot(ax=ax, color=colors.get(name, "grey"), lw=1.2, label=name)
ax.set_title("Test: Drawdowns", fontweight="bold", fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
ax.legend(fontsize=7); ax.set_xlabel("")

# (2,1): Asset allocation over time (test, stacked area)
ax = fig.add_subplot(gs[2, 1])
tw = test_weights
if len(tw) > 0:
    # Resample to weekly for cleaner plot
    tw_weekly = tw.resample("W").mean()
    asset_colors = plt.cm.tab20(np.linspace(0, 1, N_POSITIONS))
    ax.stackplot(tw_weekly.index, *[tw_weekly[c].values for c in tw_weekly.columns],
                 labels=tw_weekly.columns, colors=asset_colors, alpha=0.8)
    ax.set_ylim(0, 1)
ax.set_title("Test: Portfolio Allocation (weekly avg)", fontweight="bold", fontsize=10)
ax.set_ylabel("Weight"); ax.set_xlabel("")
ax.legend(fontsize=5, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)

# (2,2): Copper weight over time
ax = fig.add_subplot(gs[2, 2])
copper_w = weight_df["HG=F"]
test_cu = copper_w[copper_w.index >= SPLIT_TEST]
if len(test_cu) > 0:
    ax.fill_between(test_cu.index, test_cu.values, 0, color="darkorange", alpha=0.4)
    ax.plot(test_cu.index, test_cu.values, color="darkorange", lw=0.8)
    ax.axhline(test_cu.mean(), color="red", ls="--", lw=1,
               label=f"Mean={test_cu.mean()*100:.1f}%")
ax.set_title("Test: Copper (HG=F) Allocation", fontweight="bold", fontsize=10)
ax.set_ylabel("Weight"); ax.set_xlabel(""); ax.legend(fontsize=8)

# --- Row 3: Annual Sharpe bars + Allocation bar + Correlation ---
# (3,0): Annual Sharpe bars
ax = fig.add_subplot(gs[3, 0])
years = list(range(2021, 2026))
bar_w = 0.8 / len(display_strats)
for j, name in enumerate(display_strats):
    sharpes_yr = []
    for yr in years:
        r = strategies[name].dropna()
        yr_r = r[(r.index.year == yr) & (r.index >= SPLIT_TEST)]
        if len(yr_r) > 20 and yr_r.std() > 0:
            sharpes_yr.append((yr_r.mean() * 252) / (yr_r.std() * np.sqrt(252)))
        else:
            sharpes_yr.append(0)
    x_pos = np.arange(len(years)) + j * bar_w
    ax.bar(x_pos, sharpes_yr, width=bar_w, label=name[:18], color=colors.get(name, "grey"))
ax.set_xticks(np.arange(len(years)) + len(display_strats) * bar_w / 2)
ax.set_xticklabels(years)
ax.axhline(0, color="black", lw=0.8)
ax.axhline(1, color="gold", ls="--", lw=1, alpha=0.5)
ax.set_title("Annual Sharpe (Test)", fontweight="bold", fontsize=10)
ax.legend(fontsize=6)

# (3,1): Average allocation bar chart
ax = fig.add_subplot(gs[3, 1])
mean_w = test_weights.mean().sort_values(ascending=True)
bar_colors = ["darkorange" if a == "HG=F" else "steelblue" if a == "Cash" else "grey"
              for a in mean_w.index]
ax.barh(range(len(mean_w)), mean_w.values * 100, color=bar_colors, edgecolor="white")
ax.set_yticks(range(len(mean_w)))
ax.set_yticklabels(mean_w.index, fontsize=7)
ax.set_xlabel("Allocation (%)")
ax.set_title("Mean Allocation (Test)", fontweight="bold", fontsize=10)

# (3,2): Return correlation heatmap
ax = fig.add_subplot(gs[3, 2])
corr_data = {}
for name in display_strats:
    r = strategies[name].dropna()
    corr_data[name] = r[r.index >= SPLIT_TEST]
corr_df = pd.DataFrame(corr_data).dropna().corr()
im = ax.imshow(corr_df.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr_df)))
ax.set_yticks(range(len(corr_df)))
ax.set_xticklabels([s[:15] for s in corr_df.columns], fontsize=7, rotation=45, ha="right")
ax.set_yticklabels([s[:15] for s in corr_df.index], fontsize=7)
for i in range(len(corr_df)):
    for j in range(len(corr_df)):
        ax.text(j, i, f"{corr_df.values[i,j]:.2f}", ha="center", va="center", fontsize=6)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Return Correlations (Test)", fontweight="bold", fontsize=9)

# --- Row 4: Per-asset weight time series (top 6) ---
top6 = test_weights.mean().sort_values(ascending=False).head(6).index.tolist()
for j, asset in enumerate(top6):
    row_idx = 4 if j < 3 else 5
    col_idx = j % 3
    ax = fig.add_subplot(gs[row_idx, col_idx])
    w = weight_df[asset]
    tw = w[w.index >= SPLIT_TEST]
    if len(tw) > 0:
        ax.fill_between(tw.index, tw.values, 0, alpha=0.4, color="steelblue")
        ax.plot(tw.index, tw.values, color="steelblue", lw=0.5)
        ax.axhline(tw.mean(), color="red", ls="--", lw=1, label=f"Mean={tw.mean()*100:.1f}%")
    ax.set_title(f"{asset} Weight (Test)", fontweight="bold", fontsize=9)
    ax.set_ylabel("Weight"); ax.legend(fontsize=7)

# Fill remaining panels in row 5 with summary text
remaining = 3 - (len(top6) - 3) if len(top6) > 3 else 3
if remaining > 0 and len(top6) <= 3:
    # Row 5 panels available for text
    pass

# Summary text panels in remaining row 5 slots
n_row5_used = max(0, len(top6) - 3)
for j in range(n_row5_used, 3):
    ax = fig.add_subplot(gs[5, j])
    ax.axis("off")
    if j == n_row5_used:
        # Strategy summary
        txt = "Strategy Summary (Test)\n" + "=" * 40
        txt += f"\n{'Strategy':<20} {'Sharpe':>7} {'Ret%':>7} {'DD%':>7}"
        txt += f"\n{'-'*42}"
        for name in display_strats:
            m = results_all[name]["test"]
            txt += f"\n{name:<20} {m['sharpe']:>+7.2f} {m['ann_ret']*100:>+6.1f}% {m['max_dd']*100:>+6.1f}%"
        txt += f"\n\nPaper (PPO, multi-asset):"
        txt += f"\n  Sharpe: 1.17 | Ret: 12.1% | DD: -33.0%"
        txt += f"\nPaper (MVO baseline):"
        txt += f"\n  Sharpe: 0.68 | Ret: 6.5% | DD: -33.0%"
        ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    elif j == n_row5_used + 1:
        # Hyperparameters
        txt = "Hyperparameters\n" + "=" * 30
        txt += f"\nReward: DSR (eta=1/252)"
        txt += f"\nAlgorithm: PPO-Clip"
        txt += f"\n  gamma={GAMMA}, lambda={GAE_LAMBDA}"
        txt += f"\n  clip={PPO_CLIP}, epochs={PPO_EPOCHS}"
        txt += f"\n  LR: {LR_INIT}->{LR_FINAL} (annealed)"
        txt += f"\n  entropy={ENTROPY_COEF}, vf={VF_COEF}"
        txt += f"\n  grad_clip={MAX_GRAD}"
        txt += f"\nNetwork: [{HIDDEN_DIM},{HIDDEN_DIM}] MLP+tanh"
        txt += f"\nAction: softmax (long-only)"
        txt += f"\nTC: {TC_BPS}bps"
        txt += f"\nSeeds: {N_SEEDS}, Episodes: {N_EPISODES}"
        txt += f"\nRef: arXiv 2602.17098"
        ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="honeydew", alpha=0.8))
    elif j == n_row5_used + 2:
        # Allocation summary
        txt = "Mean Allocation (Test)\n" + "=" * 30
        for asset, w in mean_alloc.head(13).items():
            bar = "#" * int(w * 50)
            txt += f"\n{asset:10s} {w*100:5.1f}% {bar}"
        ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                fontsize=7, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8))

out_path = f"{BASE}/claude/rl_results.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nPlot saved: {out_path}")

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)
