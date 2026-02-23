import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import (HORIZONS, SPLIT_VAL, SPLIT_TEST, SEQ_LEN, HIDDEN,
                    N_LAYERS, DROPOUT, BATCH_SIZE, LR_RATE, MAX_EPOCHS, PATIENCE)


@dataclass
class LRResults:
    lr_models: dict
    lr_predictions: dict
    lr_metrics: dict
    lr_feature_sets: dict


@dataclass
class LSTMResults:
    lstm_models: dict
    lstm_predictions: dict
    lstm_training_curves: dict
    lstm_feature_sets: dict
    lstm_norm_stats: dict
    lstm_eval_metrics: dict


def eval_metrics(pred, actual, label, verbose=True):
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    dir_acc = np.mean(np.sign(pred) == np.sign(actual)) * 100
    ic, _ = stats.spearmanr(pred, actual)
    if verbose:
        print(f"  {label:<16s}  MSE={mse:.6f}  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  DirAcc={dir_acc:.1f}%  IC={ic:.3f}")
    return dict(mse=mse, rmse=rmse, mae=mae, r2=r2, dir_acc=dir_acc, ic=ic)


class CopperDataset(Dataset):
    def __init__(self, features, targets, seq_len):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.features) - self.seq_len
    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return x, y


class CopperLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


def predict_dataset(model, dataloader, device):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            preds.extend(pred.cpu().numpy())
            actuals.extend(y_batch.numpy())
    return np.array(preds), np.array(actuals)


def train_linear_regression(pq, selected_features):
    print("\n" + "=" * 70)
    print("  STEP 5: Linear Regression Baseline (per horizon)")
    print("=" * 70)

    lr_models = {}
    lr_predictions = {}
    lr_metrics = {}
    lr_feature_sets = {}

    for target_name in HORIZONS.keys():
        print(f"\n--- {target_name} ---")

        horizon_feats = [f for f in selected_features[target_name].index.tolist()
                         if not f.startswith("log_ret")]
        if len(horizon_feats) < 3:
            all_sel = set()
            for v in selected_features.values():
                all_sel.update(v.index.tolist())
            horizon_feats = sorted([f for f in all_sel if not f.startswith("log_ret")])
        print(f"  Features: {len(horizon_feats)}")
        lr_feature_sets[target_name] = horizon_feats

        df_lr = pq[horizon_feats + [target_name]].dropna(subset=[target_name])
        df_lr_feats = df_lr[horizon_feats].fillna(0)
        df_lr_target = df_lr[target_name]

        tr_mask = df_lr.index < SPLIT_VAL
        va_mask = (df_lr.index >= SPLIT_VAL) & (df_lr.index < SPLIT_TEST)
        te_mask = df_lr.index >= SPLIT_TEST

        X_train = df_lr_feats[tr_mask].values
        y_train = df_lr_target[tr_mask].values
        X_val = df_lr_feats[va_mask].values
        y_val = df_lr_target[va_mask].values
        X_test = df_lr_feats[te_mask].values
        y_test = df_lr_target[te_mask].values

        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_models[target_name] = lr_model

        pred_tr = lr_model.predict(X_train)
        pred_va = lr_model.predict(X_val)
        pred_te = lr_model.predict(X_test)

        lr_predictions[target_name] = {
            "train": pd.Series(pred_tr, index=df_lr.index[tr_mask]),
            "val":   pd.Series(pred_va, index=df_lr.index[va_mask]),
            "test":  pd.Series(pred_te, index=df_lr.index[te_mask]),
        }

        lr_metrics[target_name] = {}
        lr_metrics[target_name]["train"] = eval_metrics(pred_tr, y_train, "Train")
        lr_metrics[target_name]["val"]   = eval_metrics(pred_va, y_val, "Val")
        lr_metrics[target_name]["test"]  = eval_metrics(pred_te, y_test, "Test")

    return LRResults(
        lr_models=lr_models,
        lr_predictions=lr_predictions,
        lr_metrics=lr_metrics,
        lr_feature_sets=lr_feature_sets)


def train_lstm(pq, lr_feature_sets):
    print("\n" + "=" * 70)
    print("  STEP 6: LSTM Model Training (per horizon)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    lstm_models = {}
    lstm_predictions = {}
    lstm_metrics = {}
    lstm_training_curves = {}
    lstm_feature_sets = {}
    lstm_norm_stats = {}

    for target_name in HORIZONS.keys():
        print(f"\n{'='*50}")
        print(f"  Training LSTM for {target_name}")
        print(f"{'='*50}")

        target_z = f"{target_name}_zscore"

        lstm_features = lr_feature_sets[target_name]
        lstm_feature_sets[target_name] = lstm_features
        print(f"  Features: {len(lstm_features)}")

        df_model = pq[lstm_features + [target_name, target_z]].copy()
        df_model = df_model.dropna(subset=[target_z])

        train_end_idx = df_model.index < SPLIT_VAL
        train_feat = df_model.loc[train_end_idx, lstm_features]
        feat_mean = train_feat.mean()
        feat_std = train_feat.std().replace(0, 1)
        lstm_norm_stats[target_name] = (feat_mean, feat_std)

        df_norm = df_model.copy()
        df_norm[lstm_features] = (df_model[lstm_features] - feat_mean) / feat_std
        df_norm[lstm_features] = df_norm[lstm_features].fillna(0).clip(-5, 5)

        dates = df_norm.index
        train_idx = dates < SPLIT_VAL
        val_idx = (dates >= SPLIT_VAL) & (dates < SPLIT_TEST)
        test_idx = dates >= SPLIT_TEST

        X_all = df_norm[lstm_features].values
        y_all_z = df_norm[target_z].values
        y_all_raw = df_norm[target_name].values

        train_ds = CopperDataset(X_all[train_idx], y_all_z[train_idx], SEQ_LEN)
        val_ds = CopperDataset(X_all[val_idx], y_all_z[val_idx], SEQ_LEN)
        test_ds = CopperDataset(X_all[test_idx], y_all_z[test_idx], SEQ_LEN)

        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} sequences")

        torch.manual_seed(42)
        model = CopperLSTM(len(lstm_features), HIDDEN, N_LAYERS, DROPOUT).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        print(f"  Training (max {MAX_EPOCHS} epochs, patience {PATIENCE})...")
        for epoch in range(MAX_EPOCHS):
            model.train()
            epoch_loss = 0; n_batches = 0
            for X_batch, y_batch in train_dl:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item(); n_batches += 1
            train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0; n_val = 0
            with torch.no_grad():
                for X_batch, y_batch in val_dl:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    pred = model(X_batch)
                    val_loss += criterion(pred, y_batch).item(); n_val += 1
            val_loss = val_loss / max(n_val, 1)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss; patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or patience_counter == 0:
                print(f"    Epoch {epoch+1:3d}  train={train_loss:.6f}  val={val_loss:.6f}  {'*' if patience_counter == 0 else ''}")

            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        print(f"  Best val loss: {best_val_loss:.6f} ({len(train_losses)} epochs)")

        lstm_models[target_name] = model
        lstm_training_curves[target_name] = (train_losses, val_losses)

        pred_train_z, actual_train_z = predict_dataset(model, DataLoader(train_ds, batch_size=256, shuffle=False), device)
        pred_val_z, actual_val_z = predict_dataset(model, DataLoader(val_ds, batch_size=256, shuffle=False), device)
        pred_test_z, actual_test_z = predict_dataset(model, DataLoader(test_ds, batch_size=256, shuffle=False), device)

        train_dates = dates[train_idx][SEQ_LEN:]
        val_dates = dates[val_idx][SEQ_LEN:]
        test_dates = dates[test_idx][SEQ_LEN:]

        lstm_predictions[target_name] = {
            "train_z": pd.Series(pred_train_z, index=train_dates[:len(pred_train_z)]),
            "val_z":   pd.Series(pred_val_z, index=val_dates[:len(pred_val_z)]),
            "test_z":  pd.Series(pred_test_z, index=test_dates[:len(pred_test_z)]),
            "actual_train_z": actual_train_z,
            "actual_val_z":   actual_val_z,
            "actual_test_z":  actual_test_z,
            "raw_train": y_all_raw[train_idx][SEQ_LEN:],
            "raw_val":   y_all_raw[val_idx][SEQ_LEN:],
            "raw_test":  y_all_raw[test_idx][SEQ_LEN:],
        }

    # ======================================================================
    # 7. MODEL EVALUATION -- LR vs LSTM COMPARISON
    # ======================================================================
    print("\n" + "=" * 70)
    print("  STEP 7: Model Evaluation -- LR vs LSTM Comparison")
    print("=" * 70)

    lstm_eval_metrics = {}

    for target_name in HORIZONS.keys():
        print(f"\n--- {target_name} ---")
        preds = lstm_predictions[target_name]

        print("  [LSTM - Z-score domain]")
        lstm_eval_metrics[target_name] = {}
        lstm_eval_metrics[target_name]["train_z"] = eval_metrics(
            preds["train_z"].values, preds["actual_train_z"], "Train(z)")
        lstm_eval_metrics[target_name]["val_z"] = eval_metrics(
            preds["val_z"].values, preds["actual_val_z"], "Val(z)")
        lstm_eval_metrics[target_name]["test_z"] = eval_metrics(
            preds["test_z"].values, preds["actual_test_z"], "Test(z)")

        print("  [LSTM - Direction vs raw returns]")
        for label, prd, raw_act in [("Train", preds["train_z"].values, preds["raw_train"]),
                                     ("Val", preds["val_z"].values, preds["raw_val"]),
                                     ("Test", preds["test_z"].values, preds["raw_test"])]:
            n = min(len(prd), len(raw_act))
            da = np.mean(np.sign(prd[:n]) == np.sign(raw_act[:n])) * 100
            ic_raw, _ = stats.spearmanr(prd[:n], raw_act[:n])
            print(f"  {label:<16s}  DirAcc(raw)={da:.1f}%  IC(raw)={ic_raw:.3f}")

    return LSTMResults(
        lstm_models=lstm_models,
        lstm_predictions=lstm_predictions,
        lstm_training_curves=lstm_training_curves,
        lstm_feature_sets=lstm_feature_sets,
        lstm_norm_stats=lstm_norm_stats,
        lstm_eval_metrics=lstm_eval_metrics)


def print_model_comparison(lr_metrics, lstm_eval_metrics):
    # Comparison table
    print("\n" + "=" * 50)
    print("  LR vs LSTM Comparison (Test Set)")
    print("=" * 50)
    print(f"  {'Horizon':<12} {'Model':<6} {'MSE':>10} {'R2':>8} {'DirAcc':>8} {'IC':>8}")
    print(f"  {'-'*52}")

    for target_name in HORIZONS.keys():
        lr_m = lr_metrics[target_name]["test"]
        print(f"  {target_name:<12} {'LR':<6} {lr_m['mse']:>10.6f} {lr_m['r2']:>8.4f} {lr_m['dir_acc']:>7.1f}% {lr_m['ic']:>8.3f}")
        lstm_m = lstm_eval_metrics[target_name]["test_z"]
        print(f"  {'':<12} {'LSTM':<6} {lstm_m['mse']:>10.6f} {lstm_m['r2']:>8.4f} {lstm_m['dir_acc']:>7.1f}% {lstm_m['ic']:>8.3f}")
