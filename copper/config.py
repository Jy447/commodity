import os
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

BASE  = "C:/Users/sunAr/Documents/sunArise/quant/commodity"
SPLIT_VAL  = "2019-01-01"
SPLIT_TEST = "2021-01-01"
TARGET_VOL = 0.20
COPPER     = "HG=F"
SEQ_LEN    = 20
HIDDEN     = 64
N_LAYERS   = 2
DROPOUT    = 0.2
BATCH_SIZE = 64
LR_RATE    = 0.001
MAX_EPOCHS = 150
PATIENCE   = 10
CORR_THRESH = 0.10
MISSING_THRESH = 0.30
ZSCORE_WINDOW = 252

# Set USE_PARQUET=True to skip CSV+AlphaLib and load from precomputed parquet
USE_PARQUET = True

# Horizons to model (dropped log_ret5)
HORIZONS = {
    "log_ret20": 20,
    "log_ret30": 30,
    "log_ret60": 60,
}

# All horizons (for distribution analysis)
ALL_HORIZONS = {
    "log_ret1":  1,
    "log_ret5":  5,
    "log_ret20": 20,
    "log_ret30": 30,
    "log_ret60": 60,
}

np.random.seed(42)
torch.manual_seed(42)
