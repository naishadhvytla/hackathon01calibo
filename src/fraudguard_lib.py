"""
fraudguard_lib.py — shared, tested building blocks for the FraudGuard pipeline.

Every function here is dataset-agnostic, so the credit-card (primary) and the
UPI (negative-control) runs go through exactly the same code path.

Sections
  1. Config & reproducibility
  2. Loading + data-quality audit (per dataset)
  3. Time-aware split
  4. Metrics, cost model, threshold + band optimisation, bootstrap CIs
  5. Model zoo (classical, GBDT, imbalance-ensemble, deep tabular)
  6. Label-free anomaly detectors
  7. Plot style
"""
from __future__ import annotations

import os, random, time, warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Config & reproducibility
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42


@dataclass
class CostConfig:
    """Rupee cost assumptions (ADR-006). Edit here — everything downstream updates.

    X  : cost of a fraud that is approved (missed fraud)
    Y  : cost of a genuine transaction that is blocked (friction, churn risk, support call)
    Z  : cost of sending ANY transaction to step-up OTP (SMS + abandonment friction)
    otp_stop_rate : share of frauds that fail the OTP challenge (stolen-credential / ATO fraud
                    fails; social-engineering fraud where the victim types the OTP passes)
    """
    X: float = 5000.0
    Y: float = 300.0
    Z: float = 15.0
    otp_stop_rate: float = 0.70


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# 2. Loading + data-quality audit
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Dataset:
    name: str
    df: pd.DataFrame                 # cleaned frame, time-sorted, includes target + EDA cols
    target: str
    time_col: str                    # monotone time key used for the split
    num_features: list
    cat_features: list
    audit: pd.DataFrame              # one row per issue: issue, finding, treatment
    excluded: pd.DataFrame           # columns excluded and why
    amount_col: str = "amount"
    hour_col: str = "hour"

    @property
    def features(self):
        return self.num_features + self.cat_features


def featurize_creditcard(df: pd.DataFrame) -> pd.DataFrame:
    """Raw ULB columns (Time, V1..V28, Amount) -> model features. Shared by notebook and app."""
    df = df.copy()
    df["amount"] = df["Amount"]
    df["log_amount"] = np.log1p(df["Amount"])
    df["amount_zero"] = (df["Amount"] == 0).astype(int)
    df["hour"] = ((df["Time"] // 3600) % 24).astype(int)
    df["day"] = (df["Time"] // 86400).astype(int) + 1
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def load_creditcard(path: str) -> Dataset:
    raw = pd.read_csv(path)
    rows = []
    n0 = len(raw)
    rows.append(("Missing values", f"{int(raw.isna().sum().sum())} missing cells across {raw.shape[1]} columns",
                 "None needed"))
    dup_mask = raw.duplicated()
    rows.append(("Exact duplicate rows", f"{int(dup_mask.sum())} rows ({int(raw.loc[dup_mask,'Class'].sum())} labelled fraud)",
                 "Dropped (keep first) — duplicates would inflate fraud memorisation and can straddle the split"))
    df = raw.loc[~dup_mask].copy()
    neg = int((df.Amount < 0).sum())
    zero = int((df.Amount == 0).sum()); zero_f = int(df.loc[df.Amount == 0, "Class"].sum())
    rows.append(("Impossible amounts", f"{neg} negative amounts; {zero} zero-amount rows ({zero_f} fraud)",
                 "Kept zero amounts and added an `amount_zero` flag — zero-amount authorisations are a known card-testing pattern, not an error"))
    rows.append(("Timestamp / time step", f"`Time` = seconds since first txn, range 0–{int(df.Time.max())} s (~2 days), "
                 f"monotonic={raw.Time.is_monotonic_increasing}",
                 "Derived hour-of-day assuming the capture started at 00:00 (low-volume trough at 01–06 supports this); "
                 "raw `Time` dropped as a feature because absolute time does not generalise to future days"))
    rows.append(("Inconsistent categories", "No categorical columns (V1–V28 are anonymised PCA components)",
                 "Not applicable"))
    rows.append(("Leakage columns", "No balance / post-outcome columns; `Class` is the only label-derived field",
                 "Only `Class` excluded from features"))
    audit = pd.DataFrame(rows, columns=["Check", "Finding", "Treatment"])

    df = featurize_creditcard(df.sort_values("Time", kind="stable").reset_index(drop=True))
    vcols = [f"V{i}" for i in range(1, 29)]
    excluded = pd.DataFrame([
        ("Class", "Target label"),
        ("Time", "Absolute timestamp — replaced by cyclical hour features; would encode the day, not behaviour"),
        ("Amount", "Replaced by log_amount (same information, less skew for linear/deep models)"),
    ], columns=["Column", "Reason"])
    return Dataset("creditcard", df, "Class", "Time",
                   vcols + ["log_amount", "amount_zero", "hour_sin", "hour_cos"], [], audit, excluded)


UPI_CATS = ["transaction type", "merchant_category", "sender_age_group", "receiver_age_group",
            "sender_state", "sender_bank", "receiver_bank", "device_type", "network_type", "day_of_week"]


def load_upi(path: str) -> Dataset:
    raw = pd.read_csv(path)
    rows = []
    rows.append(("Missing values", f"{int(raw.isna().sum().sum())} missing cells", "None needed"))
    d_all = int(raw.duplicated().sum()); d_noid = int(raw.drop(columns=["transaction id"]).duplicated().sum())
    rows.append(("Duplicates", f"{d_all} exact duplicates; {d_noid} duplicates ignoring the ID; "
                 f"{int(raw['transaction id'].duplicated().sum())} duplicate IDs", "None to drop"))
    amt = raw["amount (INR)"]
    rows.append(("Impossible amounts", f"min ₹{amt.min()}, max ₹{amt.max():,}; {(amt<=0).sum()} non-positive",
                 "None impossible; heavy right tail handled with log transform"))
    ts = pd.to_datetime(raw["timestamp"], errors="coerce")
    bad_ts = int(ts.isna().sum())
    hour_mismatch = int((ts.dt.hour != raw["hour_of_day"]).sum())
    dow_mismatch = int((ts.dt.day_name() != raw["day_of_week"]).sum())
    rows.append(("Malformed timestamps", f"{bad_ts} unparseable; hour_of_day mismatches={hour_mismatch}; "
                 f"day_of_week mismatches={dow_mismatch}; range {ts.min().date()} → {ts.max().date()}",
                 "Timestamps valid; derived columns consistent"))
    p2p_with_merchant = int(((raw["transaction type"] == "P2P") & raw["merchant_category"].notna()).sum())
    rows.append(("Inconsistent categories",
                 f"{p2p_with_merchant:,} P2P (person-to-person) rows carry a merchant category; Recharge rows "
                 f"appear under Education/Fuel/Healthcare; amount distribution is identical across all 4 types "
                 f"(median ₹618–632)",
                 "Retained but flagged: columns look independently simulated, so they cannot carry real fraud behaviour"))
    rows.append(("Leakage columns", "`transaction_status` (SUCCESS/FAILED) is set AFTER the payment is processed — "
                 "a fraud block would show as FAILED in production; `transaction id` is a row identifier",
                 "Both excluded from features"))
    rows.append(("Account behaviour", "No sender/receiver account ID or device ID",
                 "Velocity / new-payee / mule-account features cannot be built — main limitation"))
    audit = pd.DataFrame(rows, columns=["Check", "Finding", "Treatment"])

    df = raw.copy()
    df["ts"] = ts
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df["t"] = (df["ts"] - df["ts"].min()).dt.total_seconds()
    df["amount"] = df["amount (INR)"]
    df["log_amount"] = np.log1p(df["amount"])
    df["hour"] = df["hour_of_day"]
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month"] = df["ts"].dt.month
    excluded = pd.DataFrame([
        ("fraud_flag", "Target label"),
        ("transaction_status", "Post-outcome field — known only after the payment decision (leakage)"),
        ("transaction id", "Row identifier, no behavioural meaning"),
        ("timestamp", "Replaced by hour (cyclical), day_of_week, is_weekend"),
        ("amount (INR)", "Replaced by log_amount"),
    ], columns=["Column", "Reason"])
    return Dataset("upi", df, "fraud_flag", "t",
                   ["log_amount", "hour_sin", "hour_cos", "is_weekend"], UPI_CATS, audit, excluded)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Time-aware split
# ──────────────────────────────────────────────────────────────────────────────
def time_split(ds: Dataset, fracs=(0.6, 0.2, 0.2)):
    """Data is already time-sorted. Returns index arrays for train / val / test."""
    n = len(ds.df)
    a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    idx = np.arange(n)
    return idx[:a], idx[a:b], idx[b:]


def split_summary(ds: Dataset, splits) -> pd.DataFrame:
    y = ds.df[ds.target].values
    out = []
    for name, ix in zip(["train", "validation", "test"], splits):
        t = ds.df[ds.time_col].iloc[ix]
        out.append(dict(split=name, rows=len(ix), frauds=int(y[ix].sum()),
                        fraud_rate_pct=round(100 * y[ix].mean(), 4),
                        time_from=float(t.min()), time_to=float(t.max())))
    return pd.DataFrame(out)


def make_preprocessor(ds: Dataset, scale=True):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num = ("num", StandardScaler() if scale else "passthrough", ds.num_features)
    tr = [num]
    if ds.cat_features:
        tr.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ds.cat_features))
    return ColumnTransformer(tr, verbose_feature_names_out=False)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Metrics, cost model, thresholds, bands, bootstrap
# ──────────────────────────────────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), 0)
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h


def binary_metrics(y, score, thr, cost: CostConfig | None = None) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
    pred = (score >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    d = dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
             recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
             pr_auc=average_precision_score(y, score) if len(np.unique(score)) > 1 else y.mean(),
             roc_auc=roc_auc_score(y, score) if len(np.unique(score)) > 1 else 0.5,
             tp=tp, fp=fp, fn=fn, tn=tn)
    if cost is not None:
        d["cost_inr"] = fn * cost.X + fp * cost.Y
    return d


def cost_curve(y, score, cost: CostConfig, n_points: int | None = None):
    """Exact expected cost FN·X + FP·Y at every distinct threshold (vectorised).

    Returns DataFrame [threshold, tp, fp, fn, precision, recall, cost] where a transaction is
    flagged if score >= threshold. Includes the 'flag nothing' end (threshold = +inf).
    """
    order = np.argsort(-score, kind="stable")
    s, yy = score[order], y[order]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    last = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]       # last index of each tie block
    thr = s[last]; tp, fp = tp[last], fp[last]
    P = yy.sum()
    thr = np.r_[np.inf, thr]; tp = np.r_[0, tp]; fp = np.r_[0, fp]
    fn = P - tp
    c = fn * cost.X + fp * cost.Y
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
    df = pd.DataFrame(dict(threshold=thr, tp=tp, fp=fp, fn=fn, precision=prec,
                           recall=tp / max(P, 1), flagged=tp + fp, cost=c))
    return df


def best_binary_threshold(y, score, cost: CostConfig):
    cc = cost_curve(y, score, cost)
    i = int(cc["cost"].values.argmin())
    return float(cc["threshold"].iloc[i]), cc


def band_costs(y, score, t_otp, t_block, cost: CostConfig) -> pd.DataFrame:
    """Per-band counts and rupee costs for the APPROVE / STEP-UP OTP / BLOCK rule."""
    band = np.where(score >= t_block, "BLOCK", np.where(score >= t_otp, "STEP-UP OTP", "APPROVE"))
    rows = []
    for b in ["APPROVE", "STEP-UP OTP", "BLOCK"]:
        m = band == b
        n, f = int(m.sum()), int(y[m].sum()); g = n - f
        if b == "APPROVE":
            c_f, c_g = f * cost.X, 0.0
        elif b == "STEP-UP OTP":
            c_f = f * (cost.Z + (1 - cost.otp_stop_rate) * cost.X); c_g = g * cost.Z
        else:
            c_f, c_g = 0.0, g * cost.Y
        rows.append(dict(band=b, transactions=n, pct_of_txns=100 * n / len(y), frauds=f,
                         pct_of_all_fraud=100 * f / max(y.sum(), 1), genuine=g,
                         fraud_rate_in_band_pct=100 * f / max(n, 1),
                         cost_from_fraud=c_f, cost_from_genuine=c_g, total_cost=c_f + c_g))
    out = pd.DataFrame(rows)
    return out


def optimise_bands(y, score, cost: CostConfig, n_grid: int = 300):
    """Search (t_otp <= t_block) minimising total 3-band cost. Vectorised via cumsums.

    Candidates = every distinct score in the top 5% (exact where all the action is for a rare-event
    problem) + a coarse grid below + "never". Identical rule to web/assets/model.js.
    """
    # Exact search: every distinct score in the top 5% is a candidate (the web app uses the same rule),
    # plus a coarse quantile grid below that, plus +inf ("never").
    top = np.sort(score)[::-1][: int(np.ceil(0.05 * len(score)))]
    q = np.unique(np.r_[top, np.quantile(score, 1 - np.logspace(np.log10(0.05), np.log10(0.5), max(n_grid // 10, 10))), np.inf])
    # counts of fraud / genuine with score >= t for each candidate t
    order = np.sort(score)
    ys = y[np.argsort(score, kind="stable")]
    cum_f = np.r_[0, np.cumsum(ys)]                        # frauds among the k lowest scores
    total_f = ys.sum(); total = len(ys)
    pos = np.searchsorted(order, q, side="left")           # #scores < t
    F_ge = total_f - cum_f[pos]; N_ge = total - pos; G_ge = N_ge - F_ge
    X, Y, Z, s = cost.X, cost.Y, cost.Z, cost.otp_stop_rate
    best = (np.inf, None, None)
    for i in range(len(q)):                                # t_otp = q[i]
        j = np.arange(i, len(q))                           # t_block = q[j] >= t_otp
        f_block, g_block = F_ge[j], G_ge[j]
        f_otp, g_otp = F_ge[i] - f_block, G_ge[i] - g_block
        f_app = total_f - F_ge[i]
        c = f_app * X + f_otp * (Z + (1 - s) * X) + g_otp * Z + g_block * Y
        k = int(np.argmin(c))
        if c[k] < best[0]:
            best = (float(c[k]), float(q[i]), float(q[j[k]]))
    return best   # (cost, t_otp, t_block)


def bootstrap_ci(y, score, fn, n_boot=1000, seed=SEED, stratified=True):
    """Percentile CI for any metric fn(y, score). Stratified resampling keeps fraud count stable."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        if stratified:
            ix = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        else:
            ix = rng.integers(0, len(y), len(y))
        vals.append(fn(y[ix], score[ix]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def recall_at_top(y, score, frac):
    k = max(1, int(np.ceil(frac * len(y))))
    top = np.argsort(-score, kind="stable")[:k]
    return y[top].sum() / max(y.sum(), 1), y[top].mean()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Model zoo
# ──────────────────────────────────────────────────────────────────────────────
def build_supervised_zoo(ds: Dataset, pos_weight: float, seed=SEED, fast=False):
    """Returns dict name -> dict(model, family, imbalance). Every resampler sits INSIDE an
    imblearn Pipeline, so SMOTE only ever sees the training fold."""
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    n_est = 150 if fast else 400
    smote = lambda: SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=seed)
    pre = lambda scale=True: make_preprocessor(ds, scale)
    xgb_kw = dict(n_estimators=n_est, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                  eval_metric="aucpr", tree_method="hist", random_state=seed, n_jobs=-1)
    lgb_kw = dict(n_estimators=n_est, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=seed, n_jobs=-1, verbose=-1)
    rf_kw = dict(n_estimators=200 if not fast else 100, min_samples_leaf=2, max_features="sqrt",
                 n_jobs=-1, random_state=seed)
    Z = {}
    Z["LogReg | none"] = dict(model=Pipeline([("pre", pre()), ("clf", LogisticRegression(max_iter=2000))]),
                              family="Logistic Regression", imbalance="none")
    Z["LogReg | class weights"] = dict(model=Pipeline([("pre", pre()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
                                       family="Logistic Regression", imbalance="class weights")
    Z["LogReg | SMOTE"] = dict(model=ImbPipeline([("pre", pre()), ("smote", smote()), ("clf", LogisticRegression(max_iter=2000))]),
                               family="Logistic Regression", imbalance="SMOTE (in-fold)")
    Z["RandomForest | none"] = dict(model=Pipeline([("pre", pre(False)), ("clf", RandomForestClassifier(**rf_kw))]),
                                    family="Random Forest", imbalance="none")
    Z["RandomForest | class weights"] = dict(model=Pipeline([("pre", pre(False)), ("clf", RandomForestClassifier(class_weight="balanced_subsample", **rf_kw))]),
                                             family="Random Forest", imbalance="class weights")
    Z["RandomForest | SMOTE"] = dict(model=ImbPipeline([("pre", pre(False)), ("smote", smote()), ("clf", RandomForestClassifier(**rf_kw))]),
                                     family="Random Forest", imbalance="SMOTE (in-fold)")
    Z["XGBoost | none"] = dict(model=Pipeline([("pre", pre(False)), ("clf", XGBClassifier(**xgb_kw))]),
                               family="XGBoost", imbalance="none")
    Z["XGBoost | class weights"] = dict(model=Pipeline([("pre", pre(False)), ("clf", XGBClassifier(scale_pos_weight=pos_weight, **xgb_kw))]),
                                        family="XGBoost", imbalance="class weights")
    Z["XGBoost | SMOTE"] = dict(model=ImbPipeline([("pre", pre(False)), ("smote", smote()), ("clf", XGBClassifier(**xgb_kw))]),
                                family="XGBoost", imbalance="SMOTE (in-fold)")
    Z["LightGBM | none"] = dict(model=Pipeline([("pre", pre(False)), ("clf", LGBMClassifier(**lgb_kw))]),
                                family="LightGBM", imbalance="none")
    Z["LightGBM | class weights"] = dict(model=Pipeline([("pre", pre(False)), ("clf", LGBMClassifier(scale_pos_weight=np.sqrt(pos_weight), **lgb_kw))]),
                                         family="LightGBM", imbalance="class weights (√ratio)")
    Z["LightGBM | SMOTE"] = dict(model=ImbPipeline([("pre", pre(False)), ("smote", smote()), ("clf", LGBMClassifier(**lgb_kw))]),
                                 family="LightGBM", imbalance="SMOTE (in-fold)")
    Z["CatBoost | class weights"] = dict(model=Pipeline([("pre", pre(False)), ("clf", CatBoostClassifier(
        iterations=n_est * 2, learning_rate=0.05, depth=6, auto_class_weights="SqrtBalanced",
        random_seed=seed, verbose=0, thread_count=-1))]), family="CatBoost", imbalance="class weights (√balanced)")
    Z["BalancedRF | undersample-bagging"] = dict(model=Pipeline([("pre", pre(False)), ("clf", BalancedRandomForestClassifier(
        n_estimators=300, sampling_strategy="all", replacement=True, bootstrap=False, n_jobs=-1, random_state=seed))]),
        family="Balanced RF", imbalance="per-tree undersampling")
    Z["EasyEnsemble | undersample-boosting"] = dict(model=Pipeline([("pre", pre(False)), ("clf", EasyEnsembleClassifier(
        n_estimators=10, n_jobs=-1, random_state=seed))]), family="EasyEnsemble", imbalance="per-member undersampling")
    return Z


def rule_baselines(ds: Dataset, tr_ix, seed=SEED):
    """Two transparent baselines a fraud-ops team could run without ML.

    1. Business rule: large amount (> train p95) at night (00–05h).
    2. Learned 2-split rule: depth-2 decision tree on the features (an IF/THEN a human can read).
    Returns dict name -> callable(df) -> score in [0,1], plus readable rule text.
    """
    from sklearn.tree import DecisionTreeClassifier, export_text
    df = ds.df
    amt_p95 = float(df["amount"].iloc[tr_ix].quantile(0.95))

    def business(d):
        return (((d["hour"] <= 5)) & (d["amount"] > amt_p95)).astype(float).values

    pre = make_preprocessor(ds, scale=False).fit(df.iloc[tr_ix])
    Xtr = pre.transform(df.iloc[tr_ix]); names = list(pre.get_feature_names_out())
    tree = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=seed).fit(Xtr, df[ds.target].values[tr_ix])
    tree_text = export_text(tree, feature_names=names, decimals=2)

    def learned(d):
        return tree.predict_proba(pre.transform(d))[:, 1]

    return {"Rule | night & amount>p95": (business, f"FLAG if hour ∈ [0,5] AND amount > {amt_p95:,.2f}"),
            "Rule | learned 2-split tree": (learned, tree_text)}


# ---- deep tabular models (PyTorch) ----------------------------------------------
def _torch():
    import torch, torch.nn as nn
    return torch, nn


class FocalLoss:
    """Focal loss (Lin et al., 2017): down-weights easy negatives so the gradient is driven by
    the rare, hard examples — a loss-level alternative to resampling."""
    def __init__(self, alpha=0.75, gamma=2.0):
        self.alpha, self.gamma = alpha, gamma

    def __call__(self, logits, y):
        torch, _ = _torch()
        p = torch.sigmoid(logits)
        ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        pt = p * y + (1 - p) * (1 - y)
        a = self.alpha * y + (1 - self.alpha) * (1 - y)
        return (a * (1 - pt) ** self.gamma * ce).mean()


def _make_mlp(d_in):
    torch, nn = _torch()
    return nn.Sequential(nn.Linear(d_in, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.2),
                         nn.Linear(128, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(0.2),
                         nn.Linear(64, 1))


def _make_ft_transformer(d_in, d_token=32, n_layers=2, n_heads=4):
    """Compact FT-Transformer (Gorishniy et al., NeurIPS 2021): every numeric feature becomes a
    token via its own linear embedding, a [CLS] token attends over all feature tokens."""
    torch, nn = _torch()

    class FTT(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(d_in, d_token) * 0.02)
            self.b = nn.Parameter(torch.zeros(d_in, d_token))
            self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
            layer = nn.TransformerEncoderLayer(d_token, n_heads, dim_feedforward=d_token * 2, dropout=0.1,
                                               batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, n_layers)
            self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, 1))

        def forward(self, x):
            tok = x.unsqueeze(-1) * self.w + self.b                   # (B, F, d)
            tok = torch.cat([self.cls.expand(len(x), -1, -1), tok], 1)
            return self.head(self.enc(tok)[:, 0])
    return FTT()


class TorchTabular:
    """sklearn-style wrapper: preprocessing → deep net trained with focal loss, early stopping on an
    inner time-ordered hold-out carved from the END of the training data (never the validation set)."""
    def __init__(self, ds: Dataset, arch="mlp", epochs=15, batch=1024, lr=1e-3, seed=SEED, patience=5):
        self.ds, self.arch, self.epochs, self.batch, self.lr, self.seed, self.patience = ds, arch, epochs, batch, lr, seed, patience

    def fit(self, X, y):
        torch, nn = _torch()
        from sklearn.metrics import average_precision_score
        torch.manual_seed(self.seed); torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.pre = make_preprocessor(self.ds, True).fit(X)
        Xa = self.pre.transform(X).astype(np.float32); y = np.asarray(y, np.float32)
        cut = int(0.85 * len(Xa))   # inner hold-out = last 15% of train (time-ordered)
        Xt, yt, Xv, yv = Xa[:cut], y[:cut], Xa[cut:], y[cut:]
        self.net = (_make_mlp(Xa.shape[1]) if self.arch == "mlp" else _make_ft_transformer(Xa.shape[1])).to(self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = FocalLoss()
        Xt_t, yt_t = torch.tensor(Xt), torch.tensor(yt)
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, bad = -1, None, 0
        self.history = []
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xt_t), generator=g)
            for k in range(0, len(perm), self.batch):
                b = perm[k:k + self.batch]
                if len(b) < 2: continue
                opt.zero_grad()
                loss = loss_fn(self.net(Xt_t[b].to(self.dev)).squeeze(1), yt_t[b].to(self.dev))
                loss.backward(); opt.step()
            ap = average_precision_score(yv, self._raw(Xv)) if yv.sum() > 0 else 0
            self.history.append(ap)
            if ap > best: best, bad, best_state = ap, 0, {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience: break
        if best_state: self.net.load_state_dict(best_state)
        return self

    def _raw(self, Xa):
        torch, _ = _torch()
        self.net.eval(); out = []
        with torch.no_grad():
            for k in range(0, len(Xa), 8192):
                out.append(torch.sigmoid(self.net(torch.tensor(Xa[k:k + 8192]).to(self.dev))).cpu().numpy().ravel())
        return np.concatenate(out)

    def predict_proba(self, X):
        p = self._raw(self.pre.transform(X).astype(np.float32))
        return np.c_[1 - p, p]


# ──────────────────────────────────────────────────────────────────────────────
# 6. Label-free anomaly detectors (fit on features ONLY — y never passed)
# ──────────────────────────────────────────────────────────────────────────────
class AutoencoderDetector:
    """Under-complete autoencoder; anomaly score = reconstruction MSE. Trained on all training rows
    with no labels (as in production, where new fraud is unlabelled)."""
    def __init__(self, ds, epochs=15, batch=1024, seed=SEED):
        self.ds, self.epochs, self.batch, self.seed = ds, epochs, batch, seed

    def fit(self, X):
        torch, nn = _torch()
        torch.manual_seed(self.seed)
        self.pre = make_preprocessor(self.ds, True).fit(X)
        Xa = torch.tensor(self.pre.transform(X).astype(np.float32)); d = Xa.shape[1]
        self.net = nn.Sequential(nn.Linear(d, 24), nn.SiLU(), nn.Linear(24, 8), nn.SiLU(),
                                 nn.Linear(8, 24), nn.SiLU(), nn.Linear(24, d))
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xa), generator=g)
            for k in range(0, len(perm), self.batch):
                b = Xa[perm[k:k + self.batch]]
                opt.zero_grad(); l = ((self.net(b) - b) ** 2).mean(); l.backward(); opt.step()
        self.ds = None          # drop the dataframe reference before pickling
        return self

    def score(self, X):
        torch, _ = _torch()
        Xa = torch.tensor(self.pre.transform(X).astype(np.float32))
        with torch.no_grad():
            return ((self.net(Xa) - Xa) ** 2).mean(1).numpy()


class DetectorWrap:
    """Preprocess → unsupervised estimator. Module-level so it pickles (cache + app bundle)."""
    def __init__(self, ds, name, est, kind, scale=True, max_rows=None, seed=SEED):
        self.ds, self.name, self.est, self.kind, self.scale, self.max_rows, self.seed = ds, name, est, kind, scale, max_rows, seed

    def fit(self, X):
        self.pre = make_preprocessor(self.ds, self.scale).fit(X)
        Xa = self.pre.transform(X)
        if self.max_rows and len(Xa) > self.max_rows:
            Xa = Xa[np.random.default_rng(self.seed).choice(len(Xa), self.max_rows, replace=False)]
        self.est.fit(Xa)
        self.ds = None          # drop the dataframe reference before pickling
        return self

    def score(self, X):
        Xa = self.pre.transform(X)
        if self.kind == "sk": return -self.est.score_samples(Xa)
        return self.est.decision_function(Xa)


def build_anomaly_detectors(ds, seed=SEED, lof_fit_rows=40000):
    """Each returns a fitted object exposing .score(X_df) -> higher = more anomalous."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from pyod.models.ecod import ECOD
    from pyod.models.copod import COPOD

    return {
        "Isolation Forest": DetectorWrap(ds, "Isolation Forest", IsolationForest(n_estimators=300, max_samples=512,
                                                                     contamination="auto", random_state=seed, n_jobs=-1), "sk", False),
        "ECOD": DetectorWrap(ds, "ECOD", ECOD(), "pyod"),
        "COPOD": DetectorWrap(ds, "COPOD", COPOD(), "pyod"),
        "LOF (novelty)": DetectorWrap(ds, "LOF (novelty)", LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1), "sk",
                              True, lof_fit_rows),
        "Autoencoder": AutoencoderDetector(ds),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. Plot style (Okabe–Ito colour-blind-safe palette)
# ──────────────────────────────────────────────────────────────────────────────
C = dict(fraud="#D55E00", genuine="#7F7F7F", model="#0072B2", accent="#E69F00", good="#009E73",
         purple="#CC79A7", sky="#56B4E9", ink="#222222")
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#7F7F7F", "#000000", "#F0E442"]


def set_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (9, 5), "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10.5,
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "axes.prop_cycle": __import__("cycler").cycler(color=PALETTE), "legend.frameon": False,
    })


def savefig(fig, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name), bbox_inches="tight")


def inr(x):
    x = float(x)
    if abs(x) >= 1e7: return f"₹{x/1e7:.2f} Cr"
    if abs(x) >= 1e5: return f"₹{x/1e5:.2f} L"
    return f"₹{x:,.0f}"
