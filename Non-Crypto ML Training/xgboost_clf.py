# xgb_evaluation.py
import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler  # not used for trees, but kept if you extend
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, log_loss,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

from xgboost import XGBClassifier

# ------------------ CONFIG ------------------
DATA_PATH   = "Data_Processed/combined_daily.csv"
TARGET_COL  = "activity_level"
DATE_COL    = "date"
USER_COL    = "user"
FEATURES    = ["steps_total", "distance_total_km"]
LEAK_FREE   = False
TEST_SIZE   = 0.25
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ------------------ UTILS ------------------
def plot_cm(cm, classes, title, out_path):
    fig = plt.figure(figsize=(5,4), dpi=140)
    ax = fig.add_subplot(111)
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def multiclass_brier(y_true_idx, proba, n_classes):
    # One-hot encode y_true and compute mean squared error vs proba
    Y = np.zeros((len(y_true_idx), n_classes))
    Y[np.arange(len(y_true_idx)), y_true_idx] = 1.0
    return float(np.mean(np.sum((proba - Y) ** 2, axis=1)))

def evaluate_split(name, Xtr, Xte, ytr_idx, yte_idx, classes, out_dir):
    # Train XGB
    xgb = XGBClassifier(
        objective="multi:softprob" if len(classes) > 2 else "binary:logistic",
        num_class=len(classes) if len(classes) > 2 else None,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss" if len(classes) > 2 else "logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_SEED
    )
    xgb.fit(Xtr, ytr_idx)

    proba = xgb.predict_proba(Xte)
    yhat_idx = np.argmax(proba, axis=1) if len(classes) > 2 else (proba.ravel() >= 0.5).astype(int)

    # Basic metrics
    acc  = accuracy_score(yte_idx, yhat_idx)
    balc = balanced_accuracy_score(yte_idx, yhat_idx)
    f1w  = f1_score(yte_idx, yhat_idx, average="weighted", zero_division=0)
    f1m  = f1_score(yte_idx, yhat_idx, average="macro", zero_division=0)
    mcc  = matthews_corrcoef(yte_idx, yhat_idx)
    kap  = cohen_kappa_score(yte_idx, yhat_idx)

    # Prob-based metrics
    ll    = log_loss(yte_idx, proba, labels=list(range(len(classes))))
    brier = multiclass_brier(yte_idx, proba, len(classes))

    # --- Feature importance (gain) with readable names ---
    imp_dict = xgb.get_booster().get_score(importance_type="gain")  # keys: f0, f1, ...
    pretty_imp = {}
    for k, v in imp_dict.items():
        if k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            try:
                name = FEATURES[idx]   # map f0 -> your column name
            except Exception:
                name = k
            pretty_imp[name] = v
        else:
            pretty_imp[k] = v
    pd.Series(pretty_imp).sort_values(ascending=False).to_csv(
        os.path.join(out_dir, f"{name}_feature_importance.csv")
    )

    # --- Reliability (multiclass) curve: confidence vs observed accuracy ---
    max_conf = proba.max(axis=1)
    correct  = (yhat_idx == yte_idx).astype(int)
    bins = np.quantile(max_conf, np.linspace(0, 1, 11))
    bins = np.unique(bins)  # ensure unique edges
    bin_ids = np.digitize(max_conf, bins[1:-1], right=True)

    bin_x, bin_y = [], []
    for b in range(bin_ids.min(), bin_ids.max() + 1):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        bin_x.append(float(max_conf[mask].mean()))
        bin_y.append(float(correct[mask].mean()))

    fig = plt.figure(figsize=(5,4), dpi=140)
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])              # perfect calibration
    ax.plot(bin_x, bin_y, marker="o")    # empirical reliability
    ax.set_title(f"{name} — Reliability")
    ax.set_xlabel("Predicted confidence (max class prob)")
    ax.set_ylabel("Observed accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_reliability.png"))
    plt.close(fig)

    # Macro ROC-AUC (OvR)
    try:
        yte_bin = label_binarize(yte_idx, classes=list(range(len(classes))))
        auc_macro = roc_auc_score(yte_bin, proba, average="macro", multi_class="ovr")
    except Exception:
        auc_macro = None

    # Confusion matrix & report using original labels
    idx_to_cls = {i: c for i, c in enumerate(classes)}
    y_true_lbl = pd.Series(yte_idx).map(idx_to_cls)
    y_pred_lbl = pd.Series(yhat_idx).map(idx_to_cls)
    cm = confusion_matrix(y_true_lbl, y_pred_lbl, labels=classes)

    cm_path = os.path.join(out_dir, f"{name}_confusion_matrix.png")
    plot_cm(cm, classes, f"{name} — Confusion Matrix", cm_path)
    with open(os.path.join(out_dir, f"{name}_report.txt"), "w") as f:
        f.write(classification_report(y_true_lbl, y_pred_lbl, labels=classes, digits=3))

    return {
        "split": name,
        "accuracy": float(acc),
        "balanced_accuracy": float(balc),
        "f1_weighted": float(f1w),
        "f1_macro": float(f1m),
        "mcc": float(mcc),
        "cohen_kappa": float(kap),
        "log_loss": float(ll),
        "brier_score": float(brier),
        "roc_auc_macro_ovr": None if auc_macro is None else float(auc_macro),
        "cm_path": cm_path
    }

# ------------------ LOAD DATA ------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
try:
    df["date"] = pd.to_datetime(df["date"])
except Exception:
    pass

print("\nPreview of dataset:")
print(df.head())

# Features
if LEAK_FREE:
    FEATURES = ["distance_total_km"]
X_full = df[FEATURES].values
y_lbl  = df[TARGET_COL].values

# Encode labels
classes = sorted(pd.Series(y_lbl).unique().tolist())
cls_to_idx = {c:i for i,c in enumerate(classes)}
y_idx = np.array([cls_to_idx[c] for c in y_lbl], dtype=int)

# ------------------ RUN SPLITS ------------------
os.makedirs("outputs_xgb", exist_ok=True)
results = []

# 1) Random split
Xtr, Xte, ytr, yte = train_test_split(X_full, y_idx, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_idx)
print("\nTraining on Random (Stratified) split...")
results.append(evaluate_split("random_split", Xtr, Xte, ytr, yte, classes, "outputs_xgb"))

# 2) Group split (no user overlap)
print("\nTraining on Group (User) split...")
groups = df[USER_COL].values
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
tr_idx, te_idx = next(gss.split(X_full, y_idx, groups=groups))
results.append(evaluate_split("group_split", X_full[tr_idx], X_full[te_idx], y_idx[tr_idx], y_idx[te_idx], classes, "outputs_xgb"))

# 3) Time split (earlier -> later)
print("\nTraining on Time split...")
dfs = df.sort_values(DATE_COL)
cut = int(len(dfs) * (1.0 - TEST_SIZE))
tr_df, te_df = dfs.iloc[:cut], dfs.iloc[cut:]
Xtr_t = tr_df[FEATURES].values
Xte_t = te_df[FEATURES].values
ytr_t = np.array([cls_to_idx[c] for c in tr_df[TARGET_COL].values], dtype=int)
yte_t = np.array([cls_to_idx[c] for c in te_df[TARGET_COL].values], dtype=int)
results.append(evaluate_split("time_split", Xtr_t, Xte_t, ytr_t, yte_t, classes, "outputs_xgb"))

# ------------------ SAVE ------------------
pd.DataFrame(results).to_csv("outputs_xgb/xgb_results_summary.csv", index=False)
with open("outputs_xgb/xgb_results_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved: outputs_xgb/xgb_results_summary.csv and confusion matrix images.")
print("Done.")
