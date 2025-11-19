# svm_rbf_calibrated_evaluation.py
import os, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, log_loss,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# ------------------ CONFIG ------------------
DATA_PATH   = "Data_Processed/combined_daily.csv"
TARGET_COL  = "activity_level"
DATE_COL    = "date"
USER_COL    = "user"
FEATURES    = ["steps_total", "distance_total_km"]  # set to ["distance_total_km"] for leak-free
TEST_SIZE   = 0.25
CALIB_SIZE  = 0.20    # taken from the training fold for calibration
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

def reliability_curve(proba, y_true_idx, out_path, title):
    # Multiclass reliability: bin by max predicted prob vs. observed accuracy
    max_conf = proba.max(axis=1)
    correct  = (np.argmax(proba, axis=1) == y_true_idx).astype(int)
    bins = np.unique(np.quantile(max_conf, np.linspace(0,1,11)))
    bin_ids = np.digitize(max_conf, bins[1:-1], right=True)
    xs, ys = [], []
    for b in range(bin_ids.min(), bin_ids.max()+1):
        m = bin_ids==b
        if m.sum():
            xs.append(float(max_conf[m].mean()))
            ys.append(float(correct[m].mean()))
    fig = plt.figure(figsize=(5,4), dpi=140)
    ax = fig.add_subplot(111)
    ax.plot([0,1],[0,1])
    ax.plot(xs, ys, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Predicted confidence (max class prob)")
    ax.set_ylabel("Observed accuracy")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)

def multiclass_brier(y_idx, proba, n_classes):
    Y = np.zeros((len(y_idx), n_classes))
    Y[np.arange(len(y_idx)), y_idx] = 1.0
    return float(np.mean(np.sum((proba - Y)**2, axis=1)))

def evaluate_once(tag, Xtr, Xte, ytr, yte, classes, out_dir):
    """Train uncalibrated SVM, then Platt (sigmoid) & Isotonic calibration on a calibration slice.
       Returns metric dicts for [uncalibrated, platt, isotonic]."""
    os.makedirs(out_dir, exist_ok=True)
    # Split train into model-train + calibration
    Xtr_m, Xcal, ytr_m, ycal = train_test_split(
        Xtr, ytr, test_size=CALIB_SIZE, random_state=RANDOM_SEED, stratify=ytr
    )
    # Scale on model-train only, apply to cal & test
    scaler = StandardScaler().fit(Xtr_m)
    Xtr_m_s = scaler.transform(Xtr_m)
    Xcal_s  = scaler.transform(Xcal)
    Xte_s   = scaler.transform(Xte)

    # Base SVM (probability=False for faster fit; calibrated models provide probs)
    base = SVC(kernel="rbf", C=3.0, gamma="scale", probability=False,
               class_weight="balanced", random_state=RANDOM_SEED)
    base.fit(Xtr_m_s, ytr_m)

    rows = []

    def measure(proba, yhat_idx, suffix):
        acc  = accuracy_score(yte, yhat_idx)
        balc = balanced_accuracy_score(yte, yhat_idx)
        f1w  = f1_score(yte, yhat_idx, average="weighted", zero_division=0)
        f1m  = f1_score(yte, yhat_idx, average="macro", zero_division=0)
        mcc  = matthews_corrcoef(yte, yhat_idx)
        kap  = cohen_kappa_score(yte, yhat_idx)
        ll   = log_loss(yte, proba, labels=list(range(len(classes))))
        brier= multiclass_brier(yte, proba, len(classes))
        try:
            y_bin = label_binarize(yte, classes=list(range(len(classes))))
            aucm  = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
        except Exception:
            aucm  = None

        # CM + report with original labels
        idx_to_cls = {i:c for i,c in enumerate(classes)}
        y_true_lbl = pd.Series(yte).map(idx_to_cls)
        y_pred_lbl = pd.Series(yhat_idx).map(idx_to_cls)
        cm = confusion_matrix(y_true_lbl, y_pred_lbl, labels=classes)
        cm_path = os.path.join(out_dir, f"{tag}_{suffix}_confusion_matrix.png")
        plot_cm(cm, classes, f"{tag} — {suffix} — Confusion Matrix", cm_path)
        with open(os.path.join(out_dir, f"{tag}_{suffix}_report.txt"), "w") as f:
            f.write(classification_report(y_true_lbl, y_pred_lbl, labels=classes, digits=3))

        # Reliability curve
        reliability_curve(proba, yte, os.path.join(out_dir, f"{tag}_{suffix}_reliability.png"),
                          f"{tag} — {suffix} — Reliability")

        return dict(
            split=tag, model=f"SVM-RBF-{suffix}", accuracy=float(acc), balanced_accuracy=float(balc),
            f1_weighted=float(f1w), f1_macro=float(f1m), mcc=float(mcc), cohen_kappa=float(kap),
            log_loss=float(ll), brier_score=float(brier),
            roc_auc_macro_ovr=None if aucm is None else float(aucm),
            cm_path=cm_path
        )

    # (A) Uncalibrated: approximate probs via softmax(decision_function) for comparison
    dec = base.decision_function(Xte_s)  # shape (n, K)
    if dec.ndim == 1:
        dec = np.vstack([-dec, dec]).T
    e = np.exp(dec - dec.max(axis=1, keepdims=True))
    proba_unc = e / e.sum(axis=1, keepdims=True)
    yhat_unc  = np.argmax(proba_unc, axis=1)
    rows.append(measure(proba_unc, yhat_unc, "uncalibrated"))

    # (B) Platt (sigmoid) calibration
    platt = CalibratedClassifierCV(estimator=base, method="sigmoid", cv="prefit")
    platt.fit(Xcal_s, ycal)
    proba_sig = platt.predict_proba(Xte_s)
    yhat_sig  = np.argmax(proba_sig, axis=1)
    rows.append(measure(proba_sig, yhat_sig, "platt"))

    # (C) Isotonic calibration
    iso = CalibratedClassifierCV(estimator=base, method="isotonic", cv="prefit")
    iso.fit(Xcal_s, ycal)
    proba_iso = iso.predict_proba(Xte_s)
    yhat_iso  = np.argmax(proba_iso, axis=1)
    rows.append(measure(proba_iso, yhat_iso, "isotonic"))

    return rows

# ------------------ LOAD ------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
try:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
except Exception:
    pass

print("\nPreview of dataset:")
print(df.head())

X_df = df[FEATURES].copy()
y_lbl = df[TARGET_COL].copy()
classes = sorted(y_lbl.unique())
cls_to_idx = {c:i for i,c in enumerate(classes)}
y_idx = y_lbl.map(cls_to_idx).values
X = X_df.values

# ------------------ RUN ------------------
os.makedirs("outputs_svm_cal", exist_ok=True)
all_rows = []

# Random split
Xtr, Xte, ytr, yte = train_test_split(X, y_idx, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_idx)
print("\nTraining on Random (Stratified) split...")
all_rows += evaluate_once("random_split", Xtr, Xte, ytr, yte, classes, "outputs_svm_cal")

# Group split
print("\nTraining on Group (User) split...")
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
tr_idx, te_idx = next(gss.split(X, y_idx, groups=df[USER_COL].values))
all_rows += evaluate_once("group_split", X[tr_idx], X[te_idx], y_idx[tr_idx], y_idx[te_idx], classes, "outputs_svm_cal")

# Time split
print("\nTraining on Time split...")
dfs = df.sort_values(DATE_COL)
cut = int(len(dfs) * (1.0 - TEST_SIZE))
tr_df, te_df = dfs.iloc[:cut], dfs.iloc[cut:]
all_rows += evaluate_once(
    "time_split",
    tr_df[FEATURES].values, te_df[FEATURES].values,
    tr_df[TARGET_COL].map(cls_to_idx).values, te_df[TARGET_COL].map(cls_to_idx).values,
    classes, "outputs_svm_cal"
)

# ------------------ SAVE ------------------
summary = pd.DataFrame(all_rows)
summary.to_csv("outputs_svm_cal/svm_calibrated_results_summary.csv", index=False)
with open("outputs_svm_cal/svm_calibrated_results_summary.json", "w") as f:
    json.dump(all_rows, f, indent=2)

print("\nSaved: outputs_svm_cal/svm_calibrated_results_summary.csv and all figures.")
print("Done.")
