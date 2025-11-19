# mlp_evaluation.py
import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, classification_report, confusion_matrix
)

# Deep Learning
import tensorflow as tf
from tensorflow import keras

# ------------------ CONFIG ------------------
DATA_PATH   = "Data_Processed/combined_daily.csv"
TARGET_COL  = "activity_level"
DATE_COL    = "date"
USER_COL    = "user"
FEATURES    = ["steps_total", "distance_total_km"]  # set LEAK_FREE=True to use single feature
LEAK_FREE   = False
TEST_SIZE   = 0.25
RANDOM_SEED = 42
EPOCHS      = 100
BATCH_SIZE  = 64
VAL_SPLIT   = 0.2

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ------------------ DATA ------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Ensure date is datetime
try:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
except Exception:
    pass

print("\nPreview of dataset:")
print(df.head())

# Feature selection
if LEAK_FREE:
    FEATURES = ["distance_total_km"]

X_full = df[FEATURES].copy()
y_full = df[TARGET_COL].copy()

# Label encoding
classes = sorted(y_full.unique())
cls_to_idx = {c:i for i,c in enumerate(classes)}
idx_to_cls = {i:c for c,i in cls_to_idx.items()}
y_full_i = y_full.map(cls_to_idx).values
n_classes = len(classes)

# ------------------ MODEL ------------------
def build_mlp(input_dim, n_classes):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dropout(0.30),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.20),
        keras.layers.Dense(n_classes, activation="softmax" if n_classes>2 else "sigmoid")
    ])
    loss = "sparse_categorical_crossentropy" if n_classes>2 else "binary_crossentropy"
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
    return model

def scale_fit_transform(Xtr, Xte):
    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), scaler.transform(Xte)

def plot_cm(cm, classes, title, out_path):
    fig = plt.figure(figsize=(5,4), dpi=140)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)  # default colormap, single-plot
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    # annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def evaluate_split(name, Xtr, Xte, ytr_i, yte_i, out_dir):
    # Scale for MLP
    Xtr_s, Xte_s = scale_fit_transform(Xtr, Xte)

    # Train
    model = build_mlp(Xtr_s.shape[1], n_classes)
    es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
    model.fit(Xtr_s, ytr_i, validation_split=VAL_SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=0)

    # Predict
    probs = model.predict(Xte_s, verbose=0)
    if n_classes > 2:
        yhat_i = probs.argmax(axis=1)
    else:
        yhat_i = (probs.ravel() >= 0.5).astype(int)

    # Metrics
    acc  = accuracy_score(yte_i, yhat_i)
    balc = balanced_accuracy_score(yte_i, yhat_i)
    f1w  = f1_score(yte_i, yhat_i, average="weighted", zero_division=0)
    f1m  = f1_score(yte_i, yhat_i, average="macro", zero_division=0)

    # Confusion matrix (use original labels for readability)
    y_true = pd.Series(yte_i).map(idx_to_cls)
    y_pred = pd.Series(yhat_i).map(idx_to_cls)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Save CM image
    cm_path = os.path.join(out_dir, f"{name}_confusion_matrix.png")
    plot_cm(cm, classes, f"{name} — Confusion Matrix", cm_path)

    # Text report
    report = classification_report(y_true, y_pred, digits=3)
    with open(os.path.join(out_dir, f"{name}_report.txt"), "w") as f:
        f.write(report)

    return {
        "split": name,
        "accuracy": float(acc),
        "balanced_accuracy": float(balc),
        "f1_weighted": float(f1w),
        "f1_macro": float(f1m),
        "cm_path": cm_path
    }

# ------------------ SPLITS ------------------
os.makedirs("outputs_mlp", exist_ok=True)
results = []

# 1) Random (stratified) split
X = X_full.copy().values
y = y_full_i.copy()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
print("\nTraining on Random (Stratified) split...")
results.append(evaluate_split("random_split", Xtr, Xte, ytr, yte, "outputs_mlp"))

# 2) Group-aware split (no user overlap)
print("\nTraining on Group (User) split...")
groups = df[USER_COL]
X = X_full.copy().values
y = y_full_i.copy()
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
tr_idx, te_idx = next(gss.split(X, y, groups=groups))
Xtr_g, Xte_g = X[tr_idx], X[te_idx]
ytr_g, yte_g = y[tr_idx], y[te_idx]
results.append(evaluate_split("group_split", Xtr_g, Xte_g, ytr_g, yte_g, "outputs_mlp"))

# 3) Time-aware split (train early dates, test later dates)
print("\nTraining on Time split...")
df_sorted = df.sort_values(DATE_COL)
cut = int(len(df_sorted) * (1.0 - TEST_SIZE))
train_df, test_df = df_sorted.iloc[:cut], df_sorted.iloc[cut:]
Xtr_t = train_df[FEATURES].values
Xte_t = test_df[FEATURES].values
ytr_t = train_df[TARGET_COL].map(cls_to_idx).values
yte_t = test_df[TARGET_COL].map(cls_to_idx).values
results.append(evaluate_split("time_split", Xtr_t, Xte_t, ytr_t, yte_t, "outputs_mlp"))

# ------------------ SAVE SUMMARY ------------------
pd.DataFrame(results).to_csv("outputs_mlp/mlp_results_summary.csv", index=False)
with open("outputs_mlp/mlp_results_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved: outputs_mlp/mlp_results_summary.csv and confusion matrix images.")
print("Done.")
