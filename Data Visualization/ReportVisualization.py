# ============================================================
# ReportVisualization.py
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

# ------------------- Paths -------------------
DATA_PATH = "Data_Processed/combined_daily.csv"
PLOTS_DIR = "Plots"
OUT_METRICS = "Data_Processed/metrics_summary.csv"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ------------------- Load & prep data -------------------
df = pd.read_csv(DATA_PATH)
# Basic safety fills
for col in ["steps_total", "distance_total_km", "avg_hr", "sleep_hours"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

# ------------------- CLASSIFICATION -------------------
# Predict activity_level
X_cls = df[["steps_total", "distance_total_km", "avg_hr", "sleep_hours"]]
y_cls = df["activity_level"]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_cls, y_cls, test_size=0.25, random_state=42, stratify=y_cls
)
scaler_c = StandardScaler()
Xc_train_s = scaler_c.fit_transform(Xc_train)
Xc_test_s  = scaler_c.transform(Xc_test)

# Models
logreg = LogisticRegression(max_iter=1000)
knn_c  = KNeighborsClassifier(n_neighbors=5)

logreg.fit(Xc_train_s, yc_train)
knn_c.fit(Xc_train_s, yc_train)

pred_lr  = logreg.predict(Xc_test_s)
pred_knn = knn_c.predict(Xc_test_s)

acc_lr  = accuracy_score(yc_test, pred_lr)
acc_knn = accuracy_score(yc_test, pred_knn)

# Classification reports (as dicts for table)
rep_lr  = classification_report(yc_test, pred_lr, output_dict=True, zero_division=0)
rep_knn = classification_report(yc_test, pred_knn, output_dict=True, zero_division=0)

# ------------------- REGRESSION -------------------
# Predict numeric steps_total from other features
X_reg = df[["distance_total_km", "avg_hr", "sleep_hours"]]
y_reg = df["steps_total"].astype(float)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.25, random_state=42
)
scaler_r = StandardScaler()
Xr_train_s = scaler_r.fit_transform(Xr_train)
Xr_test_s  = scaler_r.transform(Xr_test)

lin   = LinearRegression()
knn_r = KNeighborsRegressor(n_neighbors=5)

lin.fit(Xr_train_s, yr_train)
knn_r.fit(Xr_train_s, yr_train)

pred_lin = lin.predict(Xr_test_s)
pred_knr = knn_r.predict(Xr_test_s)

def reg_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

mae_lin, mse_lin, rmse_lin, r2_lin = reg_metrics(yr_test, pred_lin)
mae_knr, mse_knr, rmse_knr, r2_knr = reg_metrics(yr_test, pred_knr)

# ------------------- Save metrics table -------------------
rows = [
    # Classification (include macro avg from classification_report)
    {"task":"classification","model":"Logistic Regression",
     "accuracy":acc_lr,
     "precision_macro":rep_lr["macro avg"]["precision"],
     "recall_macro":rep_lr["macro avg"]["recall"],
     "f1_macro":rep_lr["macro avg"]["f1-score"]},
    {"task":"classification","model":"KNN (Classifier)",
     "accuracy":acc_knn,
     "precision_macro":rep_knn["macro avg"]["precision"],
     "recall_macro":rep_knn["macro avg"]["recall"],
     "f1_macro":rep_knn["macro avg"]["f1-score"]},

    # Regression
    {"task":"regression","model":"Linear Regression",
     "MAE":mae_lin,"MSE":mse_lin,"RMSE":rmse_lin,"R2":r2_lin},
    {"task":"regression","model":"KNN (Regressor)",
     "MAE":mae_knr,"MSE":mse_knr,"RMSE":rmse_knr,"R2":r2_knr},
]
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(OUT_METRICS, index=False)
print(f"✅ Metrics saved → {OUT_METRICS}")

# ------------------- PLOTS (saved as PNGs) -------------------
# 1) Correlation heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df[["steps_total","distance_total_km","avg_hr","sleep_hours"]].corr(),
            annot=True, cmap="Blues", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=200)
plt.close()

# 2) Confusion matrices (LR & KNN)
labels = sorted(y_cls.unique())
cm_lr  = confusion_matrix(yc_test, pred_lr,  labels=labels)
cm_knn = confusion_matrix(yc_test, pred_knn, labels=labels)

fig, axes = plt.subplots(1,2, figsize=(10,4))
sns.heatmap(cm_lr,  annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title("LogReg Confusion Matrix"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

sns.heatmap(cm_knn, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=labels, yticklabels=labels, ax=axes[1])
axes[1].set_title("KNN Confusion Matrix"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"), dpi=200)
plt.close()

# 3) Accuracy bar chart
acc_df = pd.DataFrame({"Model":["Logistic Regression","KNN"], "Accuracy":[acc_lr, acc_knn]})
plt.figure(figsize=(6,4))
sns.barplot(data=acc_df, x="Model", y="Accuracy")
plt.ylim(0, 1.05)
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "accuracy_bar.png"), dpi=200)
plt.close()

# 4) Trend line (Steps & Avg HR over time)
if "date" in df.columns:
    ts = df.copy()
    ts["date"] = pd.to_datetime(ts["date"])
    ts = ts.sort_values("date")
    plt.figure(figsize=(10,5))
    plt.plot(ts["date"], ts["steps_total"], label="Steps")
    plt.plot(ts["date"], ts["avg_hr"], label="Avg Heart Rate")
    plt.title("Daily Steps and Average Heart Rate Over Time")
    plt.xlabel("Date"); plt.ylabel("Value"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "trend_steps_hr.png"), dpi=200)
    plt.close()

# 5) Pairplot by activity level
g = sns.pairplot(
    df, vars=["steps_total","distance_total_km","avg_hr","sleep_hours"],
    hue="activity_level", palette="coolwarm", diag_kind="kde", height=2.2
)
g.fig.suptitle("IoT Health Features by Activity Level", y=1.02)
g.savefig(os.path.join(PLOTS_DIR, "pairplot_activity.png"), dpi=200)
plt.close()

# 6) Regression: Predicted vs Actual (Parity) + Residuals
# Parity
plt.figure(figsize=(6,5))
plt.scatter(yr_test, pred_lin, alpha=0.5, label="Linear Reg")
plt.scatter(yr_test, pred_knr, alpha=0.5, label="KNN Regr")
mx = max(yr_test.max(), pred_lin.max(), pred_knr.max())
plt.plot([0, mx], [0, mx], linestyle="--")
plt.xlabel("Actual steps_total"); plt.ylabel("Predicted steps_total")
plt.title("Predicted vs Actual (Parity Plot)")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "reg_parity.png"), dpi=200)
plt.close()

# Residuals
res_lin = yr_test - pred_lin
res_knr = yr_test - pred_knr
plt.figure(figsize=(6,5))
plt.scatter(pred_lin, res_lin, alpha=0.5, label="Linear Reg")
plt.scatter(pred_knr, res_knr, alpha=0.5, label="KNN Regr")
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted steps_total"); plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals Plot"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "reg_residuals.png"), dpi=200)
plt.close()

print(f"✅ Plots saved → {PLOTS_DIR}")
print("Done.")
