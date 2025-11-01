# ============================================================
# Models: Linear Regression, KNN Regressor
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------- Load data --------------------
df = pd.read_csv("Data_Processed/combined_daily.csv")

# Features (predictors) and numeric target
X = df[["distance_total_km", "avg_hr", "sleep_hours"]].copy()
y = df["steps_total"].astype(float)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Scale features (helps KNN; fine for Linear Regression too)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# -------------------- Model 1: Linear Regression --------------------
lin = LinearRegression()
lin.fit(X_train_s, y_train)
pred_lin = lin.predict(X_test_s)

mae_lin = mean_absolute_error(y_test, pred_lin)
mse_lin = mean_squared_error(y_test, pred_lin)
rmse_lin = np.sqrt(mse_lin)
r2_lin = r2_score(y_test, pred_lin)

# -------------------- Model 2: KNN Regressor ------------------------
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_s, y_train)
pred_knn = knn.predict(X_test_s)

mae_knn = mean_absolute_error(y_test, pred_knn)
mse_knn = mean_squared_error(y_test, pred_knn)
rmse_knn = np.sqrt(mse_knn)
r2_knn = r2_score(y_test, pred_knn)

# -------------------- Print metrics nicely --------------------------
def fmt(mae, mse, rmse, r2):
    return (f"MAE={mae:.1f}  |  MSE={mse:.1f}  |  RMSE={rmse:.1f}  |  R²={r2:.3f}")

print("\n=== Regression on steps_total ===")
print("Features: distance_total_km, avg_hr, sleep_hours")
print("\nLinear Regression :", fmt(mae_lin, mse_lin, rmse_lin, r2_lin))
print("KNN Regressor     :", fmt(mae_knn, mse_knn, rmse_knn, r2_knn))

# -------------------- Plots -----------------------
# 1) Predicted vs Actual (parity plot) for both models
plt.figure(figsize=(6,5))
plt.scatter(y_test, pred_lin, alpha=0.5, label="Linear Reg")
plt.scatter(y_test, pred_knn, alpha=0.5, label="KNN Regr")
max_val = max(y_test.max(), pred_lin.max(), pred_knn.max())
plt.plot([0, max_val], [0, max_val], linestyle="--")  # y=x reference
plt.xlabel("Actual steps_total")
plt.ylabel("Predicted steps_total")
plt.title("Predicted vs Actual (Parity Plot)")
plt.legend()
plt.tight_layout()
plt.show()

# 2) Residuals plot (prediction error) for both models
res_lin = y_test - pred_lin
res_knn = y_test - pred_knn

plt.figure(figsize=(6,5))
plt.scatter(pred_lin, res_lin, alpha=0.5, label="Linear Reg")
plt.scatter(pred_knn, res_knn, alpha=0.5, label="KNN Regr")
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted steps_total")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals Plot")
plt.legend()
plt.tight_layout()
plt.show()
