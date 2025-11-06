
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Data_Processed/combined_daily.csv")

X = df[["distance_total_km"]]     # only feature left
y = df["steps_total"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lin  = LinearRegression().fit(X_train_s, y_train)
knnr = KNeighborsRegressor(n_neighbors=5).fit(X_train_s, y_train)

pred_lin  = lin.predict(X_test_s)
pred_knnr = knnr.predict(X_test_s)

def metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

for name, pred in [("Linear", pred_lin), ("KNN Regr", pred_knnr)]:
    mae, mse, rmse, r2 = metrics(y_test, pred)
    print(f"{name:>10}: MAE={mae:.1f}  MSE={mse:.1f}  RMSE={rmse:.1f}  R²={r2:.3f}")

# Parity plot
plt.figure(figsize=(6,5))
plt.scatter(y_test, pred_lin, alpha=0.5, label="Linear")
plt.scatter(y_test, pred_knnr, alpha=0.5, label="KNN Regr")
mx = max(y_test.max(), pred_lin.max(), pred_knnr.max())
plt.plot([0, mx], [0, mx], linestyle="--")
plt.xlabel("Actual steps_total"); plt.ylabel("Predicted steps_total")
plt.title("Predicted vs Actual (Parity)")
plt.legend(); plt.tight_layout(); plt.show()

# Residuals
res_lin  = y_test - pred_lin
res_knnr = y_test - pred_knnr
plt.figure(figsize=(6,5))
plt.scatter(pred_lin,  res_lin,  alpha=0.5, label="Linear")
plt.scatter(pred_knnr, res_knnr, alpha=0.5, label="KNN Regr")
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted steps_total"); plt.ylabel("Residual (Actual - Pred)")
plt.title("Residuals"); plt.legend(); plt.tight_layout(); plt.show()
