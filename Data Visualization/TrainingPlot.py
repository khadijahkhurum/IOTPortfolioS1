# ============================================================
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# -------------------- Config --------------------
DATA_PATH = "Data_Processed/combined_daily.csv"
PLOTS_DIR = "Plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

LEAK_FREE = False  # True -> use only distance_total_km (since label uses steps)

# -------------------- Load ----------------------
df = pd.read_csv(DATA_PATH)

# Choose features dynamically based on availability
cand = [c for c in ["steps_total", "distance_total_km", "avg_hr", "sleep_hours"] if c in df.columns]
if not cand:
    raise RuntimeError("No usable feature columns found.")
if LEAK_FREE:
    cand = [c for c in cand if c != "steps_total"]
    if not cand:
        cand = ["distance_total_km"]  # fallback

X = df[cand]
y = df["activity_level"]

print("Using features:", cand)

# -------------------- CV + pipelines -------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe_lr  = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
pipe_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))

# -------------------- Learning curves ------------
def plot_learning_curve(estimator, title, filename):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=None, random_state=42
    )
    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_std    = val_scores.std(axis=1)

    plt.figure(figsize=(7,5))
    plt.plot(train_sizes, train_mean, marker="o", label="Training score")
    plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.2)
    plt.plot(train_sizes, val_mean, marker="s", label="Validation score")
    plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.2)
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = os.path.join(PLOTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"✅ Saved {out}")

plot_learning_curve(pipe_lr,  "Learning Curve — Logistic Regression", "learning_curve_logreg.png")
plot_learning_curve(pipe_knn, "Learning Curve — KNN (k=5)",         "learning_curve_knn.png")

# -------------------- Validation curve (KNN: k sweep) ----------
k_values = list(range(1, 21))
val_scores = []
for k in k_values:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    score = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()
    val_scores.append(score)

best_k = k_values[int(np.argmax(val_scores))]
best_score = max(val_scores)

plt.figure(figsize=(7,5))
plt.plot(k_values, val_scores, marker="o")
plt.title("Validation Curve — KNN (varying k)")
plt.xlabel("k (n_neighbors)")
plt.ylabel("CV Accuracy")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
for x, s in zip(k_values, val_scores):
    if x == best_k:
        plt.scatter([x], [s], s=80)
        break
out = os.path.join(PLOTS_DIR, "validation_curve_knn.png")
plt.tight_layout()
plt.savefig(out, dpi=200)
plt.close()
print(f"✅ Saved {out}")
print(f"Best k = {best_k}  |  CV Accuracy = {best_score:.3f}")
