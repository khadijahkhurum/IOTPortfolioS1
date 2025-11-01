# ============================================================
# Graphs & Heatmaps for Apple Health ML models
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv("Data_Processed/combined_daily.csv")
print("Dataset loaded:", df.shape)
print(df.head())

# ------------------------------------------------------------
# 2. CORRELATION HEATMAP
# ------------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(df[["steps_total", "distance_total_km", "avg_hr", "sleep_hours"]].corr(),
            annot=True, cmap="Blues", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 3. TRAIN MODELS AGAIN (to visualize confusion matrices)
# ------------------------------------------------------------
X = df[["steps_total", "distance_total_km", "avg_hr", "sleep_hours"]]
y = df["activity_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# ------------------------------------------------------------
# 4. CONFUSION MATRIX HEATMAPS
# ------------------------------------------------------------
labels = sorted(y.unique())

cm_lr = confusion_matrix(y_test, y_pred_lr, labels=labels)
cm_knn = confusion_matrix(y_test, y_pred_knn, labels=labels)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.heatmap(cm_lr, annot=True, fmt='d', cmap="YlGnBu",
            xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title("Logistic Regression Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

sns.heatmap(cm_knn, annot=True, fmt='d', cmap="YlGnBu",
            xticklabels=labels, yticklabels=labels, ax=axes[1])
axes[1].set_title("KNN Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. BAR CHART OF MODEL ACCURACIES
# ------------------------------------------------------------
from sklearn.metrics import accuracy_score

acc_lr = accuracy_score(y_test, y_pred_lr)
acc_knn = accuracy_score(y_test, y_pred_knn)

acc_df = pd.DataFrame({
    "Model": ["Logistic Regression", "KNN"],
    "Accuracy": [acc_lr, acc_knn]
})

plt.figure(figsize=(5, 4))
sns.barplot(data=acc_df, x="Model", y="Accuracy", palette="coolwarm")
plt.ylim(0, 1.1)
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.show()

print("\nVisualization complete ✅")
