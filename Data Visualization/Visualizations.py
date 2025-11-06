# ============================================================
# Visualizations.py 
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv("Data_Processed/combined_daily.csv")

# Correlation heatmap (2 features)
plt.figure(figsize=(4,4))
sns.heatmap(df[["steps_total","distance_total_km"]].corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Feature Correlation Heatmap"); plt.tight_layout(); plt.show()

# Train small models for confusion matrices
X = df[["steps_total","distance_total_km"]]   # or just ["distance_total_km"] if LEAK_FREE desired
y = df["activity_level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)

lr  = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_s, y_train)

pred_lr  = lr.predict(X_test_s)
pred_knn = knn.predict(X_test_s)

labels = sorted(y.unique())
fig, axes = plt.subplots(1,2, figsize=(9,4))
sns.heatmap(confusion_matrix(y_test, pred_lr,  labels=labels), annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title("LogReg"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

sns.heatmap(confusion_matrix(y_test, pred_knn, labels=labels), annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=labels, yticklabels=labels, ax=axes[1])
axes[1].set_title("KNN"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

plt.tight_layout(); plt.show()

# Accuracy bar
acc_df = pd.DataFrame({
    "Model":["LogReg","KNN"],
    "Accuracy":[accuracy_score(y_test,pred_lr), accuracy_score(y_test,pred_knn)]
})
plt.figure(figsize=(5,4)); sns.barplot(data=acc_df, x="Model", y="Accuracy")
plt.ylim(0,1.05); plt.title("Model Accuracy"); plt.tight_layout(); plt.show()
