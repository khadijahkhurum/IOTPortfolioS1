# ============================================================
#  Logistic Regression & KNN
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. LOAD DATA ------------------------------------------------
print("Loading dataset...")
df = pd.read_csv("Data_Processed/combined_daily.csv")

print("\nPreview of dataset:")
print(df.head())

# 2. SELECT FEATURES AND LABEL --------------------------------
X = df[["steps_total", "distance_total_km", "avg_hr", "sleep_hours"]]
y = df["activity_level"]

# 3. SPLIT INTO TRAIN AND TEST --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. SCALE DATA -----------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. MODEL 1: LOGISTIC REGRESSION ------------------------------
print("\nTraining Logistic Regression model...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# 6. MODEL 2: K-NEAREST NEIGHBORS ------------------------------
print("\nTraining K-Nearest Neighbors model...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# 7. EVALUATE BOTH MODELS -------------------------------------
print("\n==================== RESULTS ====================")

# Logistic Regression
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"\n🔹 Logistic Regression Accuracy: {acc_lr:.3f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# KNN
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"\n🔹 K-Nearest Neighbors Accuracy: {acc_knn:.3f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))
