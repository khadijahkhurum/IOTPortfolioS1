import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LEAK_FREE = False  # True -> use ONLY distance_total_km as feature

print("Loading dataset...")
df = pd.read_csv("Data_Processed/combined_daily.csv")

print("\nPreview of dataset:")
print(df.head())

# ---- SELECT FEATURES ----
if LEAK_FREE:
    X = df[["distance_total_km"]]        # safer, no label leakage
else:
    X = df[["steps_total", "distance_total_km"]]  # simple & accurate

y = df["activity_level"]

# ---- SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---- SCALE ----
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ---- MODEL ----
print("\nTraining Logistic Regression model...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_s, y_train)
pred_lr = lr.predict(X_test_s)

# ---- RESULTS ----
print("\n==================== RESULTS ====================")
acc_lr = accuracy_score(y_test, pred_lr)

print(f"\n Logistic Regression Accuracy: {acc_lr:.3f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_lr))
print("\nClassification Report:\n", classification_report(y_test, pred_lr))
