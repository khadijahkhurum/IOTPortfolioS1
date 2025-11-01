# ============================================================
# CompareUsers.py
# Compare IoT features (Steps, Distance, Sleep) between users
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Data_Processed/combined_daily.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["user", "date"])

# Keep only common useful features
features = ["steps_total", "distance_total_km", "sleep_hours"]

# ---------------- TREND: Steps & Distance ----------------
plt.figure(figsize=(10,5))
sns.lineplot(data=df, x="date", y="steps_total", hue="user", linewidth=1.3)
plt.title("Daily Steps Over Time — User1 vs User2")
plt.xlabel("Date")
plt.ylabel("Steps")
plt.legend(title="User")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
sns.lineplot(data=df, x="date", y="distance_total_km", hue="user", linewidth=1.3)
plt.title("Daily Distance (km) Over Time — User1 vs User2")
plt.xlabel("Date")
plt.ylabel("Distance (km)")
plt.legend(title="User")
plt.tight_layout()
plt.show()

# ---------------- AVERAGE COMPARISON ----------------
user_means = (
    df.groupby("user")[features]
    .mean()
    .reset_index()
    .melt(id_vars="user", var_name="Feature", value_name="Average")
)

plt.figure(figsize=(8,5))
sns.barplot(data=user_means, x="Feature", y="Average", hue="user", palette="Set2")
plt.title("Average IoT Metrics per User")
plt.xlabel("Feature")
plt.ylabel("Average Value")
plt.legend(title="User")
plt.tight_layout()
plt.show()

# ---------------- DISTRIBUTION COMPARISON ----------------
plt.figure(figsize=(8,5))
for f in features:
    sns.kdeplot(data=df, x=f, hue="user", fill=True, common_norm=False)
    plt.title(f"Distribution of {f.replace('_', ' ').title()} — User Comparison")
    plt.xlabel(f.replace('_', ' ').title())
    plt.tight_layout()
    plt.show()
