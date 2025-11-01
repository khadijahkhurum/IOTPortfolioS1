# ============================================================
# Trend of Steps and Average Heart Rate Over Time
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Data_Processed/combined_daily.csv")
df["date"] = pd.to_datetime(df["date"])

# Sort by date
df = df.sort_values("date")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df["date"], df["steps_total"], label="Steps", color="blue", linewidth=2)
plt.plot(df["date"], df["avg_hr"], label="Avg Heart Rate", color="red", linewidth=2)

plt.title("Daily Steps and Average Heart Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
