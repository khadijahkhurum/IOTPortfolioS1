# ============================================================
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "Data_Processed/combined_daily.csv"
PLOTS_DIR = "Plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------- LOAD ----------
df = pd.read_csv(DATA_PATH)
need = {"date", "user", "steps_total", "distance_total_km"}
missing = need - set(df.columns)
if missing:
    raise RuntimeError(f"Missing columns in {DATA_PATH}: {missing}")

# types & sort
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values(["user", "date"])

# Palette auto-scales with #users
users = df["user"].astype(str).unique()
palette = sns.color_palette("husl", len(users))

# ---------- 1) STEPS TREND ----------
plt.figure(figsize=(11,5))
sns.lineplot(data=df, x="date", y="steps_total", hue="user", palette=palette, linewidth=1.4, alpha=0.9)
plt.title("Daily Steps — User Comparison")
plt.xlabel("Date"); plt.ylabel("Total Steps")
plt.grid(True, alpha=0.3); plt.tight_layout()
out1 = os.path.join(PLOTS_DIR, "user_steps_trend.png")
plt.savefig(out1, dpi=200); plt.close()

# ---------- 2) DISTANCE TREND ----------
plt.figure(figsize=(11,5))
sns.lineplot(data=df, x="date", y="distance_total_km", hue="user", palette=palette, linewidth=1.4, alpha=0.9)
plt.title("Daily Distance (km) — User Comparison")
plt.xlabel("Date"); plt.ylabel("Distance (km)")
plt.grid(True, alpha=0.3); plt.tight_layout()
out2 = os.path.join(PLOTS_DIR, "user_distance_trend.png")
plt.savefig(out2, dpi=200); plt.close()

# ---------- 3) AVERAGE METRICS (grouped bar) ----------
user_means = (
    df.groupby("user")[["steps_total", "distance_total_km"]]
      .mean()
      .reset_index()
      .melt(id_vars="user", var_name="Feature", value_name="Average")
)
plt.figure(figsize=(8,5))
sns.barplot(data=user_means, x="Feature", y="Average", hue="user", palette=palette)
plt.title("Average Daily Metrics by User")
plt.xlabel("Feature"); plt.ylabel("Average Value")
plt.tight_layout()
out3 = os.path.join(PLOTS_DIR, "user_avg_metrics.png")
plt.savefig(out3, dpi=200); plt.close()

# ---------- 4) DISTRIBUTIONS (KDE) ----------
# Steps
plt.figure(figsize=(8,5))
sns.kdeplot(data=df, x="steps_total", hue="user", fill=True, common_norm=False, palette=palette)
plt.title("Distribution — Steps by User")
plt.xlabel("Total Steps"); plt.ylabel("Density")
plt.tight_layout()
out4a = os.path.join(PLOTS_DIR, "user_distribution_steps.png")
plt.savefig(out4a, dpi=200); plt.close()

# Distance
plt.figure(figsize=(8,5))
sns.kdeplot(data=df, x="distance_total_km", hue="user", fill=True, common_norm=False, palette=palette)
plt.title("Distribution — Distance (km) by User")
plt.xlabel("Distance (km)"); plt.ylabel("Density")
plt.tight_layout()
out4b = os.path.join(PLOTS_DIR, "user_distribution_distance.png")
plt.savefig(out4b, dpi=200); plt.close()

# ---------- 5) SCATTER: Steps vs Distance ----------
plt.figure(figsize=(7,6))
sns.scatterplot(data=df, x="distance_total_km", y="steps_total", hue="user", palette=palette, alpha=0.7, s=25, edgecolor="none")
plt.title("Steps vs Distance — Colored by User")
plt.xlabel("Distance (km)"); plt.ylabel("Total Steps")
plt.grid(True, alpha=0.3); plt.tight_layout()
out5 = os.path.join(PLOTS_DIR, "user_scatter_steps_vs_distance.png")
plt.savefig(out5, dpi=200); plt.close()

print("✅ Saved:")
for p in [out1, out2, out3, out4a, out4b, out5]:
    print("  -", p)
