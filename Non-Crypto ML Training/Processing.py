# ============================================================
# Processing.py
# Aggregates all users’ data into daily summaries
# ============================================================

import pandas as pd
import numpy as np
import os

input_files = [
    "Data_Converted/User1_filtered.csv",
    "Data_Converted/User2_filtered.csv",
    "Data_Converted/User3_filtered.csv"
]

output_dir = "Data_Processed"
os.makedirs(output_dir, exist_ok=True)

def make_daily(df_raw):
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["start"]).dt.date
    steps = df[df["metric"]=="steps"].groupby("date")["value"].sum().rename("steps_total")
    dist  = df[df["metric"]=="distance"].groupby("date")["value"].sum().rename("distance_total_km")
    daily = pd.concat([steps, dist], axis=1).reset_index()
    daily = daily.fillna(0)
    return daily

all_users = []

for file in input_files:
    if os.path.exists(file):
        user = os.path.basename(file).split("_")[0]  # "User1"
        df = pd.read_csv(file, parse_dates=["start", "end"])
        daily = make_daily(df)
        daily["user"] = user
        all_users.append(daily)
    else:
        print(f"⚠️ Missing file: {file}")

combined = pd.concat(all_users, ignore_index=True)

# Label activity level
combined["activity_level"] = pd.cut(
    combined["steps_total"],
    bins=[-1, 3000, 8000, np.inf],
    labels=["Low", "Medium", "High"]
)

# Save combined dataset
out_path = os.path.join(output_dir, "combined_daily.csv")
combined.to_csv(out_path, index=False)

print(f"✅ Saved combined dataset: {out_path} ({combined.shape[0]} rows, {combined.shape[1]} columns)")
