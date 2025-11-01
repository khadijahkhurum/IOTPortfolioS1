# Processing.py
# Build daily dataset from Apple Health CSVs (NO cryptography)

import os
import pandas as pd
import numpy as np

INPUT_USER1 = "Data_Converted/User1_filtered.csv"
INPUT_USER2 = "Data_Converted/User2_filtered.csv"
OUT_DIR = "Data_Processed"
os.makedirs(OUT_DIR, exist_ok=True)

def make_daily(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert event-level Apple Health rows into one row per day with simple features.
    Ensures a real 'date' column exists (never just 'index').
    """
    # Ensure timestamps & numeric
    df = df_raw.copy()
    if "start" in df.columns:
        df["start"] = pd.to_datetime(df["start"], errors="coerce")
    if "end" in df.columns:
        df["end"] = pd.to_datetime(df["end"], errors="coerce")
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # --- Create 'date' from 'start'
    df["date"] = df["start"].dt.date

    # --- Aggregate features ---
    # Steps / Distance: sum per day
    steps = (
        df[df["metric"] == "steps"]
        .groupby("date")["value"].sum()
        .rename("steps_total")
    )
    distance = (
        df[df["metric"] == "distance"]
        .groupby("date")["value"].sum()
        .rename("distance_total_km")
    )
    # Heart rate: average per day
    avg_hr = (
        df[df["metric"] == "heartrate"]
        .groupby("date")["value"].mean()
        .rename("avg_hr")
    )
    # Sleep: hours per day (if present)
    sleep_df = df[df["metric"] == "sleep"].copy()
    if not sleep_df.empty:
        sleep_hours = (sleep_df["end"] - sleep_df["start"]).dt.total_seconds() / 3600.0
        sleep = (
            pd.DataFrame({"date": sleep_df["date"], "hours": sleep_hours})
            .groupby("date")["hours"]
            .sum()
            .rename("sleep_hours")
        )
    else:
        sleep = pd.Series(dtype=float, name="sleep_hours")

    # Combine to a DataFrame and force a real 'date' column
    daily = pd.concat([steps, distance, avg_hr, sleep], axis=1)

    # If index has no name, reset_index() will call it 'index' – fix that:
    daily = daily.reset_index()
    if "index" in daily.columns and "date" not in daily.columns:
        daily = daily.rename(columns={"index": "date"})

    # Final sanity: ensure 'date' exists
    if "date" not in daily.columns:
        raise RuntimeError("Failed to create 'date' column in daily dataset.")

    return daily

def load_user_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path, parse_dates=["start", "end"])

# ---- Build daily tables for each user ----
u1 = load_user_csv(INPUT_USER1)
u2 = load_user_csv(INPUT_USER2)

d1 = make_daily(u1)
d2 = make_daily(u2)

d1["user"] = "User1"
d2["user"] = "User2"

# ---- Combine, clean, label ----
combined = pd.concat([d1, d2], ignore_index=True)

# Fill sensible defaults
for col, fill in [
    ("steps_total", 0.0),
    ("distance_total_km", 0.0),
    ("avg_hr", combined["avg_hr"].median() if not combined["avg_hr"].dropna().empty else 0.0),
    ("sleep_hours", combined["sleep_hours"].median() if "sleep_hours" in combined.columns and not combined["sleep_hours"].dropna().empty else 0.0),
]:
    if col in combined.columns:
        combined[col] = combined[col].fillna(fill)

# Sort safely
combined = combined.sort_values(["user", "date"]).reset_index(drop=True)

# Easy target label from steps (Low/Medium/High)
combined["activity_level"] = pd.cut(
    combined["steps_total"],
    bins=[-1, 3000, 8000, np.inf],  # <3k, 3k–8k, >8k
    labels=["Low", "Medium", "High"]
)

# Keep final columns (add missing defensively)
need_cols = ["user", "date", "steps_total", "distance_total_km", "avg_hr", "sleep_hours", "activity_level"]
have_cols = [c for c in need_cols if c in combined.columns]
combined = combined[have_cols]

out_path = os.path.join(OUT_DIR, "combined_daily.csv")
combined.to_csv(out_path, index=False)

print(f"✅ Saved daily dataset: {out_path}")
print("   Rows:", combined.shape[0], "Cols:", combined.shape[1])
print("\nPreview:")
print(combined.head(10).to_string(index=False))
