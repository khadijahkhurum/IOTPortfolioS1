
# Extract "useful" Apple Health records from XML into clean CSVs (one per user)

import os, xml.etree.ElementTree as ET
import pandas as pd

# --- INPUT XML PATHS: update if your paths differ ---
USER_FILES = {
    "User1": "Datasets/User1 Data.xml",
    "User2": "Datasets/User2 Data.xml"
}

os.makedirs("Data_Converted", exist_ok=True)

USEFUL_TYPES = {
    "HKQuantityTypeIdentifierStepCount": "steps",
    "HKQuantityTypeIdentifierHeartRate": "heartrate",
    "HKQuantityTypeIdentifierDistanceWalkingRunning": "distance",
    "HKCategoryTypeIdentifierSleepAnalysis": "sleep"   # may or may not exist
}

def extract_to_csv(user, xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []

    for r in root.findall("Record"):
        r_type = r.get("type")
        if r_type not in USEFUL_TYPES:  # skip everything else
            continue
        rows.append({
            "metric": USEFUL_TYPES[r_type],
            "value": r.get("value"),
            "start": r.get("startDate"),
            "end": r.get("endDate"),
            "source": r.get("sourceName")
        })

    df = pd.DataFrame(rows)
    # tidy types
    df["start"] = pd.to_datetime(df["start"])
    df["end"]   = pd.to_datetime(df["end"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    out = f"Data_Converted/{user}_filtered.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out} ({len(df)} rows)")

if __name__ == "__main__":
    for user, path in USER_FILES.items():
        extract_to_csv(user, path)
