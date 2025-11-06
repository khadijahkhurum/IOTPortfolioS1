
import xml.etree.ElementTree as ET
import pandas as pd
import os

# ---------- CONFIG ----------
user_files = {
    "User1": "Datasets/User1 Data.xml",
    "User2": "Datasets/User2 Data.xml",
    "User3": "Datasets/User3 Data.xml"
}

output_dir = "Data_Converted"
os.makedirs(output_dir, exist_ok=True)

# ---------- FUNCTION TO EXTRACT RECORDS ----------
def extract_records(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    for record in root.findall("Record"):
        r_type = record.get("type")
        if r_type in [
            "HKQuantityTypeIdentifierStepCount",
            "HKQuantityTypeIdentifierDistanceWalkingRunning",
        ]:
            rows.append({
                "metric": "steps" if "Step" in r_type else "distance",
                "value": record.get("value"),
                "start": record.get("startDate"),
                "end": record.get("endDate"),
                "source": record.get("sourceName"),
            })
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    return df

# ---------- PROCESS EACH USER ----------
for user, path in user_files.items():
    if os.path.exists(path):
        print(f"Processing {user} ...")
        df = extract_records(path)
        out_path = os.path.join(output_dir, f"{user}_filtered.csv")
        df.to_csv(out_path, index=False)
        print(f"✅ Saved: {out_path} ({len(df)} rows)")
    else:
        print(f"⚠️ File not found for {user}: {path}")

print("\nAll XMLs processed successfully.")



