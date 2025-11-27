# crypto_oscore_cose_visual.py
"""
OSCORE-style object security (visual) without external COSE deps.
We mimic COSE_Encrypt0 semantics using ChaCha20-Poly1305:
- Protect = encrypt(payload, aad=external_context)
- Verify  = decrypt(... same external AAD)
We also include a Partial IV (piv) to illustrate replay protection.

Outputs:
  Crypto/outputs_oscore/<YYYYMMDD-HHMMSS>/
    - oscore_bench.csv
    - oscore_median_us.png
    - oscore_min_us.png
"""

import os, time, secrets
from statistics import median
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

HERE   = Path(__file__).resolve().parent
RUN_ID = time.strftime("%Y%m%d-%H%M%S")
OUT    = HERE / "outputs_oscore" / RUN_ID
OUT.mkdir(parents=True, exist_ok=True)

# CoAP-like payload + external AAD (authenticated metadata)
payload = (
    b"steps=1234,distance_km=0.98,activity=Medium,"
    b"user=User1,date=2025-11-24"
)
ext_aad = b"coap:/activity?u=User1&d=2025-11-24"  # authenticated, not encrypted
piv     = secrets.token_bytes(5)                  # Partial IV (nonce component)
N       = 300                                     # iterations

# Derive a per-session AEAD key (here we just generate one)
key  = ChaCha20Poly1305.generate_key()
aead = ChaCha20Poly1305(key)

def protect_once():
    # Compose a nonce from PIV + random padding to 12 bytes
    nonce = piv + secrets.token_bytes(12 - len(piv))
    _ = aead.encrypt(nonce, payload, ext_aad)

# Create one blob to time verify()
nonce0 = piv + secrets.token_bytes(12 - len(piv))
blob0  = aead.encrypt(nonce0, payload, ext_aad)

def verify_once():
    _ = aead.decrypt(nonce0, blob0, ext_aad)

def bench(fn, n=N):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), median(ts)

pmin, pmed = bench(protect_once)
vmin, vmed = bench(verify_once)

# Save CSV
df = pd.DataFrame([{
    "object": "OSCORE-style (ChaCha20-Poly1305, external AAD + Partial IV)",
    "payload_bytes": len(payload),
    "protect_min_us": round(pmin*1e6, 2),
    "protect_median_us": round(pmed*1e6, 2),
    "verify_min_us": round(vmin*1e6, 2),
    "verify_median_us": round(vmed*1e6, 2)
}])
csv_path = OUT / "oscore_bench.csv"
df.to_csv(csv_path, index=False)

# Charts
fig = plt.figure(figsize=(7,4), dpi=140)
plt.bar(["protect","verify"], [df["protect_median_us"].iloc[0], df["verify_median_us"].iloc[0]])
plt.title(f"{df['object'].iloc[0]} — Median time (~{df['payload_bytes'].iloc[0]}B)")
plt.ylabel("µs (lower is better)"); plt.xlabel("Operation")
plt.tight_layout(); plt.savefig(OUT / "oscore_median_us.png"); plt.show()

fig = plt.figure(figsize=(7,4), dpi=140)
plt.bar(["protect","verify"], [df["protect_min_us"].iloc[0], df["verify_min_us"].iloc[0]])
plt.title(f"{df['object'].iloc[0]} — Min time (best of runs)")
plt.ylabel("µs"); plt.xlabel("Operation")
plt.tight_layout(); plt.savefig(OUT / "oscore_min_us.png"); plt.show()

print("Saved:", csv_path)
print("Folder:", OUT)
