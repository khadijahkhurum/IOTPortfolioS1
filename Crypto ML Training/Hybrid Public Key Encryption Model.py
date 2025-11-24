"""
HPKE-compatible (Base mode) demo with visuals, no external hpke package needed.
KEM:   X25519
KDF:   HKDF-SHA256
AEAD:  ChaCha20-Poly1305

Outputs (timestamped folder next to this script):
  Crypto/outputs_hpke/<YYYYMMDD-HHMMSS>/
    - hpke_bench.csv
    - hpke_seal_open_median_us.png
    - hpke_seal_open_min_us.png
"""

import os, time
from statistics import median
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
import secrets

# -------- Output folder (timestamped) --------
HERE = Path(__file__).resolve().parent
RUN_ID = time.strftime("%Y%m%d-%H%M%S")
OUT_DIR = HERE / "outputs_hpke" / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- Config --------
N_RUNS  = 200                 # iterations for timing
CEK_LEN = 32                  # 32-byte content-encryption key
AAD     = b"day=batch-001"    # associated data

def hkdf_derive(shared_secret: bytes, enc_pub_bytes: bytes, info: bytes) -> bytes:
    """Derive an AEAD key from ECDH shared secret (HPKE-style context)."""
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,            # 256-bit AEAD key
        salt=None,            # Base mode (no PSK)
        info=enc_pub_bytes + info,  # bind encap pub + context
    )
    return hkdf.derive(shared_secret)

def hpke_seal(pkR: x25519.X25519PublicKey, cek: bytes, aad: bytes):
    """
    Sender:
      - generate ephemeral skE, compute dh = skE.exchange(pkR)
      - derive AEAD key via HKDF
      - encrypt CEK with ChaCha20-Poly1305
    Returns: enc (ephemeral public), nonce, ciphertext
    """
    skE = x25519.X25519PrivateKey.generate()
    enc = skE.public_key()
    dh  = skE.exchange(pkR)
    aead_key = hkdf_derive(dh, enc.public_bytes_raw(), b"HPKE-Base-Mode")
    aead = ChaCha20Poly1305(aead_key)
    nonce = secrets.token_bytes(12)
    ct = aead.encrypt(nonce, cek, aad)
    return enc, nonce, ct

def hpke_open(skR: x25519.X25519PrivateKey, enc_pub: x25519.X25519PublicKey, nonce: bytes, ct: bytes, aad: bytes):
    """
    Receiver:
      - compute dh = skR.exchange(enc_pub)
      - derive same AEAD key
      - decrypt CEK
    """
    dh = skR.exchange(enc_pub)
    aead_key = hkdf_derive(dh, enc_pub.public_bytes_raw(), b"HPKE-Base-Mode")
    aead = ChaCha20Poly1305(aead_key)
    return aead.decrypt(nonce, ct, aad)

def bench(fn, n=N_RUNS):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), median(times)

# Receiver static keypair (gateway)
skR = x25519.X25519PrivateKey.generate()
pkR = skR.public_key()

# One CEK for timing demo (rotate per batch/day in real use)
CEK = os.urandom(CEK_LEN)

# Warm-up
enc, nonce, ct = hpke_seal(pkR, CEK, AAD)
_ = hpke_open(skR, enc, nonce, ct, AAD)

def seal_once():
    hpke_seal(pkR, CEK, AAD)

def open_once():
    hpke_open(skR, enc, nonce, ct, AAD)

# Timings
seal_min, seal_med = bench(seal_once)
open_min, open_med = bench(open_once)

# Save CSV
df = pd.DataFrame(
    [
        {"suite": "HPKE-compatible (X25519/HKDF-SHA256/ChaCha20-Poly1305)",
         "op": "seal", "cek_bytes": CEK_LEN,
         "min_us": round(seal_min * 1e6, 2), "median_us": round(seal_med * 1e6, 2)},
        {"suite": "HPKE-compatible (X25519/HKDF-SHA256/ChaCha20-Poly1305)",
         "op": "open", "cek_bytes": CEK_LEN,
         "min_us": round(open_min * 1e6, 2), "median_us": round(open_med * 1e6, 2)},
    ]
)
csv_path = OUT_DIR / "hpke_bench.csv"
df.to_csv(csv_path, index=False)

# Visuals
fig = plt.figure(figsize=(6,4), dpi=140)
plt.bar(["seal", "open"], [df.loc[df.op=="seal","median_us"].iloc[0],
                           df.loc[df.op=="open","median_us"].iloc[0]])
plt.title("HPKE-compatible (X25519/HKDF-SHA256/ChaCha20-Poly1305)\nMedian time for 32-byte CEK")
plt.ylabel("µs (lower is better)")
plt.xlabel("Operation")
plt.tight_layout()
plt.savefig(OUT_DIR / "hpke_seal_open_median_us.png")
plt.show()

fig = plt.figure(figsize=(6,4), dpi=140)
plt.bar(["seal", "open"], [df.loc[df.op=="seal","min_us"].iloc[0],
                           df.loc[df.op=="open","min_us"].iloc[0]])
plt.title("HPKE-compatible — Min time (best of runs)")
plt.ylabel("µs")
plt.xlabel("Operation")
plt.tight_layout()
plt.savefig(OUT_DIR / "hpke_seal_open_min_us.png")
plt.show()

print("Saved:", csv_path)
print("Folder:", OUT_DIR)
