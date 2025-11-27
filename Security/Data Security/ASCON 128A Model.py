# AEAD visual benchmark (AES-GCM, ChaCha20-Poly1305, optional Ascon-128a/XChaCha20-Poly1305)
import os, time, secrets, csv
from statistics import median
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# --------- output folder (timestamped next to this script) ---------
BASE_DIR = Path(__file__).resolve().parent
RUN_ID = time.strftime("%Y%m%d-%H%M%S")
OUT_DIR = BASE_DIR / "outputs_aead" / RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

PT = os.urandom(4096)             # 4KB payload
AAD = b"date=2025-11-24,user=User1"
N  = 50

def bench(fn, n=N):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), median(times)

def kbps(byte_len, seconds):
    return (byte_len/seconds)/1024

rows = []

# --- AES-GCM ---
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    aes_key = AESGCM.generate_key(bit_length=128)
    aes = AESGCM(aes_key)
    def aesgcm_once():
        nonce = secrets.token_bytes(12)  # unique per key
        aes.encrypt(nonce, PT, AAD)
    tmin, tmed = bench(aesgcm_once)
    rows.append(["AES-GCM", len(PT), round(tmin*1e6,2), round(tmed*1e6,2), round(kbps(len(PT), tmed),2)])
except Exception:
    rows.append(["AES-GCM", len(PT), None, None, None])

# --- ChaCha20-Poly1305 ---
try:
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    ch_key = ChaCha20Poly1305.generate_key()
    ch = ChaCha20Poly1305(ch_key)
    def chacha_once():
        nonce = secrets.token_bytes(12)  # unique per key
        ch.encrypt(nonce, PT, AAD)
    tmin, tmed = bench(chacha_once)
    rows.append(["ChaCha20-Poly1305", len(PT), round(tmin*1e6,2), round(tmed*1e6,2), round(kbps(len(PT), tmed),2)])
except Exception:
    rows.append(["ChaCha20-Poly1305", len(PT), None, None, None])

# --- Ascon-128a (if available), else XChaCha20-Poly1305 ---
added_optional = False
try:
    import ascon
    asc_key = os.urandom(16)  # 128-bit key
    def ascon_once():
        nonce = secrets.token_bytes(16)  # unique per key
        ascon.encrypt(asc_key, nonce, PT, AAD, variant="ascon-128a")
    tmin, tmed = bench(ascon_once)
    rows.append(["Ascon-128a", len(PT), round(tmin*1e6,2), round(tmed*1e6,2), round(kbps(len(PT), tmed),2)])
    added_optional = True
except Exception:
    pass

if not added_optional:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import XChaCha20Poly1305
        x_key = XChaCha20Poly1305.generate_key()
        xch = XChaCha20Poly1305(x_key)
        def xchacha_once():
            nonce = secrets.token_bytes(24)  # 192-bit nonce
            xch.encrypt(nonce, PT, AAD)
        tmin, tmed = bench(xchacha_once)
        rows.append(["XChaCha20-Poly1305", len(PT), round(tmin*1e6,2), round(tmed*1e6,2), round(kbps(len(PT), tmed),2)])
    except Exception:
        pass

# ---- save CSV ----
csv_path = OUT_DIR / "aead_bench.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["algorithm","payload_bytes","min_us","median_us","throughput_kBps"])
    w.writerows(rows)

# ---- visuals ----
df = pd.DataFrame(rows, columns=["algorithm","payload_bytes","min_us","median_us","throughput_kBps"])
df_avail = df.dropna(subset=["median_us"]).copy()

# chart 1: median time
if not df_avail.empty:
    fig = plt.figure(figsize=(6,4), dpi=140)
    plt.bar(df_avail["algorithm"], df_avail["median_us"])
    plt.title("AEAD: Median Encrypt Time (4KB)")
    plt.ylabel("µs (lower is better)")
    plt.xlabel("Algorithm")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "aead_median_time.png")
    plt.show()

    # chart 2: throughput
    fig = plt.figure(figsize=(6,4), dpi=140)
    plt.bar(df_avail["algorithm"], df_avail["throughput_kBps"])
    plt.title("AEAD: Throughput (4KB payload)")
    plt.ylabel("kB/s (higher is better)")
    plt.xlabel("Algorithm")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "aead_throughput.png")
    plt.show()

print("Saved:", csv_path)
print("Folder:", OUT_DIR)
