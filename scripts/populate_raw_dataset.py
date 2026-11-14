"""Populate data/raw with real soil images from verified datasets:
1. Phantom-fs Soil-Classification-Dataset (Orignal-Dataset: Alluvial, Black, Laterite, Red)
   License: CC BY-NC 4.0
2. Indian Soil Types Dataset (Kaggle: Clay, Alluvial, Black, Red)
   License: CC0: Public Domain
"""
import hashlib
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import cv2

RAW_DIR = Path("data/raw")
PHANTOM_ORIG = Path(os.environ.get("TEMP", r"C:\Users\BOOBATHI P\AppData\Local\Temp")) / "phantom_soil" / "Orignal-Dataset"
VIDYUL_ZIP = Path(os.environ.get("TEMP", r"C:\Users\BOOBATHI P\AppData\Local\Temp")) / "vidyul_soil" / "Soil Types Dataset.zip"

MIN_SIDE = 100

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

stats = Counter()
seen_hashes = set()

# Process Clay Soil from Indian Soil Types Dataset (Kaggle / CC0)
clay_dir = RAW_DIR / "clay_soil"
clay_dir.mkdir(parents=True, exist_ok=True)

if VIDYUL_ZIP.exists():
    print("Processing Clay Soil from Soil Types Dataset (Kaggle CC0)...")
    with zipfile.ZipFile(VIDYUL_ZIP) as z:
        clay_names = [n for n in z.namelist() if "Clay soil" in n and n.lower().endswith(('.jpg', '.jpeg', '.png'))]
        c_idx = 1
        for n in sorted(clay_names):
            data = z.read(n)
            h = sha256_bytes(data)
            if h in seen_hashes:
                stats["clay_duplicate"] += 1
                continue
            # Validate OpenCV decodability
            import numpy as np
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                stats["clay_corrupt"] += 1
                continue
            if min(img.shape[:2]) < MIN_SIDE:
                stats["clay_too_small"] += 1
                continue
            seen_hashes.add(h)
            out_path = clay_dir / f"clay_{c_idx:04d}.jpg"
            with open(out_path, "wb") as f:
                f.write(data)
            c_idx += 1
            stats["clay_soil"] += 1
    print(f"-> Populated {stats['clay_soil']} clay_soil images")
else:
    print(f"Warning: {VIDYUL_ZIP} not found")

# Process Alluvial, Black, Laterite, Red from Phantom-fs Orignal-Dataset (CC BY-NC 4.0)
phantom_map = {
    "Alluvial_Soil": "alluvial_soil",
    "Black_Soil": "black_soil",
    "Laterite_Soil": "laterite_soil",
    "Red_Soil": "red_soil",
}

if PHANTOM_ORIG.exists():
    print("Processing from Phantom-fs Orignal-Dataset (CC BY-NC 4.0)...")
    for src_folder, target_folder in phantom_map.items():
        src_path = PHANTOM_ORIG / src_folder
        tgt_path = RAW_DIR / target_folder
        tgt_path.mkdir(parents=True, exist_ok=True)
        if not src_path.exists():
            print(f"Warning: {src_path} does not exist")
            continue
        idx = 1
        for f in sorted(src_path.iterdir()):
            if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            data = f.read_bytes()
            h = sha256_bytes(data)
            if h in seen_hashes:
                stats[f"{target_folder}_dup"] += 1
                continue
            import numpy as np
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                stats[f"{target_folder}_corrupt"] += 1
                continue
            if min(img.shape[:2]) < MIN_SIDE:
                stats[f"{target_folder}_too_small"] += 1
                continue
            seen_hashes.add(h)
            out_path = tgt_path / f"{target_folder}_{idx:04d}.jpg"
            with open(out_path, "wb") as out_f:
                out_f.write(data)
            idx += 1
            stats[target_folder] += 1
        print(f"-> Populated {stats[target_folder]} {target_folder} images")
else:
    print(f"Warning: {PHANTOM_ORIG} not found")

# Also add additional benchmark Alluvial, Black, Red images from Kaggle CC0 if present
if VIDYUL_ZIP.exists():
    with zipfile.ZipFile(VIDYUL_ZIP) as z:
        add_map = {
            "Alluvial soil": "alluvial_soil",
            "Black Soil": "black_soil",
            "Red soil": "red_soil",
        }
        for zip_key, target_folder in add_map.items():
            names = [n for n in z.namelist() if zip_key in n and n.lower().endswith(('.jpg', '.jpeg', '.png'))]
            added = 0
            curr_max = stats[target_folder]
            for n in sorted(names):
                data = z.read(n)
                h = sha256_bytes(data)
                if h in seen_hashes:
                    stats[f"{target_folder}_dup"] += 1
                    continue
                import numpy as np
                arr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None or min(img.shape[:2]) < MIN_SIDE:
                    continue
                seen_hashes.add(h)
                curr_max += 1
                out_path = RAW_DIR / target_folder / f"{target_folder}_{curr_max:04d}.jpg"
                with open(out_path, "wb") as out_f:
                    out_f.write(data)
                added += 1
                stats[target_folder] += 1
            print(f"-> Added {added} additional benchmark images for {target_folder} (Total: {stats[target_folder]})")

print("\nDataset population summary:")
for k, v in stats.items():
    print(f"  {k}: {v}")
