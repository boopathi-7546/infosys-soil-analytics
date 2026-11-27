"""Download verified real photographic sandy soil images from Wikimedia Commons / Open Source.
Uses 800px high-resolution thumburl from Wikimedia CDN for fast, reliable ingestion.
Records source URL, title, author, and license.
"""
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
import cv2
import numpy as np

DEST_DIR = Path("data/raw/sandy_soil")
DEST_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "SoilSenseAI-Research/1.0 (Agricultural Soil Analytics Dataset Prep; contact: boopathi)"

QUERIES = [
    "sandy soil",
    "sandy soil ground",
    "sandy soil texture",
    "sand soil surface",
    "Arenosol soil",
    "sandy agricultural soil",
    "sandy loam soil",
]

EXCLUDE_TERMS = ["map", "diagram", "chart", "graph", "icon", "drawing", "illustration", "logo", "flag", "pdf", "svg"]

downloaded_records = []
seen_hashes = set()

# Pre-populate seen_hashes with already existing files
for f in DEST_DIR.glob("*.jpg"):
    
    seen_hashes.add(hashlib.sha256(f.read_bytes()).hexdigest())

count = len([f for f in DEST_DIR.glob("*.jpg")])
print(f"Starting with {count} already downloaded sandy soil images...")

for query in QUERIES:
    if count >= 60:
        break
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&generator=search"
        f"&gsrsearch={urllib.parse.quote(query)}"
        f"&gsrnamespace=6&prop=imageinfo&iiprop=url|size|mime|extmetadata"
        f"&iiurlwidth=800&format=json&gsrlimit=50"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            data = json.loads(res.read().decode("utf-8"))
    except Exception as e:
        print(f"Failed query {query}: {e}")
        continue

    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if count >= 60:
            break
        title = page.get("title", "")
        ii = page.get("imageinfo", [{}])[0]
        mime = ii.get("mime", "")
        if mime not in ("image/jpeg", "image/png"):
            continue

        title_lower = title.lower()
        if any(term in title_lower for term in EXCLUDE_TERMS):
            continue

        # Use 800px thumburl for quick high-quality download, fallback to full url
        file_url = ii.get("thumburl") or ii.get("url")
        if not file_url:
            continue

        ext_meta = ii.get("extmetadata", {})
        license_name = ext_meta.get("LicenseShortName", {}).get("value", "CC BY / Open License")
        artist = ext_meta.get("Artist", {}).get("value", "Wikimedia Contributor")
        artist_clean = re.sub(r"<[^>]+>", "", artist).strip()[:50]

        try:
            down_req = urllib.request.Request(file_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(down_req, timeout=10) as down_res:
                img_bytes = down_res.read()

            sha = hashlib.sha256(img_bytes).hexdigest()
            if sha in seen_hashes:
                continue
            seen_hashes.add(sha)

            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None or min(img.shape[:2]) < 100:
                continue

            count += 1
            safe_name = f"sandy_{count:04d}.jpg"
            dest_path = DEST_DIR / safe_name
            with open(dest_path, "wb") as f:
                f.write(img_bytes)

            downloaded_records.append({
                "filename": safe_name,
                "title": title,
                "url": ii.get("descriptionurl") or file_url,
                "license": license_name,
                "author": artist_clean,
                "shape": f"{img.shape[1]}x{img.shape[0]}",
            })
            print(f"[{count:02d}] Saved {safe_name} ({img.shape[1]}x{img.shape[0]}) - {license_name} - {title}")
        except Exception:
            continue

print(f"\nCompleted! Total sandy soil images: {count} in {DEST_DIR}")

meta_file = Path("data/raw/sandy_soil_metadata.json")
with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(downloaded_records, f, indent=2)
