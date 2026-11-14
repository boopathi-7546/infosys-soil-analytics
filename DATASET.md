# Soil Dataset Architecture & Collection Guide

This guide describes the dataset structure, public source attribution, OpenCV ingestion pipeline, and procedures for collecting real-world soil images.

> **CRITICAL SCIENTIFIC PRINCIPLE:**
> - A soil photograph captures **visual morphology, surface texture, particle aggregation, colour-space indicators, and moisture status**.
> - Exact quantitative chemical concentrations of Nitrogen (N), Phosphorus (P), and Potassium (K) require standard **laboratory chemical analysis** (e.g. Kjeldahl digestion, Bray/Olsen extraction, flame photometry).
> - The AI model classifies soil taxonomy and physical state, while the advisory engine combines image analysis with laboratory test measurements.

---

## 1. Directory Structure

```text
data/
  raw/
    alluvial_soil/     <- Silt / loamy riverine soils (342 images)
    black_soil/        <- High organic / vertisol samples (355 images)
    clay_soil/         <- Fine-textured, heavy clay soils (119 images)
    laterite_soil/     <- Deep weathering tropical soils (214 images)
    red_soil/          <- Iron-rich alfisol / lateritic samples (265 images)
    sandy_soil/        <- Coarse textured, low retention soils (60 images)
  processed/
    manifest.csv       <- Complete inventory mapping splits, hashes, and augmentation kinds
    train/             <- Training split (70% original + 3x augmented copies per image)
    val/               <- Validation split (15% original images)
    test/              <- Test split (15% original images)
    7_proc.jpg         <- Pre-existing system test sample (preserved)
  sources.csv          <- Attribution, licenses, and catalog records
```

---

## 2. Public Dataset Sources & Licensing

All public data is documented in `data/sources.csv`:

| Source | Platform | License | Categories | Downloaded | Description |
|---|---|---|---|---|---|
| Soil Classification Research Dataset (Phantom-fs / Sheth et al. 2025) | GitHub | CC BY-NC 4.0 | Alluvial, Black, Laterite, Red | 900 | Peer-reviewed agricultural research dataset (*Engineering Applications of AI*) |
| Soil Classification Dataset (Prasansha Satpathy) | Kaggle | CC0: Public Domain | Clay, Black, Alluvial, Red | 395 | Benchmark Indian soil dataset (119 Clay, 130 Black, 117 Alluvial, 40 Red) |
| Wikimedia Commons Soil Pedology Archive | Wikimedia | Open Licenses (CC BY-SA / CC BY / PD) | Sandy Soil | 60 | Field captures of Arenosols and sandy agricultural soils with verified attribution |
| Soil Type Identification | Roboflow Universe | CC BY 4.0 | Sandy, Laterite, Black | 0 | Requires private Roboflow API key / account; skipped to strictly comply with sanctioned auth policies |
| ICAR-NBSS&LUP Soil Resource Database | ICAR (Govt. of India) | Government Open Data | Alluvial, Black, Red, Laterite | 0 | Requires institutional login and manual clearance; documented without bypassing |
| USDA-NRCS Soil Survey Image Gallery | USDA-NRCS | Public Domain (US Gov) | Sandy, Clay, Alluvial | 0 | Reference taxonomy archives for pedon morphology validation |

---

## 3. Dataset Distribution & Stratified Partitioning

The raw dataset contains **1,355 unique, real photographic soil images** distributed across all 6 target classes.
When partitioned with `scripts/prepare_dataset.py` (70% Train, 15% Val, 15% Test with `--augment 3`):

| Soil Category | Raw Originals | Train (Original) | Val (Original) | Test (Original) | Train (Augmented) | Total Processed |
|---|---|---|---|---|---|---|
| **Alluvial Soil** | 342 | 239 | 51 | 52 | 717 | 1,059 |
| **Black Soil** | 355 | 248 | 53 | 54 | 744 | 1,099 |
| **Clay Soil** | 119 | 83 | 18 | 18 | 249 | 368 |
| **Laterite Soil** | 214 | 149 | 32 | 33 | 447 | 661 |
| **Red Soil** | 265 | 185 | 40 | 40 | 555 | 820 |
| **Sandy Soil** | 60 | 42 | 9 | 9 | 126 | 186 |
| **Total** | **1,355** | **946** | **203** | **206** | **2,838** | **4,193** (+ 1 preserved sample) |

---

## 4. Building the Preprocessed Dataset

Run the automated preparation script to validate, resize, and partition data without data leakage:

```bash
python scripts/prepare_dataset.py --raw data/raw --out data/processed --augment 3
```

### Pipeline Safety Guards:
- **Zero Data Leakage**: Train/validation/test split (70/15/15) is executed **before** any image augmentation. No original source image appears in more than one partition, and all augmented images exist solely in `train/`.
- **Deduplication**: Exact duplicate files are filtered via SHA-256 hashing.
- **Corrupt File Filter**: Images failing OpenCV decoding are discarded.
- **Decompression Bomb Guard**: Rejects malformed images exceeding 50 million pixels.
- **Laplacian Variance Quality Filter**: Rejects unreadable or severely blurred images (< 40 variance).
- **OpenCV Normalization**: Applies partial Gray-World color balancing, CLAHE illumination correction, and bilinear resizing to 224 × 224 pixels.

---

## 5. Live Dataset API

The system provides real-time dataset health tracking via the REST endpoint:
```http
GET /api/dataset/status
```

Returns:
- Image count per soil class
- Total raw and processed image tallies
- Full catalog of sources and licenses from `data/sources.csv`
- Dynamic rendering in the dashboard home page.
