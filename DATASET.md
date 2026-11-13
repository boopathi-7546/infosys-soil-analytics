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
    black_soil/        <- High organic / vertisol samples
    red_soil/          <- Iron-rich alfisol / lateritic samples
    alluvial_soil/     <- Silt / loamy riverine soils
    laterite_soil/     <- Deep weathering tropical soils
    sandy_soil/        <- Coarse textured, low retention soils
    clay_soil/         <- Fine textured, heavy clay soils
  processed/           <- Preprocessed 224x224 RGB images
  sources.csv          <- Attribution, licenses, and catalog records
```

---

## 2. Public Dataset Sources & Licensing

All public data must be documented in `data/sources.csv`:

| Source | Platform | License | Categories | Description |
|---|---|---|---|---|
| Soil Types Classification | Kaggle | CC0: Public Domain | Black, Red, Alluvial, Clay | Diverse field captures across Indian agro-ecological zones |
| Soil Type Identification | Roboflow Universe | CC BY 4.0 | Sandy, Laterite, Black | Annotated multi-angle soil imagery under natural sunlight |
| NBSS&LUP Soil Resource Database | ICAR (Govt. of India) | Open Data (India) | Alluvial, Black, Red, Laterite | Authoritative morphological benchmarks |
| Soil Survey Image Gallery | USDA-NRCS | Public Domain | Sandy, Clay, Alluvial | Standard soil pedology reference imagery |

---

## 3. How to Add New Labelled Soil Images

1. Take photographs using an Android mobile phone or digital camera in natural, indirect daylight.
2. Hold the camera perpendicular to the soil surface at a distance of **15–25 cm**.
3. Clear away large stones, plant foliage, and debris from the frame.
4. Save the files into the appropriate class directory:
   ```text
   data/raw/black_soil/img_001.jpg
   data/raw/red_soil/img_001.jpg
   ```
5. Update `data/sources.csv` with the collection date, contributor name, and source notes.

---

## 4. Building the Preprocessed Dataset

Run the automated preparation script to validate, resize, and partition data without data leakage:

```bash
python scripts/prepare_dataset.py --raw data/raw --out data/processed --augment 3
```

### Pipeline Safety Guards:
- **No Data Leakage**: Train/validation/test split (70/15/15) is executed **before** any image augmentation.
- **Deduplication**: Exact duplicate files are filtered via SHA-256 hashing.
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
