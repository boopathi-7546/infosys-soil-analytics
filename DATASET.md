# Dataset collection guide (Milestone 1)

**The repo contains no real soil images.** Real data has to be collected by you. The pipeline
(`scripts/prepare_dataset.py`) is ready and tested, but only on synthetic images made by
`scripts/make_sample_images.py`. Never train or report accuracy on those synthetic images.

## 1. Decide the labels first
A soil *photo* can reliably show soil type, texture, colour and moisture. It cannot reliably show
N/P/K levels (that comes from the lab values). So start with image classes such as:
`black_soil, red_soil, alluvial_soil, laterite_soil, sandy_soil, clay_soil`
and add condition classes (e.g. `dry_cracked`, `waterlogged`) only if you find labelled data.
Confirm the label scheme with your mentor.

## 2. Where to look
- Kaggle and Roboflow Universe: search "soil type image dataset" / "soil classification". Check each licence.
- ICAR institutes (e.g. NBSS&LUP) and state agricultural universities: soil survey reports and photos.
- USDA NRCS soil survey resources (the brief lists USDA/ICAR).
- Your own photos: the most valuable data because they match what farmers will upload. Use a
  phone, daylight, a flat patch of soil, camera ~30 cm above the ground, no shadows of your own body,
  no flash. Vary the field, time of day and moisture. Aim for 100+ per class.

## 3. Put files here
```
data/raw/<class_name>/photo1.jpg
data/raw/<class_name>/photo2.png
```
Fill in `data/sources.csv` for every batch (source, licence, count, date). Your documentation needs it.

## 4. Build the dataset
```
python scripts/prepare_dataset.py --raw data/raw --out data/processed --augment 3
```
This removes corrupt, tiny and duplicate files, splits 70/15/15 per class (before augmenting, to avoid
leakage), applies the OpenCV preprocessing, writes `data/processed/manifest.csv` and prints the
class balance. Classes with under 100 images are flagged.
