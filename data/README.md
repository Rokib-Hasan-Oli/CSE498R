# Bangla License Plate Datasets Collection

A curated collection of benchmark datasets for Bangla Automatic License Plate Recognition (ALPR), vehicle detection, and character recognition under diverse environmental conditions.

---

## 📊 Dataset Overview

| Dataset | Size | Key Notes & Coverage | License | Direct Link |
| :--- | :--- | :--- | :--- | :--- |
| **Bangla License Plate Dataset 2.5k** *(Zenodo, 2022)* | 5,259 images | Standard train/test split for detector benchmarking | CC BY 4.0 | [Zenodo Record](https://zenodo.org/records/7110401) |
| **Vehicle License Plate Detection Dataset** *(Mendeley / DIU, 2025)* | 2,947 images (5 classes) | Real-world traffic scenes from Dhaka, Ashulia, and Narsingdi | CC BY 4.0 | [Mendeley Data](https://data.mendeley.com/datasets/nntnnzffw4/3) |
| **Foggy License Plates Worldwide** *(Mendeley)* | 4,420 images (2,754 BD) | Heavy fog and low-visibility conditions; robustness benchmarking | Open Access | [Mendeley Data](https://data.mendeley.com/datasets/rgpddwxrx5/1) |
| **BD-ALPDR** | 725 images | Multi-view vehicle imagery captured across 6 distinct angles | CC BY-NC-SA 4.0 | [Project Page](https://bdalpdr.github.io/) • [GitHub](https://github.com/snooruddin/BD-ALPDR) |

---

## 🗂️ Dataset Details

### 1. Bangla License Plate Dataset 2.5k (Zenodo, 2022)
* **Volume:** 5,259 images
* **Split:** Pre-partitioned into training and testing sets.
* **Intended Use:** Baseline training and evaluation for vehicle detection and plate localization models.
* **Access:** [Download via Zenodo](https://zenodo.org/records/7110401)

### 2. Vehicle License Plate Detection Dataset (DIU / Mendeley, 2025)
* **Volume:** 2,947 annotated images across 5 distinct vehicle classes.
* **Geographical Scope:** Urban and suburban road networks across Dhaka, Ashulia, and Narsingdi.
* **Intended Use:** Vehicle bounding box regression, category classification, and license plate segmentation in dense South Asian traffic.
* **Access:** [Download via Mendeley Data](https://data.mendeley.com/datasets/nntnnzffw4/3)

### 3. Foggy License Plates Worldwide (Mendeley)
* **Volume:** 4,420 total images, containing **2,754 Bangladeshi license plates**.
* **Conditions:** Captured under adverse weather, synthetic/natural fog, haze, and low-contrast lighting.
* **Intended Use:** Stress-testing detection and character recognition pipelines for environmental robustness.
* **Access:** [Download via Mendeley Data](https://data.mendeley.com/datasets/rgpddwxrx5/1)

### 4. BD-ALPDR
* **Volume:** 725 curated images.
* **Feature:** Multi-perspective captures covering 6 distinct camera angles.
* **Intended Use:** Perspective-invariant plate extraction, rectification, and alphanumeric character segmentation.
* **Access:** [GitHub Repository](https://github.com/snooruddin/BD-ALPDR) | [Official Homepage](https://bdalpdr.github.io/)

---

## ⚖️ Usage & Licensing Notice

* Ensure compliance with the respective licenses when using these datasets for research or production:
  * **CC BY 4.0:** Requires attribution to the original dataset authors.
  * **CC BY-NC-SA 4.0:** Restricted to non-commercial use with reciprocal licensing terms.
