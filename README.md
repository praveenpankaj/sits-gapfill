# sits-gapfill
# Robust Satellite Image Time-Series Spectral Index Extraction & Gap Imputation using Prophet, Timestral, Linear Interpolation, Cubic Interpolation and Bi-LSTM

This repository contains an end-to-end cloud-native pipeline built with **Google Earth Engine (GEE)**, **PyTorch**, **Meta Prophet**, and **Mistral AI's Timestral**, **Bi-LSTM** to build a high-resolution consensus cropland mask, extract robust spatial-temporal spectral index profiles, and perform advanced temporal gap-filling on agricultural time-series data.

---

## 📌 Project Overview

In satellite remote sensing of agriculture, monitoring crop health is often hindered by two primary challenges:
1. **Classification Discrepancies**: Different global LULC (Land Use / Land Cover) products often disagree on cropland boundaries.
2. **Atmospheric Contamination**: Clouds and aerosols introduce persistent temporal data gaps in optical indices like NDVI.

This framework resolves both issues by:
* **A Spatial Consensus Engine**: Fusing five distinct global LULC products (**ESRI LULC**, **ESA WorldCover**, **Copernicus Land Cover**, **FROM-GLC10**, and **GLC-FCS30D**) using Earth Engine lazy-evaluation scales (no forced reprojection overhead).
* **Temporal Reconstruction & Imputation**: Preprocessing Sentinel-2 Surface Reflectance (with Cloud Score+ quality filtering) to extract 15-day composites, then comparing Classical Interpolation (Linear/Cubic Spline), **Meta Prophet**, **Timestral-209M**, and a custom **Deep Learning Bidirectional LSTM (Bi-LSTM)** temporal imputer designed to reconstruct cloud-induced gaps.

---

## 🛠️ Architecture & Methodology

### 1. Spatial Consensus Cropland Masking
Instead of relying on a single, potentially biased product, we combine five datasets. To prevent memory and projection limit issues (such as `EEException: Reprojection output too large`), we allow Google Earth Engine's lazy evaluation engine to handle the dynamic resolution scales:
* **ESRI 10m LULC**
* **ESA WorldCover 10m**
* **Copernicus 100m**
* **FROM-GLC10**
* **GLC-FCS30D 30m**

Only pixels where **all 5 datasets agree** on cropland are used for NDVI spatial reductions. This can be used for weakly supervised learning.

### 2. Time-Series Compositing (Sentinel-2)
* **Preprocessing**: Cloud, cirrus, and shadow masking using QA60 and SCL bands.
* **Clear-Sky Optimization**: Integration of Google's Cloud Score+ (`cs >= 0.5`) to isolate highest-quality observations.
* **Compositing**: Quality-of-mosaic matching based on maximum `cs` over sequential 15-day intervals covering the Kharif season.

### 3. Gap Imputation Baselines & Models
* **Linear Interpolation**: Standard temporal baseline.
* **Cubic Spline**: Higher-order polynomial temporal baseline.
* **Meta Prophet**: Additive regression model for trend modeling.
* **Mistral Timestral-209M**: SOTA foundation time-series forecaster.
* **Bidirectional LSTM**: Custom neural network leveraging forward and backward context states to minimize reconstruction error at masked gap locations.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Google Earth Engine (GEE) Account & Project
* GPU Runtime (Recommended for Bi-LSTM training & Timestral inference)

### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/your-username/sits-gapfill.git
cd sits-gapfill
pip install -r requirements.txt
pip install git+https://github.com/mistralai/timestral.git
Usage

Authenticate Google Earth Engine:
import ee
ee.Authenticate()
ee.Initialize(project='YOUR_EE_PROJECT_ID')
Run the main notebook or pipeline script to:
Define your study area boundary (e.g., Krishna District, FAO GAUL Level 2).
Build the Cropland Consensus Mask.
Extract 15-day Sentinel-2 NDVI composites.
Evaluate and impute synthetic temporal gaps.
📊 Quantitative Performance

Evaluation on simulating a cloud-gap at peak NDVI conditions demonstrates the performance of the models (measured by RMSE and MAE):

Method	RMSE	MAE	Characteristics
Linear Interpolation	0.1025	0.1025	Simple, but fails at peak curvature shifts.
Cubic Spline	0.1201	0.1201	Susceptible to polynomial overshoot over small datasets.
Bidirectional LSTM	0.0375	0.0375	Learns temporal patterns using bilateral context windows.
