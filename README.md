# 🚀 Conflict-Aware Solar Site Prioritization in India

## **A Hybrid Deep Learning and Fuzzy MCDM Framework for SCOPUS Publication**

| Status | Goal | Target | License |
| :--- | :--- | :--- | :--- |
| **In Development** | **SCOPUS Publication** | **India Solar Site Optimization** | MIT |

-----

## 💡 Project Overview & Novelty

This project moves beyond simple solar farm detection to provide a **Conflict-Aware Solar Site Suitability Index (SSSI)** for India. It addresses the critical issue of solar energy development conflicting with high-value land (agriculture and forest).

The core novelty is a **two-stage, tightly-coupled hybrid framework**:

1.  **Stage 1: Detection:** A state-of-the-art **Hybrid CNN-Transformer Segmentation Model** is implemented to map solar farms with high fidelity.
2.  **Stage 2: Prioritization:** The output from the detection model is used as a **validation/calibration layer** for an advanced **Fuzzy Multi-Criteria Decision-Making (F-MCDM)** system to prioritize future sites based on **economic viability, resource potential, and low environmental/social conflict.**

### Target Users and Problems Solved

| Target User | Pain Point Solved |
| :--- | :--- |
| **Solar Energy Developers** | Reduces **Quantifiable Project Risk** by prioritizing sites closer to the HV power grid and on flatter land (Technical/Economic factors). |
| **Policy Makers (MNRE)** | Identifies **High-Potential, Low-Conflict land**, enabling targeted development and mitigating social disputes and land-use conflicts. |
| **Financial Institutions** | Provides a **financially weighted Suitability Index** for due diligence, offering greater confidence than resource-only assessments. |

-----

## ⚙️ Core Technical Workflow

The project follows a rigorous, four-step methodical approach derived from the Design Document:

### 1\. 🗺️ Data Acquisition & Geospatial Preprocessing

  * **Goal:** Collect, standardize, and feature-engineer all geospatial input layers at a uniform resolution (e.g., 10m/30m).
  * **Key Inputs:** Sentinel-2 L2A data, NASA POWER GHI/DNI, SRTM DEM, OpenStreetMap (for Grid/Roads), and LULC maps.
  * **Feature Engineering:** Raster layers are derived for key constraints: **Slope (%)** (from DEM), **Euclidean Distance to HV Grid** (from OSM), and a categorical **LULC Conflict Index**.

### 2\. 🧠 Hybrid Transformer Segmentation (ML/DL Refinement)

  * **Goal:** Achieve best-in-class performance for solar farm detection, significantly improving recall over the baseline U-Net model.
  * **Architecture:** **Swin-Unet / MaxViT-Unet** with a deep backbone (e.g., ResNet-50) using the `PyTorch` and `segmentation_models_pytorch` framework.
  * **Training:** Uses a Compound Loss (BCE + Dice/Jaccard) and a rigorous, multi-stage **Hard Negative Mining (HNM)** strategy to eliminate false positives (e.g., water, construction sites).
  * **Output:** A GeoJSON layer of all detected solar farm polygons across India for validation.

### 3\. ⚖️ Fuzzy AHP-TOPSIS MCDM Framework

  * **Goal:** Systematically derive objective weights for all decision criteria and rank site alternatives.
  * **Weighting:** **Fuzzy AHP** is used to transform linguistic expert judgment into objective, numerical criteria weights ($W_i$), accounting for ambiguity.
  * **Ranking:** **TOPSIS** is applied to rank grid cells by their proximity to the **Fuzzy Positive Ideal Solution (FPIS)**.
  * **Validation Loop:** The detection output from **Step 2** is used to **statistically validate** the F-AHP weights by correlating the real-world density of farms with the predicted suitability scores. This is the **patentable core** of the methodology.

### 4\. 📈 Suitability Mapping and Conflict Analysis

  * **Goal:** Generate the final **Solar Site Suitability Index (SSSI)** map and extract actionable policy insights.
  * **Process:** All normalized, weighted criteria are combined via the TOPSIS formula to create the final continuous SSSI raster. Hard constraints (water, protected areas, high slope) are masked.
  * **Analysis:** The final map is analyzed to identify regions of **High-Potential, Low-Conflict Land** (ideal for future development) and quantify the environmental impact of development built on high-conflict land.

-----

## 🗂️ Repository Structure

The file structure is organized logically according to the four major workflow steps, ensuring clarity and traceability for future developers and reviewers.

```
.
├── 📂 data/
│   ├── raw/                  # Source data (DEM, LULC shapefiles, Grid/Road GeoJSONs, NASA POWER CSVs)
│   ├── processed/            # Standardized, preprocessed raster layers (Slope.tif, Dist_Grid.tif, LULC_Conflict_Index.tif)
│   └── external/             # Pre-trained model weights (checkpoint.pth.tar)
├── 📂 src/
│   ├── 🛠️ cv_models/
│   │   ├── __init__.py
│   │   ├── swin_unet_model.py # Hybrid CNN-Transformer model definition (Step 2.0)
│   │   └── inference_framework.py # Class for loading model and running batch inference (Step 2.0)
│   ├── ⚖️ mcdm_tools/
│   │   ├── fuzzy_ahp.py         # F-AHP implementation for weight derivation (Step 3.0)
│   │   └── topsis.py            # TOPSIS ranking algorithm (Step 3.0)
│   └── 🗺️ geo_utils.py          # Functions for slope, distance, and LULC reclassification (Step 1.0)
├── 📓 notebooks/
│   ├── 1_Data_Preprocessing.ipynb  # Executes Step 1.0 tasks (Geo Data Prep)
│   ├── 2_Model_Training_and_HNM.ipynb # Executes Step 2.0 tasks (ML Training)
│   ├── 3_MCDM_Weight_Derivation.ipynb # Executes Step 3.0 tasks (FAHP-TOPSIS)
│   ├── 4_Final_Mapping_Analysis.ipynb # Executes Step 4.0 tasks (SSSI Map Generation & Conflict Analysis)
│   └── legacy/                   # Previous PBL-1 notebooks (for reference)
├── README.md
├── requirements.txt              # List of all required Python libraries (PyTorch, geopandas, rasterio, etc.)
└── LICENSE
```

### Key Implementation Locations

| Feature | Implementation File(s) | Workflow Step |
| :--- | :--- | :--- |
| **Grid Proximity & Slope** | `src/geo_utils.py` | 1.0 Data Preprocessing |
| **Hybrid Swin-Unet** | `src/cv_models/swin_unet_model.py` | 2.0 Model Development |
| **Fuzzy AHP Logic** | `src/mcdm_tools/fuzzy_ahp.py` | 3.0 MCDM Framework |
| **Final SSSI Map** | `4_Final_Mapping_Analysis.ipynb` | 4.0 Suitability Mapping |

-----

## 🔒 Patentable Approach: CCDLRSO

The unique contribution of this project is the **Conflict-Calibrated Deep Learning for Renewable Site Optimization (CCDLRSO) System**.

This system introduces a **self-correcting mechanism** into the spatial decision model:

> **The system iteratively uses the highly accurate real-world solar farm density detected by the Hybrid Transformer (Step 2.0) to dynamically validate and refine the weighting set of the Fuzzy AHP-TOPSIS model (Step 3.0).**

This ensures that the final SSSI map is not just theoretically sound but is **calibrated to the practical, historical economic and technical drivers** that influence where solar farms are successfully built in India.
