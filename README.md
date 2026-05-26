# Brain Injury Spatial Analysis — Thesis Repository

**Multi-Scale Spatial Differential Analysis Framework for Detecting and Profiling Injury-Associated Cellular Neighborhoods Beyond Cortical Layers in Multiplex Brain Tissue Imaging**

> BS Honors Thesis · Biomedical Engineering · University of Houston
> Author: Razan Yousef (rkyousef)
> Advisor: Dr. Badri Roysam

---

## Overview

This repository contains the complete computational pipeline developed for my honors thesis. The central question of this work is:

> **Do injury-associated cellular abnormalities remain confined to the cortical layers after mild traumatic brain injury (mTBI), or do they extend into deeper brain structures?**

Using multiplex immunohistochemistry (IHC) brain tissue images from a midline fluid percussion injury (mFPI) rat model, we developed a multi-scale spatial proteomics analysis framework that:

1. Identifies injury-associated **emergent spatial cellular neighborhoods** that are weak or absent in healthy (sham) tissue
2. Determines whether these neighborhoods extend **beyond the cortical rim** into internal brain structures
3. Quantifies whether **Li+VPA treatment** selectively rescues the injury-associated phenotype

### Key Finding

Cluster C5, identified as the primary injury-associated neighborhood, showed:
- **3.43× enrichment** in vehicle vs. sham brains (p = 0.029)
- **3.38× enrichment** in the **cortex-band compartment** (p = 0.029)
- **3.58× enrichment** in the **internal brain compartment** (p = 0.029)
- **Partial rescue (36%)** by Li+VPA treatment

This confirms that injury-associated cellular microenvironments extend beyond the cortical surface into deeper brain regions.

---

## Experimental Groups

| Group | n | Description |
|-------|---|-------------|
| **Sham** | 4 brains | Healthy control — no injury |
| **Vehicle** | 4 brains | mFPI injured, 14 days post-injury |
| **Li+VPA** | 2 brains | mFPI injured + lithium/valproate treatment |

**Injury model:** Midline fluid percussion injury (mFPI) — diffuse injury, affects multiple brain regions

**Time point:** 14 days post-injury (chronic inflammatory phase)

---

## Marker Panel (16 binary channels)

| Category | Markers |
|----------|---------|
| Neuronal | NeuN, NFH, MAP2, GAD67, Parvalbumin, Calretinin |
| Glial | S100, GFAP |
| Microglial / Inflammatory | IBA1, IBA1++ (activated) |
| Vascular | RECA1, TomatoLectin, CD31 |
| Myelin | MBP |
| Cell Death | CC3 |
| Proliferation | PCNA |

---

## Repository Structure

```
brain-injury-spatial-analysis-thesis/
│
├── visualize_br10.py              # Step 1: Initial AVDGP cluster visualization
│                                  #   → Revealed the "accidental eureka" discovery
│
├── autonomous_delineator.py       # Step 2: GMM-based anatomical segmentation
│                                  #   → Objective 2: Autonomous anatomical mapping
│
├── spatial_sweep.py               # Step 3: Spatial weight parameter optimization
│                                  #   → Swept SW=[2.0, 5.0, 10.0]; selected SW=3.0
│
├── multiscale_framework.py        # Step 4: CORE THESIS ALGORITHM
│                                  #   → Multi-radius fingerprints (BallTree)
│                                  #   → Joint GMM clustering (n=15 components)
│                                  #   → Mann-Whitney U differential analysis
│                                  #   → Concordance-based validation
│                                  #   → Li+VPA rescue quantification
│
├── cortex_internal_analysis.py    # Step 5: Beyond-cortex compartment analysis
│                                  #   → Central thesis finding: C5 extends
│                                  #     into internal brain (3.58×, p=0.029)
│
├── .gitignore
└── README.md
```

---

## Pipeline Architecture

### Stage 1 — Initial Discovery (visualize_br10.py, autonomous_delineator.py)

The project began with AVDGP-based clustering for cortical layer delineation. An unexpected finding — the "accidental eureka" — occurred when the clustering isolated a spatially coherent region with altered cellular composition extending into deeper brain structures. This serendipitous discovery motivated the full multi-scale analysis framework.

### Stage 2 — Parameter Optimization (spatial_sweep.py)

A spatial weight sweep identified the optimal balance between **phenotype-driven** and **geography-driven** clustering. Spatial weight = 3.0 was selected as it produces region-like clusters with biological meaning (vs. pure cell-type or pure geographic zones).

### Stage 3 — Core Multi-Scale Framework (multiscale_framework.py)

The central computational contribution implements four novel components:

#### 3.1 Multi-Radius Spatial Phenotype Fingerprinting
For each cell, the fraction of marker-positive neighbors is computed at multiple radii [50, 100, 200 μm]:

```
fingerprint(cell_i, r) = fraction of cells within radius r that are marker_j positive
                         for each marker j in {NeuN, IBA1, GFAP, ...}
```

This is analogous to **SIFT/SURF descriptors** in computer vision or **multi-resolution wavelet coefficients** in signal processing — a scale-space descriptor for spatial proteomics.

#### 3.2 Joint GMM Clustering in Shared Feature Space
All sham and vehicle brains are jointly clustered in the same feature space (fingerprint + z-normalized coordinates × spatial_weight=3.0). This ensures clusters are **directly comparable** across brains and experimental groups.

#### 3.3 Brain-Level Statistical Testing
Differential analysis is performed at the **brain level** (n=4 vs n=4), not the cell level. This avoids pseudoreplication — cells are observations, but **brains are the biological replicates**.

- Test: Mann-Whitney U (two-sided)
- Minimum achievable p-value with n=4 vs n=4: **0.029**
- Concordance rule: finding must be significant at ≥ 2 of 3 scales (cell, neighborhood, region)

#### 3.4 Treatment Rescue Quantification

```
rescue_score = 1 - |mean_treated - mean_sham| / |mean_vehicle - mean_sham|

rescue = 1.0  →  full rescue to sham level
rescue = 0.0  →  no rescue (treated = vehicle)
rescue < 0    →  overcorrection
```

### Stage 4 — Compartment Analysis (cortex_internal_analysis.py)

The brain tissue is segmented into two anatomical compartments using morphological image processing on the cell density grid:
- **Cortex-band**: cells within the outer ~20% of tissue depth
- **Internal brain**: all remaining deeper structures

Cluster enrichment is compared across compartments and experimental groups, directly answering the central thesis question.

---

## Results Summary

| Cluster | Compartment | Veh/Sham Ratio | p-value | Rescue (Li+VPA) |
|---------|-------------|----------------|---------|-----------------|
| **C5** | Cortex-band | 3.38× | 0.029 | 36% |
| **C5** | Internal brain | 3.58× | 0.029 | 36% |
| C7 | Internal brain | 1.47× | 0.114 | 65% |
| C9 | Internal brain | 1.17× | 0.114 | 93% |

**Biological interpretation of C5:**
- IBA1+ microglial enrichment
- Activated microglia (IBA1++) co-elevation
- Extended into both cortical and subcortical regions
- Partial but selective rescue by Li+VPA (vascular and astrocytic signals rescued most strongly)

---

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy h5py scikit-image pickle5
```

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.3 | Data loading and manipulation |
| numpy | ≥1.21 | Numerical computation |
| scikit-learn | ≥0.24 | GaussianMixture, BallTree, StandardScaler |
| scipy | ≥1.7 | Mann-Whitney U test, Fisher exact test |
| matplotlib | ≥3.4 | Visualization |
| seaborn | ≥0.11 | Heatmaps |
| h5py | ≥3.0 | Loading .mat (MATLAB HDF5) files |
| scikit-image | ≥0.18 | Morphological operations for compartment detection |

---

## Data Structure

Each brain CSV contains one row per cell with the following columns:

- **centroid_x, centroid_y** — spatial coordinates in micrometers
- **NeuN, S100, IBA1, ...** — binary marker positivity (0/1) per cell
- **Area**, morphological features (if applicable)

Brain IDs follow the pattern: `br{number}` (e.g., `br10` = vehicle brain 10, `br24` = sham brain 24)

---

## Usage

Run scripts in order:

```bash
# 1. Initial visualization (requires AVDGP results .mat file)
python visualize_br10.py

# 2. Autonomous GMM-based anatomical segmentation
python autonomous_delineator.py

# 3. Sweep spatial weights to optimize clustering
python spatial_sweep.py

# 4. Run the core multi-scale framework (WARNING: slow on first run — fingerprinting takes ~1hr)
python multiscale_framework.py

# 5. Cortex vs internal brain compartment analysis (requires fingerprints.pkl from step 4)
python cortex_internal_analysis.py
```

---

## Relationship to Prior Lab Work

This thesis builds on and extends two prior frameworks from the Roysam Lab:

| Framework | Description | Relation to This Thesis |
|-----------|-------------|------------------------|
| **AIDPMM** (Aditi's work) | Active-learning-guided DPMM for cortical layer delineation | Provided the cortical structural context; this thesis extends beyond cortical layers |
| **mViSE** (Liqiang's work) | Multiplex Visual Search Engine — ViT encoders + community detection | Independent clustering that convergently identified similar deeper abnormal ROIs |

Both AIDPMM and mViSE identified deeper affected regions. This thesis provides **convergent validation** through an independent clustering pipeline, plus **biological profiling** and **treatment rescue analysis** of the identified deeper neighborhoods.

---

## Thesis Contributions

1. **Multi-scale spatial phenotype fingerprinting** using BallTree multi-radius neighborhood descriptors
2. **Joint cross-brain clustering** enabling direct compartment-by-compartment comparison
3. **Concordance-based multi-scale validation** (cell, neighborhood, region scales)
4. **Treatment rescue quantification** framework (Li+VPA vs. vehicle vs. sham)
5. **Central biological finding**: Injury-associated cellular neighborhoods (C5) extend beyond the cortical rim into internal brain structures, and are partially but selectively rescued by Li+VPA

---

## Citation

If you use this code, please cite:

```
Yousef, R. (2026). Multi-Scale Spatial Differential Analysis Framework for Detecting 
and Profiling Injury-Associated Cellular Neighborhoods Beyond Cortical Layers in 
Multiplex Brain Tissue Imaging. BS Honors Thesis, Department of Biomedical Engineering,
University of Houston.
```

---

## Acknowledgments

- **Dr. Badri Roysam** — Thesis advisor
- **Aditi** — Prior AIDPMM cortical delineation framework
- **Liqiang** — mViSE encoder-based region discovery framework
- University of Houston Department of Biomedical Engineering
