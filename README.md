# Hierarchical Cluster Analysis (HCA) Python Class

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)

## 📌 Overview

This Python class provides a comprehensive and customizable tool for **Hierarchical Cluster Analysis (HCA)** of multivariate datasets.
It offers tools for data preprocessing (z-score normalization, missing value imputation), flat cluster extraction, dendrogram generation, and linkage distance analysis.

Designed for experimental, analytical, or statistical workflows, this module helps uncover meaningful groupings and relationships within complex datasets.

---

## ✨ Features

- 🔄 Automatic **data standardization** via z-score normalization
- 🧱 Flexible missing data imputation strategies (`mean`, `median`, or `constant`)
- 🔗 Supports a wide range of **linkage methods** and **distance metrics**
- 🌿 Easily generate and customize **dendrograms** with cluster cut lines
- 📄 Tabular **summary** of hierarchical clustering steps
- 🔢 Retrieve **cluster assignments** for use in external workflows (e.g., PCA)
- 📉 Plot linkage **distance progression** to aid in cluster number estimation
- ✅ Compatible with `pandas`, `numpy`, `matplotlib`, `scipy`, and `scikit-learn`

---

## 🔧 Installation

To use this class, simply copy the `HCA` class to your project or package.  
Required dependencies:

```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

## 🚀 Usage
```bash
import pandas as pd
from hca import HCA   # adjust import path if necessary
```

### 1. Load your data: first column = labels, rest = numeric variables
```bash
df = pd.read_csv("your_data.csv")
```

### 2. Initialize HCA (impute missing values with 0, apply z-score scaling)
``` bash
hca = HCA(df, z_score=True, impute='constant', fill_value=0)
``` 

### 3. Generate a summary table of clustering steps
   - method: linkage method ('ward', 'average', etc.)
   - metric: distance metric ('euclidean', 'cityblock', etc.)
   - threshold: cut-off value for forming flat clusters
   - criterion: 'distance', 'maxclust', 'inconsistent', etc.

``` bash
summary_df = hca.summary(
    method='ward',
    metric='euclidean',
    threshold=5.0,
    criterion='distance'
)
print(summary_df)
``` 

### 4. Retrieve cluster assignments for each sample
``` bash
clusters = hca.get_clusters(
    method='ward',
    metric='euclidean',
    threshold=5.0,
    criterion='distance'
)
print("Cluster assignments:", clusters)
``` 

### 5. Plot and save dendrogram and linkage distance plot
  - orientation: 'top', 'bottom', 'left', 'right'

``` bash
hca.plot(
    method='average',
    metric='euclidean',
    threshold=3,
    criterion='maxclust',
    orientation='top'
)
```
### 📎 Notes

- The `threshold` and `criterion` parameters define how clusters are formed from the dendrogram.
- To specify a fixed number of clusters, use `criterion='maxclust'` and set threshold to the desired number.
- All results are computed using `scipy.cluster.hierarchy` under the hood.

### 📂 Output Files

- `Dendrogram - Hierarchical Cluster Analysis.png`
- `Linkage Distances - Hierarchical Cluster Analysis.png`

These images are saved automatically in the current working directory.
