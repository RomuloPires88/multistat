# Hierarchical Cluster Analysis (HCA) Python Class

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)

## 📌 Overview

This Python class provides a comprehensive and customizable tool for **Hierarchical Cluster Analysis (HCA)** of multivariate data.  
It includes utilities for data preprocessing (scaling and imputation), clustering, dendrogram visualization, and linkage distance plotting.

Designed for experimental and statistical workflows, this module helps interpret relationships and groupings within complex datasets.

---

## ✨ Features

- 📊 Automatic **data standardization** and **missing value imputation**
- 🔗 Supports a wide range of **linkage methods** and **distance metrics**
- 🌿 Easily generate and customize **dendrograms**
- 📈 View clustering step details and **linkage distance evolution**
- 🧪 Built-in **summary** and **plot** methods for quick insights
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
