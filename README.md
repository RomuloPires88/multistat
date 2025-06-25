# MultiStat

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)

## 📌 Overview

**MultiStat** is a Python module that provides a comprehensive and customizable tool for **Hierarchical Cluster Analysis (HCA)** and **Principal Component Analysis (PCA)** of multivariate datasets.

It includes functionality for data preprocessing (z-score normalization, missing value imputation), dendrogram generation, cluster extraction, linkage distance analysis, scree plots, and biplots.

Designed for experimental, analytical, or statistical workflows, this module helps uncover meaningful groupings and relationships within complex datasets.

---

## ✨ HCA Features

- 🔄 Automatic **data standardization** via z-score normalization  
- 🧱 Flexible missing data imputation strategies (`mean`, `median`, or `constant`)  
- 🔗 Support for a wide range of **linkage methods** and **distance metrics**  
- 🌿 Generate and customize **dendrograms** with cluster cut lines  
- 📄 Tabular **summary** of hierarchical clustering steps  
- 🔢 Retrieve **cluster assignments** for use in other workflows (e.g., PCA)  
- 📉 Plot **linkage distance progression** to help estimate optimal number of clusters  
- ✅ Fully compatible with `pandas`, `numpy`, `matplotlib`, `scipy`, and `scikit-learn`  

---

## ✨ PCA Features

- 📊 Compute **principal components** and project multivariate data in 2D space  
- ⚖️ Optional **z-score standardization** and missing value imputation  
- 📈 Generate **scree plots** with explained variance and loadings  
- 🧭 Create **biplots** combining scores and variable vectors  
- 🎯 Visualize **clusters** overlaid from HCA results  
- ⚙️ Adjustable **vector scaling** and selection of component axes  
- 📋 Summary of PCA **loadings** and explained variance by component  

---

## 🔧 Installation

To use this software, simply copy the folder to your project or package:
To use this software, simply copy the folder to your project or package.  

```text
/multistat
├── __init__.py
├── hca.py
└── pca.py
```

Required dependencies:

```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

## 🚀 Usage
```bash
import pandas as pd
import multistat as mv
```

### 1. Load your data: first column = labels, rest = numeric variables
```bash
df = pd.read_csv("your_data.csv")
```

### 2. Initialize HCA or PCA (impute missing values with 0, apply z-score scaling)
``` bash
hca = mv.HCA(df, z_score=True, impute='constant', fill_value=0)
pca = mv.PCA(df, z_score=True, impute='constant', fill_value=0)
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
![imagem](https://github.com/user-attachments/assets/6c94510e-f794-464e-9e35-f68f40e9d9ee)


### 4. Retrieve cluster assignments for each sample
``` bash
clusters = hca.get_clusters(
    method='ward',
    metric='euclidean',
    threshold=4.0,
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
    threshold=4,
    criterion='maxclust',
    orientation='top'
)
```
![imagem](https://github.com/user-attachments/assets/ea0388d3-8566-4358-8533-3136a830f7bd)

![imagem](https://github.com/user-attachments/assets/aa27a136-fe88-485b-aa70-fdd57cce2d64)

### 6. Plot and save Scree plot, with Explained Variance Summary and PCA Loadings
 
``` bash
pca.scree()
```
![imagem](https://github.com/user-attachments/assets/b6936989-33ad-49b3-b7c4-51b380fab88c)

![imagem](https://github.com/user-attachments/assets/a5b02b61-70ab-42a2-a5e5-b1585e1069e4)

![imagem](https://github.com/user-attachments/assets/30307c26-b04e-4f60-9bbc-71a76a6e1792)


### 7. Plot and save PCA Analysis with clusters built in HCA
 
``` bash
pca.plot(clusters=clusters)
```
![imagem](https://github.com/user-attachments/assets/6a232aa1-85c3-4d9a-ac39-f7ca40e5c4e4)

### 8. Plot and save Biplot with clusters built in HCA and changing the vector scale
 
``` bash
pca.biplot(clusters=clusters, scale_vectors=2)
```
![imagem](https://github.com/user-attachments/assets/df3d9892-5ee1-4475-b18f-a25b07bbbe0e)


### 📎 Notes

- The `threshold` and `criterion` parameters define how clusters are formed from the dendrogram.
- To specify a fixed number of clusters, use `criterion='maxclust'` and set threshold to the desired number.
- All results are computed using `scipy.cluster.hierarchy` under the hood.

### 📂 Output Files

- `Dendrogram - Hierarchical Cluster Analysis.png`
- `Linkage Distances - Hierarchical Cluster Analysis.png`
- `Scree plot - Scree Plot.png`
- `PCA Analysis - PCA Analysis.png`
- `PCA Biplot - PCA Biplot.png`

These images are saved automatically in the current working directory.
