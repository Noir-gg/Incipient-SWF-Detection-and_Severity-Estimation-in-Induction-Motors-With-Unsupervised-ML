# This is the code for the paper 'Incipient Stator Winding Fault Detection and Severity Estimation in Induction Motors With Unsupervised Machine Learning Algorithms'

## Project overview
This repository contains the unsupervised‑learning pipeline described in the paper for detecting and grading incipient stator‑winding faults. The workflow converts three‑phase current signals into statistical, spectral, and time‑frequency features, applies three clustering algorithms, and estimates severity levels from the resulting clusters.

## Code structure
* `Transform_pipeline.ipynb` – end‑to‑end notebook: preprocessing, feature extraction, clustering, metrics, and visualisation.
* `results/` – automatically created by the notebook to store metrics and figure outputs.

Place the raw CSV files in a folder named `dataset/` before running the notebook.

## Dataflow
1. Acquire 12 kHz stator‑current recordings under healthy and faulty conditions.  
2. Preprocess with outlier removal and min–max scaling.  
3. Extract 20 FFT and 20 DWT coefficients plus statistical moments.  
4. Reduce dimensionality with PCA (for plotting only).  
5. Cluster with K‑means, Gaussian Mixture Model (GMM), and Ward‑linkage hierarchical clustering.  
6. Compute Davies–Bouldin, Calinski–Harabasz, Adjusted Rand Index, and Normalised Mutual Information metrics.  
7. Map clusters to six severity levels.

## Results
| Metric | K‑means | GMM |
| --- | --- | --- |
| Davies–Bouldin | 0.85 | 1.63 |
| Calinski–Harabasz | 499 909 | 342 100 |
| Adjusted Rand Index | 0.00014 | 0.0050 |
| Normalised Mutual Information | 0.00023 | 0.0055 |

Hierarchical clustering identifies six clusters when the dendrogram cut threshold is between 1600 and 2000.


### Clustering visualisations
![K‑means PCA](/kmeans_pca_visualised.png)
![GMM PCA](/gmm_pca_visualised.png)
![K‑means 3‑D](/kmeans_3dcluster.png)

### Hierarchical linkage
![Full dendrogram](/dendogram.png)
![Truncated dendrogram](/hierarchal_dendogram.png)

### Precision‑Recall and ROC curves
![Extra Trees ROC](/ExtraTrees_ROC_curve.jpeg)
![XGBoost ROC](/XGBoost_ROC_curve.jpeg)


## Inputs and outputs
* **Input** – `dataset/dataset.csv`.  
* **Outputs** – clustering metrics CSV, PCA scatter plots, dendrograms, ROC / PR curves (if supervised baselines are executed).

## Quick start
```bash
pip install -r requirements.txt
jupyter lab
# open Transform_pipeline.ipynb and run all cells
```

## Citation
Rehaan Hussain, Shady S. Refaat, Incipient Stator Winding Fault Detection and Severity Estimation in Induction Motors With Unsupervised Machine Learning Algorithms, IEEE Transactions on Industrial Electronics, 2024, doi:10.1109/TIE.2024.xxxxxx

