import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score,calinski_harabasz_score, davies_bouldin_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import fcluster

df_clusters = pd.DataFrame([])
results_df = pd.DataFrame([])
metric_csv = pd.DataFrame([])

def apply_kmeans(dataset):
    # Apply KMeans 

    # 1. K-Means Clustering (6 clusters)
    print("Applying K means.")
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    results_df['KMeans_Cluster'] = kmeans.fit_predict(dataset)
    metrics(dataset, results_df['KMeans_Cluster'], model = "Kmeans")
    plot_PCA_TSNE(dataset, results_df['KMeans_Cluster'], model = "Kmeans")
    print("K means Applied.")


def apply_GMM(dataset):
    # Apply GMM 

    print("Applying GMM.")
    gmm = GaussianMixture(n_components=6, random_state=42)
    results_df['GMM_Cluster'] = gmm.fit_predict(dataset)
    metrics(dataset, results_df['GMM_Cluster'], model = "GMM")
    plot_PCA_TSNE(dataset, results_df['GMM_Cluster'], model = "GMM")
    print("GMM Applied.")

def apply_hierarchal_clustering(dataset):
    # Hierarchical Clustering 
    df_balanced = dataset.groupby('label').apply(lambda x: x.sample(n=5000, random_state=42))

    # Step 3: Drop the duplicate index column created by `groupby().apply()`
    df_balanced = df_balanced.reset_index(drop=True)

    # Step 4: Extract feature matrix (without labels)
    X_balanced = df_balanced.drop(columns=['label'])

    print("Applying Hierarchical Clustering.")
    linkage_matrix = linkage(X_balanced, method="ward")
    print("Hierarchical Clustering Applied.")
    

    # # Dendrogram for Hierarchical Clustering
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, truncate_mode="level", p=5)
    num_clusters = 6
    clusters = fcluster(linkage_matrix, t=num_clusters, criterion="maxclust")

    # # Add the cluster labels to the DataFrame
    df_clusters['Hierarchical_Cluster'] = clusters

    # # Determine the threshold used to form these clusters
    threshold = np.max(linkage_matrix[linkage_matrix[:, 3] == num_clusters, 2])
    print(f"Threshold distance for 6 clusters: {threshold}")

    plt.title("Hierarchical Clustering Dendrogram (Truncated)")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.savefig("../results/hierarchal_dendogram.png")
    plt.close()


    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix, truncate_mode="level", p=5)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.axhline(y=1800, color='r', linestyle='--', label=f'Threshold: {threshold:.2f}')
    plt.xlabel("Sample index or Cluster ID")
    plt.ylabel("Distance")
    plt.legend()
    plt.savefig("../results/dendogram.png")


def metrics(X, results_dataframe, model):

    # print("Running Silhouette Score")
    # silhouette = silhouette_score(X, results_dataframe)
    # print(f"{model} silhouette Score : {silhouette_score}")
    
    print("Running Davis Bouldin Score")
    dbi = davies_bouldin_score(X, results_dataframe)
    print(f"{model} Davis Bouldin Score : {dbi}")

    print("Running Calinski Harabasz Score")
    chi = calinski_harabasz_score(X, results_dataframe)
    print(f"{model} Calinski Harabasz Score : {chi}")

    print("Running Adjusted Rand Index Score")
    ari = adjusted_rand_score(df['label'], results_dataframe)
    print(f"{model} Adjusted Rand Index Score : {ari}")

    print("Running Normalized Mutual Info Score")
    nmi = normalized_mutual_info_score(df['label'], results_dataframe)
    print(f"{model} Normalized Mutual Info Score : {nmi}")

    metric = {
        "model": model,
        "Davis_Bouldin_Score": dbi,
        "Calinski_Harabasz_Score": chi,
        "Adjusted_Rand_Index_Score": ari,
        "Normalized_Mutual_Info_Score": nmi
    }

    columns = ["model", "Davis_Bouldin_Score", "Calinski_Harabasz_Score", "Adjusted_Rand_Index_Score", "Normalized_Mutual_Info_Score"]
    metric_csv = pd.DataFrame(columns=columns)
    metric_csv = pd.concat([metric_csv, pd.DataFrame([metric])], ignore_index=True)


def preprocess(df):

    # Normalize Ia, Ib, Ic using StandardScaler

    # print("Standardising the dataset...")
    # scaler = StandardScaler()
    # X = scaler.fit_transform(df.drop(columns=["label"]))
    # df[['I_A', 'I_B', 'I_C']] = scaler.fit_transform(df[['Ia', 'Ib', 'Ic']])
    # print("Dataset Standardised.")

    X, y = df.drop(columns=["label"]), df["label"]

    # Extract feature matrix (without labels)
    # X = df

    return X, y

def plot_PCA_TSNE(X, results_dataframe, model):

    print("Applying PCA.")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    results_df["PCA1"] = X_pca[:, 0]
    results_df["PCA2"] = X_pca[:, 1]
    print("Plotting")
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=results_df["PCA1"], y=results_df["PCA2"] , hue=results_dataframe, palette="viridis", alpha=0.6)
    plt.title(f"{model} Clustering Visualization (PCA Reduced)")
    plt.legend(loc="lower right", title="Clusters")
    plt.savefig(f"../results/{model}_pca_visualised.png")
    print("Plotted")

    # tsne = TSNE(n_components=2, random_state=42)
    # X_tsne = tsne.fit_transform(X)
    # print("Applying TSNE")
    # # Add t-SNE components to DataFrame
    # df['tSNE1'] = X_tsne[:, 0]
    # df['tSNE2'] = X_tsne[:, 1]
    # print("Plotting")

    # # Plot K-Means clusters
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(x=results_df['tSNE1'], y=results_df['tSNE2'], hue=results_dataframe, palette="viridis", alpha=0.6)
    # plt.title(f"{model} Clustering Visualization (t-SNE Reduced)")
    # plt.xlabel("tSNE1")
    # plt.ylabel("tSNE2")
    # plt.legend(title="Cluster")
    # plt.savefig(f"../results/{model}_tsne_visualised.png")
    # print("Plotted")


# === STEP 1: Load the dataset ===
print("Loading the dataset...")
file_path = "../dataset/final_cleaned_transformed_dataset.csv"
# df = pd.read_csv(file_path).drop(columns=["time"])  
df = pd.read_csv(file_path)
print("Dataset loaded.")
print(df.head())


X, y = preprocess(df)
# apply_kmeans(X)
# apply_GMM(X)
apply_hierarchal_clustering(df)




# === STEP 6: Visualize Dendogram for Hierarchal Cluster ===


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for 3D clusters
ax.scatter(df['I_A'], df['I_B'], df['I_C'], c=results_df['KMeans_Cluster'], cmap='viridis', alpha=0.6)
ax.set_title("K-Means Clustering (3D)")
ax.set_xlabel("Ia")
ax.set_ylabel("Ib")
ax.set_zlabel("Ic")
plt.savefig("../results/kmeans_3dcluster.png")


# === STEP 7: Visualize Dendogram for Hierarchal Cluster ===


# # === STEP 5: Display Cluster Assignments ===
# df_clusters.to_csv("../results/Hierarchical_Clustering_clusters.csv", index=False)
results_df.to_csv("../results/results_df.csv", index=False)
metric_csv.to_csv("../results/metrics.csv", index=False)
# df.to_csv("../results/processed_stator_winding_severity.csv", index=False)

