from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt


data, _ = arff.loadarff('artificial/donutcurves.arff')
df = pd.DataFrame(data)
df = df.iloc[:, :2]

X = df.values

clustering_results = []
silhouette_results = []
CAH_results = []
DB_results = []

nb_cluster = range(2, 16)

def norm(results, invert=False) :
    maxi = max(results)
    mini = min(results)

    if invert :
        return [1 - (res-mini)/(maxi-mini) for res in results]
    
    return [(res-mini)/(maxi-mini) for res in results]

def plot(best_clustering) :
    fig, axes = plt.subplots(2, 2, figsize=(8, 12))  # 3 lignes, 1 colonne

    # --- 1. Silhouette ---
    axes[0, 0].plot(nb_cluster, silhouette_results, marker='o', color='blue', label='Silhouette')
    axes[0, 0].set_title('Indice Silhouette (MAX)')
    axes[0, 0].set_xlabel('Nombre de clusters')
    axes[0, 0].set_ylabel('Silhouette')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # --- 2. CAH ---
    axes[0, 1].plot(nb_cluster, CAH_results, marker='o', color='green', label='CAH')
    axes[0, 1].set_title('Indice CAH (MAX)')
    axes[0, 1].set_xlabel('Nombre de clusters')
    axes[0, 1].set_ylabel('CAH')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # --- 3. Davies-Bouldin ---
    axes[1, 0].plot(nb_cluster, DB_results, marker='o', color='red', label='Davies-Bouldin')
    axes[1, 0].set_title('Indice Davies-Bouldin (MIN)')
    axes[1, 0].set_xlabel('Nombre de clusters')
    axes[1, 0].set_ylabel('DB')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # --- 4. Best Clustering ---
    labels = clustering_results[best_clustering].labels_
    centers = clustering_results[best_clustering].cluster_centers_

    axes[1, 1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
    axes[1, 1].scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centres')
    axes[1, 1].set_title(f"K-Means avec {best_clustering + 2} clusters")
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

def clusterise(method) :
    for i in nb_cluster :

        match method :
            # --- K-Means ---
            case "KMeans" :
                clustering = KMeans(n_clusters=i, init='k-means++', random_state=0).fit(X)

            # --- Clustering hiérarchique ---
            case "Agglomerative" :
                clustering = AgglomerativeClustering(n_clusters=i, linkage='ward').fit(X)

            # --- DBSCAN ---
            case "DBScan" :
                clustering = DBSCAN(eps=0.5, min_samples=5).fit(X)

            # --- HDBSCAN ---
            case "HDBScan" :
                # clustering = hdbscan.HDBSCAN(min_cluster_size=5).fit(X)
                pass

            case _ :
                raise ValueError(f"Méthode de clustering inconnue : {method}")
        

        clustering_results.append(clustering)
        silhouette_results.append(silhouette_score(X, clustering.labels_))
        CAH_results.append(calinski_harabasz_score(X, clustering.labels_))
        DB_results.append(davies_bouldin_score(X, clustering.labels_))

    # Norm the results between 0 and 1
    silhouette_results_normed = norm(silhouette_results)
    CAH_results_normed = norm(CAH_results)
    DB_results_normed = norm(DB_results, True)


    # Choose the best nb_cluster
    average_results = [(silhouette_results_normed[i] + CAH_results_normed[i] + DB_results_normed[i])/3 for i in range(len(nb_cluster))]
    best_clustering = average_results.index(max(average_results))

    plot(best_clustering)


clusterise("KMeans")