from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


fichier = "artificial/banana.arff"
if len(sys.argv) >= 3 and os.path.isfile(sys.argv[2]) :
    fichier = sys.argv[2]

data, _ = arff.loadarff(fichier)
df = pd.DataFrame(data).iloc[:, :2]

X = df.values

nb_clusters = 8


def norm(results, invert=False) :
    '''
    Normalise une liste de résultats entre 0 et 1 : la plus petite valeur devient 0, la plus grande 1.
    Si invert est True, la plus grande valeur devient 0, la plus petite 1 (dans le cas d'un indicateur à minimiser comme DB).
    '''
    maxi = max(results)
    mini = min(results)

    if maxi == mini :
        return [0.0 for _ in results]

    if invert :
        return [1 - (res-mini)/(maxi-mini) for res in results]
    
    return [(res-mini)/(maxi-mini) for res in results]


def plot_kMeans(best_clustering, clustering_results, silhouette_results, CAH_results, DB_results) :
    print(f"\nPlotting K-Means results")

    _, axes = plt.subplots(2, 2, figsize=(8, 12))  # 2 lignes, 2 colonnes
    
    # --- 1. Silhouette ---
    axes[0, 0].plot(range(2, nb_clusters+2), silhouette_results, marker='o', color='blue', label='Silhouette')
    axes[0, 0].set_title('Indice Silhouette (MAX)')
    axes[0, 0].set_xlabel('Nombre de clusters')
    axes[0, 0].set_ylabel('Silhouette')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # --- 2. CAH ---
    axes[0, 1].plot(range(2, nb_clusters+2), CAH_results, marker='o', color='green', label='CAH')
    axes[0, 1].set_title('Indice CAH (MAX)')
    axes[0, 1].set_xlabel('Nombre de clusters')
    axes[0, 1].set_ylabel('CAH')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # --- 3. Davies-Bouldin ---
    axes[1, 0].plot(range(2, nb_clusters+2), DB_results, marker='o', color='red', label='Davies-Bouldin')
    axes[1, 0].set_title('Indice Davies-Bouldin (MIN)')
    axes[1, 0].set_xlabel('Nombre de clusters')
    axes[1, 0].set_ylabel('DB')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # --- 4. Best Clustering ---
    clustering = clustering_results[best_clustering]

    labels = clustering.labels_
    centers = clustering.cluster_centers_

    axes[1, 1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
    if centers.size > 0:
        axes[1, 1].scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centres')

    axes[1, 1].set_title(f"{type(clustering).__name__} avec {best_clustering+2} clusters")
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, wspace=0.15, hspace=0.25)
    plt.show()


def plot_agglomerative(clustering_results_by_linkage) :
    print(f"\nPlotting Agglomerative Clustering results")

    _, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (linkage, (clustering, best_k)) in enumerate(clustering_results_by_linkage.items()):
        ax = axes[i]
        labels = clustering.labels_
        
        # Calcul des centres à partir des labels
        unique_labels = np.unique(labels)
        centers = np.array([X[labels == lab].mean(axis=0) for lab in unique_labels if lab != -1])

        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
        if centers.size > 0:
            ax.scatter(centers[:, 0], centers[:, 1], c='red', s=150, marker='X', label='Centres')
        
        ax.set_title(f"Linkage: '{linkage}' (k={best_k})")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True)


    plt.tight_layout()
    plt.subplots_adjust(left=0.05, wspace=0.15, hspace=0.25)
    plt.show()


def run_kMeans():
    clustering_results, silhouette_results, CAH_results, DB_results = [], [], [], []
    print("Exécution de K-Means")
    print('_'*(100//nb_clusters*nb_clusters))

    for i in range(nb_clusters) :
        clustering = KMeans(n_clusters=i+2, n_init=10).fit(X)
        clustering_results.append(clustering)
        labels = clustering.labels_

        silhouette_results.append(silhouette_score(X, labels))
        CAH_results.append(calinski_harabasz_score(X, labels))
        DB_results.append(davies_bouldin_score(X, labels))
        print(100//nb_clusters*'#', end='', flush=True)


    print()
    silhouette_normed = norm(silhouette_results)
    CAH_normed = norm(CAH_results)
    DB_normed = norm(DB_results, True)

    average_scores = [(silhouette_normed[i] + CAH_normed[i] + DB_normed[i]) / 3 for i in range(nb_clusters)]
    best_clustering_index = average_scores.index(max(average_scores))

    plot_kMeans(best_clustering_index, clustering_results, silhouette_results, CAH_results, DB_results)


def run_agglomerative():
    results_by_linkage = {}
    print("Exécution du Clustering Hiérarchique")

    for linkage in ['ward', 'complete', 'average', 'single']:
        print(f"\nTesting {linkage} linkage")
        print('_'*(100//nb_clusters*nb_clusters))
        clustering_results, silhouette_results, CAH_results, DB_results = [], [], [], []

        for i in range(nb_clusters) :
            clustering = AgglomerativeClustering(n_clusters=i+2, linkage=linkage).fit(X)

            clustering_results.append(clustering)
            labels = clustering.labels_
            
            silhouette_results.append(silhouette_score(X, labels))
            CAH_results.append(calinski_harabasz_score(X, labels))
            DB_results.append(davies_bouldin_score(X, labels))
            print(100//nb_clusters*'#', end='', flush=True)


        print()
        silhouette_normed = norm(silhouette_results)
        cah_normed = norm(CAH_results)
        db_normed = norm(DB_results, True)

        average_scores = [(silhouette_normed[j] + cah_normed[j] + db_normed[j])/3 for j in range(nb_clusters)]
        best_clustering_index = average_scores.index(max(average_scores))
        
        best_k_for_linkage = best_clustering_index + 2
        best_clustering_for_linkage = clustering_results[best_clustering_index]
        
        results_by_linkage[linkage] = (best_clustering_for_linkage, best_k_for_linkage)

    plot_agglomerative(results_by_linkage)


def clusterise(method) :
    match method :
        # --- K-Means ---
        case "KMeans" :
            run_kMeans()

        # --- Clustering hiérarchique ---
        case "Agglomerative" :
            run_agglomerative()

        # --- DBSCAN ---
        case "DBScan" :
            clustering = DBSCAN(eps=0.5, min_samples=5).fit(X)

        # --- HDBSCAN ---
        case "HDBScan" :
            # clustering = hdbscan.HDBSCAN(min_cluster_size=5).fit(X)
            print("HDBScan non implémenté.")

        case _ :
            raise ValueError(f"Méthode de clustering inconnue : {method}")


if __name__ == "__main__" :
    method = "KMeans"
    if len(sys.argv) >= 2 and sys.argv[1] in ["KMeans", "Agglomerative", "DBScan", "HDBScan"] :
        method = sys.argv[1]

    clusterise(method)