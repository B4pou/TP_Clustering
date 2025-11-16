from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt
import sys, os


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


def run_kMeans():
    clustering_results, silhouette_results, CAH_results, DB_results = [], [], [], []
    print('_'*(100//NB_CLUSTER_SIZES_TO_TEST*NB_CLUSTER_SIZES_TO_TEST))

    for i in range(NB_CLUSTER_SIZES_TO_TEST) :
        clustering = KMeans(n_clusters=i+2, n_init=10).fit(X)
        clustering_results.append(clustering)
        labels = clustering.labels_

        silhouette_results.append(silhouette_score(X, labels))
        CAH_results.append(calinski_harabasz_score(X, labels))
        DB_results.append(davies_bouldin_score(X, labels))
        print(100//NB_CLUSTER_SIZES_TO_TEST*'#', end='', flush=True)


    print()
    silhouette_normed = norm(silhouette_results)
    CAH_normed = norm(CAH_results)
    DB_normed = norm(DB_results, True)

    average_scores = [(silhouette_normed[i] + CAH_normed[i] + DB_normed[i]) / 3 for i in range(NB_CLUSTER_SIZES_TO_TEST)]
    best_clustering_index = average_scores.index(max(average_scores))

    plot_kMeans(best_clustering_index, clustering_results, silhouette_results, CAH_results, DB_results)


def plot_kMeans(best_clustering, clustering_results, silhouette_results, CAH_results, DB_results) :
    print(f"\nPlotting K-Means results")

    _, axes = plt.subplots(2, 2, figsize=(8, 12))  # 2 lignes, 2 colonnes
    
    # --- 1. Silhouette ---
    axes[0, 0].plot(range(2, NB_CLUSTER_SIZES_TO_TEST+2), silhouette_results, marker='o', color='blue', label='Silhouette')
    axes[0, 0].set_title('Indice Silhouette (MAX)')
    axes[0, 0].set_xlabel('Nombre de clusters')
    axes[0, 0].set_ylabel('Silhouette')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # --- 2. CAH ---
    axes[0, 1].plot(range(2, NB_CLUSTER_SIZES_TO_TEST+2), CAH_results, marker='o', color='green', label='CAH')
    axes[0, 1].set_title('Indice CAH (MAX)')
    axes[0, 1].set_xlabel('Nombre de clusters')
    axes[0, 1].set_ylabel('CAH')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # --- 3. Davies-Bouldin ---
    axes[1, 0].plot(range(2, NB_CLUSTER_SIZES_TO_TEST+2), DB_results, marker='o', color='red', label='Davies-Bouldin')
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


def run_agglomerative():
    results_by_linkage = {}

    for linkage in ['ward', 'complete', 'average', 'single']:
        print(f"\nTesting {linkage} linkage")
        print('_'*(100//NB_CLUSTER_SIZES_TO_TEST*NB_CLUSTER_SIZES_TO_TEST))
        clustering_results, silhouette_results, CAH_results, DB_results = [], [], [], []

        for i in range(NB_CLUSTER_SIZES_TO_TEST) :
            clustering = AgglomerativeClustering(n_clusters=i+2, linkage=linkage).fit(X)

            clustering_results.append(clustering)
            labels = clustering.labels_
            
            silhouette_results.append(silhouette_score(X, labels))
            CAH_results.append(calinski_harabasz_score(X, labels))
            DB_results.append(davies_bouldin_score(X, labels))
            print(100//NB_CLUSTER_SIZES_TO_TEST*'#', end='', flush=True)


        print()
        silhouette_normed = norm(silhouette_results)
        cah_normed = norm(CAH_results)
        db_normed = norm(DB_results, True)

        average_scores = [(silhouette_normed[j] + cah_normed[j] + db_normed[j])/3 for j in range(NB_CLUSTER_SIZES_TO_TEST)]
        best_clustering_index = average_scores.index(max(average_scores))
        
        best_k_for_linkage = best_clustering_index + 2
        best_clustering_for_linkage = clustering_results[best_clustering_index]
        
        results_by_linkage[linkage] = (best_clustering_for_linkage, best_k_for_linkage)

    plot_agglomerative(results_by_linkage)


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


def find_elbow(k):
    '''
    Utilise la méthode du coude pour déterminer une valeur appropriée de epsilon pour DBSCAN.
    Le coude est calculé en cherchant le point le plus éloigné de la droite reliant le premier et le dernier point
    '''
    neighbors = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = neighbors.kneighbors(X)
    distances = np.sort(distances[:, k-1])
    
    (vx, vy) = (len(distances), (distances[-1] - distances[0]))
    (a, b) = (1, -vx/vy)  # Vecteur orthogonal à la droite passant pas le premier et le dernier point

    coude = 0  # Abscisse du point le plus éloigné de la droite, correspondant au coude
    max_delta = 0

    for i in range(len(distances)) :
        (x, y) = (i, distances[i])

        delta = abs(-(a*x + b*(y-distances[0]))/(a*a + b*b)**0.5)  # Distance du point (x, y) à la droite

        if delta > max_delta :
            max_delta = delta
            coude = x
    

    print(f"Coude trouvé à l'index {coude}, distance = {round(distances[coude], 2)}")

    plt.plot(distances)
    plt.plot([0, len(distances)], [distances[0], distances[-1]], 'k--')
    plt.plot(coude, distances[coude], 'ro', label='Coude')
    plt.xlabel('Points triés par distance')
    plt.ylabel(f'Distance au {k}-ème voisin')
    plt.legend()
    plt.title('Méthode du coude pour choisir epsilon')
    plt.show()

    return distances[coude]


def run_DBSCAN() :
    clustering_results, silhouette_results, CAH_results, DB_results = [], [], [], []
    coude = find_elbow(3)  # Données à N dimensions, k = 2N-1

    for eps in np.arange(coude, 3*coude, coude/4) :
        print(f"\nDBSCAN avec eps = {round(eps, 3)}")
        clustering = DBSCAN(eps=eps, min_samples=4).fit(X)
        
        labels = clustering.labels_

        if len(set(labels)) >= 2 :
            clustering_results.append(clustering)
            silhouette_results.append(silhouette_score(X, labels))
            CAH_results.append(calinski_harabasz_score(X, labels))
            DB_results.append(davies_bouldin_score(X, labels))

        labels = clustering.labels_

        # Calcul des centres à partir des labels
        unique_labels = np.unique(labels)
        centers = np.array([X[labels == lab].mean(axis=0) for lab in unique_labels if lab != -1])

        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
        if centers.size > 0:
            plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centres')

        plt.title(f"{type(clustering).__name__}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
            

    print()
    silhouette_normed = norm(silhouette_results)
    CAH_normed = norm(CAH_results)
    DB_normed = norm(DB_results, True)

    average_scores = [(silhouette_normed[i] + CAH_normed[i] + DB_normed[i]) / 3 for i in range(len(clustering_results))]
    best_clustering_index = average_scores.index(max(average_scores))

    plot_DBSCAN(best_clustering_index, clustering_results)

    
def plot_DBSCAN(best_clustering_index, clustering_results) :
    clustering = clustering_results[best_clustering_index]
    labels = clustering.labels_

    # Calcul des centres à partir des labels
    unique_labels = np.unique(labels)
    centers = np.array([X[labels == lab].mean(axis=0) for lab in unique_labels if lab != -1])

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
    if centers.size > 0:
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centres')

    plt.title(f"{type(clustering).__name__} avec {best_clustering_index+2} clusters")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


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
            run_DBSCAN()

        # --- HDBSCAN ---
        case "HDBScan" :
            # clustering = hdbscan.HDBSCAN(min_cluster_size=5).fit(X)
            print("HDBScan non implémenté.")

        case _ :
            raise ValueError(f"Méthode de clustering inconnue : {method}")


if __name__ == "__main__" :
    NB_CLUSTER_SIZES_TO_TEST = 8  # Pour K-Means et Clustering Hiérarchique

    method = "KMeans"
    if len(sys.argv) >= 2 and sys.argv[1] in ["KMeans", "Agglomerative", "DBScan", "HDBScan"] :
        method = sys.argv[1]

    fichier = "2d-20c-no0"
    if len(sys.argv) >= 3 and os.path.isfile("artificial/" + sys.argv[2] + ".arff") :
        fichier = sys.argv[2]

    print(f"Chargement des données depuis le fichier '{fichier}' et utilisation de la méthode '{method}'")
    data, _ = arff.loadarff("artificial/" + fichier + ".arff")
    df = pd.DataFrame(data).iloc[:, :2]
    X = df.values

    clusterise(method)