KMeans : 
    - nb_clusters
    - algorithm (lloyd ou elkan) ?


Agglomerative :
    - nb_clusters ou distance_threshold
        itérer sur le nombre de cluster est plus rapide car il y a moins de valeur possibles
        distance_threshold permettrait, en regardant le dandogramme de trouver où couper, mais demanderait beaucoup plus de code

    - linkage (ward, complete, average ou single)
        single est le seul qui regroupe bien en deux clusters banana.arff
        single permet de mieux regrouper lorsque les clusters sont etalés en longueur (banana, cuboids, ...)
        mais il y a des cas où il marche mal
        -> ward est un bon compromis, même si plus lent


DBSCAN :
    - eps
    - min_samples
    - alogrithm ?