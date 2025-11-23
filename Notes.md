KMeans : 
    - nb_clusters
    Mauvais lorsque les clusters ne sont pas convexes


Agglomerative :
    - nb_clusters ou distance_threshold
        itérer sur le nombre de cluster est plus rapide car il y a moins de valeur possibles
        distance_threshold permettrait, en regardant le dendogramme de trouver où couper, mais demanderait beaucoup plus de code

    - linkage (ward, complete, average ou single)
        single est le seul qui regroupe bien en deux clusters banana.arff
        single permet de mieux regrouper lorsque les clusters sont pas convexes (banana, cuboids, ...)
        mais il y a des cas où il marche mal
        -> ward est un bon compromis, même si plus lent


DBSCAN :
    - eps -> 2x distance max à la diagonale pour trouver le coude :
        (x,y) et (a,b) => (xa + yb = 0) => (a=1, b = -x/y)

    - min_samples
        itérer sur des petites valeurs (3, 4, 5, 6)