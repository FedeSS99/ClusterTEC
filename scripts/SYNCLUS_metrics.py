import numpy as np

def calinski(N, max_K, BK, WK):
    """
    Calinski-Harabasz index calculation.
    
    Parameters:
    N : int
        Number of data points
    max_K : int
        Maximum number of clusters considered
    BK : array-like
        Between-cluster dispersion for each number of clusters
    WK : array-like
        Within-cluster dispersion for each number of clusters
        
    Returns:
    dict
        Dictionary with Calinski-Harabasz scores and the optimal number of clusters
    """
    CH = np.zeros(max_K)
    
    for K in range(2, max_K):
        numerator = BK[K] / (K * (K + 1) / 2)
        denominator = WK[K] / (((N * (N - 1)) - (K * (K + 1))) / 2)
        CH[K] = numerator / denominator
    
    K_optim = np.argmax(CH)
    
    return {"calinski": CH, "K_optim": K_optim}

def hartigan(N, WK, max_K):
    """
    Hartigan's index calculation.
    
    Parameters:
    N : int
        Number of data points
    WK : array-like
        Within-cluster dispersion for each number of clusters
    max_K : int
        Maximum number of clusters considered
        
    Returns:
    dict
        Dictionary with Hartigan's scores and the optimal number of clusters
    """
    HK = np.zeros(max_K - 1)
    criterio2 = np.zeros(max_K - 1, dtype=bool)
    
    for K in range(1, max_K):
        f1 = (WK[K - 1] / WK[K]) - 1
        f2 = (((N * (N - 1)) - (K * (K + 1))) / 2) - 1
        HK[K - 1] = f1 * f2
        criterio2[K - 1] = HK[K - 1] <= 5 * N
    
    optimo = np.where(criterio2)[0]
    if optimo.size == 0:
        optimo = 0  # If no optimal value is found, set to 1st index (equivalent to R)
    else:
        optimo = optimo[0] + 1  # Adding 1 to match cluster index
    
    return {"hartigan": HK, "K_optim": optimo}

def silhouette_score(D, result_kmeans):
    """
    Silhouette score calculation for clustering.
    
    Parameters:
    D : ndarray
        Data points array (N x features)
    result_kmeans : dict
        KMeans result with keys 'clusters' indicating the cluster assignment of each data point
        
    Returns:
    ndarray
        Silhouette scores for each data point
    """
    def calcula_E(clusters):
        """
        Calculate the cluster size or count of points in each cluster.
        """
        _, counts = np.unique(clusters, return_counts=True)
        return counts
    
    N = D.shape[0]
    clusters = result_kmeans['clusters']
    E = calcula_E(clusters)
    
    ais = np.zeros(N)
    bis = np.zeros(N)
    silhouette_scores = np.zeros(N)
    unique_clusters = np.unique(clusters)
    
    for i in range(N):
        current_cluster = clusters[i]
        
        # Calculate 'a' - mean intra-cluster distance
        intra_cluster_distances = np.mean([np.linalg.norm(D[i] - D[j]) 
                                           for j in range(N) if clusters[j] == current_cluster and i != j])
        ais[i] = intra_cluster_distances
        
        # Calculate 'b' - mean nearest-cluster distance
        inter_cluster_distances = []
        for cluster in unique_clusters:
            if cluster != current_cluster:
                mean_inter_dist = np.mean([np.linalg.norm(D[i] - D[j]) 
                                           for j in range(N) if clusters[j] == cluster])
                inter_cluster_distances.append(mean_inter_dist)
        
        bis[i] = np.min(inter_cluster_distances) if inter_cluster_distances else 0
        
        # Silhouette score for point i
        a_i = ais[i]
        b_i = bis[i]
        silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) != 0 else 0
    
    return silhouette_scores