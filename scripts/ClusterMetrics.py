import numpy as np

def calinski_harabasz_score_precomputed(D, labels):
    n_samples = len(labels)  # Total number of data points
    unique_labels = np.unique(labels)  # Unique cluster labels
    n_clusters = len(unique_labels)  # Total number of clusters
    
    # Compute the overall mean (centroid of all points)
    overall_mean = np.mean(D, axis=0)
    
    # Initialize between-cluster dispersion
    B_k = 0
    for label in unique_labels:
        # Extract points belonging to the current cluster
        cluster_points = D[labels == label]
        # Compute mean of the current cluster
        cluster_mean = np.mean(cluster_points, axis=0)
        # Sum the squared distance between cluster mean and overall mean, weighted by cluster size
        B_k += len(cluster_points) * np.sum((cluster_mean - overall_mean) ** 2)
    
    # Initialize within-cluster dispersion
    W_k = 0
    for label in unique_labels:
        cluster_points = D[labels == label]
        cluster_mean = np.mean(cluster_points, axis=0)
        # Sum the squared distances between points and their cluster mean
        W_k += np.sum((cluster_points - cluster_mean) ** 2)
    
    # Calculate Calinski-Harabasz score
    score = (B_k / W_k) * (n_samples - n_clusters) / (n_clusters - 1)
    
    return score

def davies_bouldin_score_precomputed(D, labels):
    n_samples = len(labels)  # Total number of data points
    unique_labels = np.unique(labels)  # Unique cluster labels
    n_clusters = len(unique_labels)  # Total number of clusters
    
    # Compute the centroids of each cluster
    centroids = np.array([np.mean(D[labels == label], axis=0) for label in unique_labels])
    
    # Compute within-cluster scatter for each cluster
    S = np.zeros(n_clusters)
    for idx, label in enumerate(unique_labels):
        cluster_points = D[labels == label]
        cluster_mean = centroids[idx]
        # Average distance between points and the cluster centroid
        S[idx] = np.mean(np.linalg.norm(cluster_points - cluster_mean, axis=1))
    
    # Compute pairwise distances between centroids
    M = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                M[i, j] = np.linalg.norm(centroids[i] - centroids[j])
    
    # Calculate Davies-Bouldin score
    db_score = 0
    for i in range(n_clusters):
        # Find the maximum ratio of scatter to centroid separation for cluster i
        max_ratio = np.max([(S[i] + S[j]) / M[i, j] for j in range(n_clusters) if i != j])
        db_score += max_ratio
    # Average the maximum ratios to compute the final score
    db_score /= n_clusters
    
    return db_score

def silhouette_score_precomputed(D, labels):
    n_samples = len(labels)  # Total number of data points
    unique_labels = np.unique(labels)  # Unique cluster labels
    
    silhouette_scores = np.zeros(n_samples)  # Array to store silhouette scores for each point
    
    for i in range(n_samples):
        current_label = labels[i]  # Cluster label of the current point
        
        # Compute a(i): average distance to other points in the same cluster
        in_cluster_mask = (labels == current_label)
        if np.sum(in_cluster_mask) > 1:
            a_i = np.mean(D[i][in_cluster_mask][D[i][in_cluster_mask] != 0])  # Exclude self-distance (0)
        else:
            a_i = 0  # If the point is alone in its cluster
        
        # Compute b(i): average distance to the nearest different cluster
        b_i = np.inf
        for label in unique_labels:
            if label == current_label:
                continue  # Skip the same cluster
            other_cluster_mask = (labels == label)
            b_i = min(b_i, np.mean(D[i][other_cluster_mask]))
        
        # Calculate silhouette score for point i
        silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
    
    # Return the overall silhouette score as the mean of all individual scores
    return np.mean(silhouette_scores)
