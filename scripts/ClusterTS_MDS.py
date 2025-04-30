from scripts.libraries import *

class ClusterMDS:
    """
    Class to perform clustering on VTEC data using Multidimensional Scaling (MDS).
    """
    def __init__(self, dissimilarity: np.ndarray) -> None:
        """
        Initialize the class with a dissimilarity matrix.

        Parameters:
        - dissimilarity: Precomputed dissimilarity matrix for the data.
        """
        self.__dissimilarity_matrix = dissimilarity

    def ComputeMDS(self, num_comps_mds = 2, method="Classic", max_iter: int = 500, eps: float = 1e-6, verbose: int = 0, visualize_shepard: bool = True) -> float:
        """
        Perform Multidimensional Scaling (MDS) on the dissimilarity matrix.

        Parameters:
        - num_comps_mds: Number of components for MDS.
        - method: MDS method ("Classic" or other).
        - max_iter: Maximum number of iterations.
        - eps: Convergence tolerance.
        - verbose: Verbosity level.
        - visualize_shepard: Whether to visualize Shepard diagram.

        Returns:
        - Normalized stress value as a measure of MDS quality.
        """
        # Perform MDS (Multidimensional Scaling) on the dissimilarity matrix
        self.__MDS_TScluster = TimeSeriesMDS(self.__dissimilarity_matrix)
        self.Xc_TS = self.__MDS_TScluster.fit(num_comps_mds, method = method, max_iter = max_iter, eps = eps, verbose = verbose, visualize_shepard = visualize_shepard)

        # Return the normalized stress value as a measure of MDS quality
        return self.__MDS_TScluster.normalized_stress

    def ClusterTSVectors(self, num_clusters=2, cluster_method="K-Means") -> None:
        """
        Cluster the time series vectors obtained from MDS.

        Parameters:
        - num_clusters: Number of clusters.
        - cluster_method: Clustering method ("K-Means" or "GaussMix").
        """
        # Cluster the time series vectors obtained from MDS
        if num_clusters >= 2 and cluster_method in ["K-Means", "Gaussian"]:
            # Apply K-Means clustering if no labels are provided
            if cluster_method == "K-Means":
                KMeans_Cluster_TS = KMeans(n_clusters = num_clusters, init = "k-means++")
                self.Xc_Labels = KMeans_Cluster_TS.fit_predict(self.Xc_TS)
                # Store cluster centers
                self.centers = KMeans_Cluster_TS.cluster_centers_
            
            # Apply Gaussian Mixture clustering if specified
            elif cluster_method == "GaussMix":
                GaussianMix_Cluster_TS = GaussianMixture(n_components = num_clusters, covariance_type = "full")
                self.Xc_Labels = GaussianMix_Cluster_TS.fit_predict(self.Xc_TS)
                
                # Store cluster means as centers
                self.centers = GaussianMix_Cluster_TS.means_

            # Calculate clustering evaluation scores
            silhouette_score_cluster = silhouette_score(self.Xc_TS, self.Xc_Labels)
            CH_score_kmeans_cluster = calinski_harabasz_score(self.Xc_TS, self.Xc_Labels)
            DB_score_kmeans_cluster = davies_bouldin_score(self.Xc_TS, self.Xc_Labels)
            print(f"--Scores with {cluster_method} clustering--\nSH coefficient = {silhouette_score_cluster}\nCH index = {CH_score_kmeans_cluster}\nDB index = {DB_score_kmeans_cluster}")

            # Calculate and print the total number of series in each cluster
            TotalSeriesPerCluster = dict(Counter(self.Xc_Labels))
            print("--Total series for every cluster--")
            for key_cluster in sorted(list(TotalSeriesPerCluster.keys())):
                print(f"{key_cluster} -> {TotalSeriesPerCluster[key_cluster]}")

            # Compute distances to centroids and sort each cluster
            self.cluster_order = {}  # Dictionary to store ordered indices per cluster
            for cluster in range(num_clusters):
                indices_in_cluster = np.where(self.Xc_Labels == cluster)[0]
                distances = np.linalg.norm(self.Xc_TS[indices_in_cluster] - self.centers[cluster], axis=1)
                sorted_indices = indices_in_cluster[np.argsort(distances)]  # Sort indices by distance
                self.cluster_order[cluster] = sorted_indices  # Store the ordered indices

        else:
            # If labels are provided, no clustering is performed
            self.Xc_Labels = None

    def VisualizeClustering(self, Labels=None) -> None:
        """
        Visualize the clustering results.

        Parameters:
        - Labels: Optional array of labels for visualization.
        """
        self.__ColorLabels = None
        # Visualize the clustering results
        if isinstance(Labels, np.ndarray):
            # Use provided labels for visualization
            self.__ColorLabels = colormaps["brg"](Normalize(vmin = Labels.min(), vmax = Labels.max())(Labels))
        elif Labels == None and isinstance(self.Xc_Labels, np.ndarray):
            # Visualize based on calculated cluster labels if available
            self.__ColorLabels = colormaps["brg"](Normalize(vmin = self.Xc_Labels.min(), vmax = self.Xc_Labels.max())(self.Xc_Labels))
        
        # Pass centroids to the visualization method
        self.__MDS_TScluster.VisualizeVectors(Colors = self.__ColorLabels, Centroids = self.centers if hasattr(self, 'centers') else None)
