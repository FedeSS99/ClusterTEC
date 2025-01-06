from scripts.libraries import *

class ClusterVTECDataMDS:
    def __init__(self, dissimilarity:np.ndarray) -> None:
        # Initialize with a dissimilarity matrix
        self.__dissimilarity_matrix = dissimilarity

    def ComputeMDS(self, num_comps_mds = 2, method = "Classic") -> float:
        # Perform MDS (Multidimensional Scaling) on the dissimilarity matrix
        self.__MDS_TScluster = ClustTimeMDS(self.__dissimilarity_matrix)
        self.Xc_TS = self.__MDS_TScluster.fit(num_comps_mds, method = method)

        # Return the normalized stress value as a measure of MDS quality
        return self.__MDS_TScluster.normalized_stress

    def ClusterTSVectors(self, num_clusters = 2, cluster_method = "K-Means", Labels = None) -> None:
        # Cluster the time series vectors obtained from MDS
        if not isinstance(Labels, np.ndarray):
            # Apply K-Means clustering if no labels are provided
            if cluster_method == "K-Means":
                KMeans_Cluster_TS = KMeans(n_clusters = num_clusters, init = "k-means++")
                self.Xc_Labels = KMeans_Cluster_TS.fit_predict(self.Xc_TS)
                
                # Calculate clustering evaluation scores
                silhouette_score_kmeans = silhouette_score(self.Xc_TS, self.Xc_Labels)
                CH_score_kmeans = calinski_harabasz_score(self.Xc_TS, self.Xc_Labels)
                DB_score_kmeans = davies_bouldin_score(self.Xc_TS, self.Xc_Labels)
                
                # Store cluster centers
                self.centers = KMeans_Cluster_TS.cluster_centers_
                
                # Print evaluation scores
                print(f"--Scores with K-Means clustering--\nSH coefficient = {silhouette_score_kmeans}\nCH index = {CH_score_kmeans}\nDB index = {DB_score_kmeans}")
            
            # Apply Gaussian Mixture clustering if specified
            elif cluster_method == "Gaussian":
                GaussianMix_Cluster_TS = GaussianMixture(n_components = num_clusters, covariance_type = "full")
                self.Xc_Labels = GaussianMix_Cluster_TS.fit_predict(self.Xc_TS)
                
                # Calculate clustering evaluation scores
                silhouette_score_gaussmix = silhouette_score(self.Xc_TS, self.Xc_Labels)
                CH_score_gaussmix = calinski_harabasz_score(self.Xc_TS, self.Xc_Labels)
                DB_score_gaussmix = davies_bouldin_score(self.Xc_TS, self.Xc_Labels)
                
                # Store cluster means as centers
                self.centers = GaussianMix_Cluster_TS.means_
                
                # Print evaluation scores
                print(f"--Scores with GaussianMix clustering--\nSH coefficient = {silhouette_score_gaussmix}\nCH index = {CH_score_gaussmix}\nDB index = {DB_score_gaussmix}")
        else:
            # If labels are provided, no clustering is performed
            self.Xc_Labels = None
        
        # Calculate and print the total number of series in each cluster
        TotalSeriesPerCluster = dict(Counter(self.Xc_Labels))
        print("--Total series for every cluster--")
        for key_cluster in sorted(list(TotalSeriesPerCluster.keys())):
            print(f"{key_cluster} -> {TotalSeriesPerCluster[key_cluster]}")

    def VisualizeClustering(self, Labels = None) -> None:
        # Visualize the clustering results
        if isinstance(Labels, np.ndarray):
            # Use provided labels for visualization
            self.__ColorLabels = colormaps["brg"](Normalize(vmin = Labels.min(), vmax = Labels.max())(Labels))
            self.__MDS_TScluster.VisualizeVectors(Colors = self.__ColorLabels)
        elif Labels == None and isinstance(self.Xc_Labels, np.ndarray):
            # Visualize based on calculated cluster labels if available
            self.__ColorLabels = colormaps["brg"](Normalize(vmin = self.Xc_Labels.min(), vmax = self.Xc_Labels.max())(self.Xc_Labels))
            self.__MDS_TScluster.VisualizeVectors(Colors = self.__ColorLabels)
        else:
            # Handle the case where labels are not defined
            print("Labels are not defined or were not given. Please check")
