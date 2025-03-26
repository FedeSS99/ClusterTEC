import numpy as np
from tqdm import tqdm
from scipy.stats.mstats import mquantiles

class SYNCLUS:
    """
    SYNCLUS class implements a clustering algorithm based on minimizing a dissimilarity measure.
    """

    def __init__(self, dissim: np.ndarray, K: int, iter: int, reps: int) -> None:
        """
        Initialize the SYNCLUS class with dissimilarity matrix, number of clusters, iterations, and repetitions.

        Parameters:
        - dissim: Dissimilarity matrix (NxN).
        - K: Number of clusters.
        - iter: Maximum number of iterations for KMeans.
        - reps: Number of repetitions for the clustering process.
        """
        self.K = K
        self.iter = iter
        self.reps = reps

        self.D2 = dissim ** 2.0
        self.EPS = np.zeros(self.reps)
        self.KMS = self.reps * [0]


    def __random_centroids(self):
        """
        Generate random initial centroids for the clustering process.
        """
        RNG = np.random.default_rng()
        centroids = RNG.choice(np.arange(self.D2.shape[0]), self.K, replace = False)

        return centroids
    
    def __assign_centers(self, centers, N):
        """
        Assign initial cluster labels to the centroids.

        Parameters:
        - centers: Indices of the centroids.
        - N: Total number of data points.

        Returns:
        - clusters: Array with cluster assignments.
        """
        clusters = np.zeros(N)
        clusters[centers] = np.arange(1, centers.size + 1)
        return clusters

    def __initial_assign(self, centers):
        """
        Assign each data point to the nearest centroid.

        Parameters:
        - centers: Indices of the centroids.

        Returns:
        - clus: Array with initial cluster assignments.
        """
        N = self.D2.shape[0]

        # Assign initial clusters for the centroids
        clus = self.__assign_centers(centers, N)

        # For each point, if it's not a centroid, assign it to the nearest centroid
        for i in range(N):
            if i not in centers:
                # Find the index of the nearest centroid for point i
                cen = centers[np.argmin(self.D2[i, centers])]  # Nearest centroid
                clus[i] = np.argwhere(centers == cen)[0] # Find the corresponding cluster (1-based index)
    
        return clus


    def __KMeans_SYNCLUS(self):    
        """
        Perform the KMeans clustering algorithm with the SYNCLUS approach.

        Returns:
        - Dictionary containing start clusters, end clusters, and EP values.
        """
        centers = self.__random_centroids()  # Initial centroids
    
        # Initial assignment of points to clusters
        start_clusters = self.__initial_assign(centers = centers)
    
        new_clusters = np.copy(start_clusters).astype(int)  # Initial clusters
        EP = [np.finfo(np.float64).max]  # First value of EP (Machine double max)
        clus_itera = np.zeros((self.iter + 1, self.D2.shape[0]), dtype=int)  # Store clusters at each iteration
        clus_itera[0, :] = start_clusters  # First clusters
    
        t = 1
        while t <= self.iter:
            Js = np.bincount(new_clusters)[1:]  # Number of elements in each cluster (ignore cluster 0)

            if len(Js) != self.K:  # If not all clusters are assigned, restart
                t = 0
                centers = self.__random_centroids()
                start_clusters = self.__initial_assign(centers=centers)
                new_clusters = start_clusters
                clus_itera[0, :] = start_clusters
            else:
                data = np.zeros((self.D2.shape[0], self.K))  # Will store D^2_{jk}
                ep = 0  # Variable to store EP

                for k in range(self.K):
                    Jk = Js[k]  # Number of elements in cluster k
                    Dk2 = (1 / (2 * (Jk ** 2))) * np.sum(self.D2[np.ix_(new_clusters == (k + 1), new_clusters == (k + 1))])  # D^2_k
                    data[:, k] = (1 / Jk) * np.sum(self.D2[:, new_clusters == (k + 1)] - Dk2, axis=1)  # D^2_{jk}
                    ep_aux = (1 / (2 * Jk)) * np.sum(self.D2[np.ix_(new_clusters == (k + 1), new_clusters == (k + 1))])  # EP
                    ep += ep_aux  # Cumulate EP

                EP.append(ep)  # Update EP
                new_clusters = np.argmin(data, axis=1) + 1  # Select new clusters
                clus_itera[t, :] = new_clusters  # Store clusters

                if EP[t] >= EP[t - 1]:  # Stop criterion if clusters don't change
                    break
                
            t += 1
    
        end_clusters = clus_itera[np.argmin(EP), :]  # Final clusters
        EP = EP[1:]  # Remove the first max value

        return {
            'start_clusters': start_clusters,
            'end_clusters': end_clusters,
            'EP': EP
        }

    def __preKMeans(self):
        """
        Preprocess the data and handle edge cases before running KMeans.

        Returns:
        - Dictionary containing start clusters, end clusters, and EP values.
        """
        N = self.D2.shape[0]

        if self.K == 1:
            start_clusters = np.ones(N)
            end_clusters = np.ones(N)
            EP = 0.0
        
        elif self.K == N:
            start_clusters = np.arange(1, N+ 1)
            end_clusters = np.arange(1, N+ 1)
            EP = 0.0

        elif self.K > N:
            print("More clusters than data")
            return False
        
        else:
            KMeans_Results = self.__KMeans_SYNCLUS()
            start_clusters = KMeans_Results["start_clusters"]
            end_clusters = KMeans_Results["end_clusters"]
            EP = KMeans_Results["EP"]

        return {"start_clusters":start_clusters, "end_clusters":end_clusters, "EP":EP}
 
    def fit_predict(self):
        """
        Run the SYNCLUS algorithm for the specified number of repetitions and return the best clustering result.

        Returns:
        - Dictionary containing the best clustering result based on the minimum EP value.
        """
        for index in tqdm(range(self.reps)):
            KMeans_results = self.__preKMeans()

            self.EPS[index] = KMeans_results["EP"][-1]
            self.KMS[index] = KMeans_results

        min_EPS_index = np.argmin(self.EPS)
        min_quartiles_max = np.round(mquantiles(self.EPS, prob = [0.0, 0.25, 0.5, 0.75, 1.0]), decimals = 2).tolist()

        print("Best SYNCLUS at ", min_EPS_index, "with ", self.K, " clusters")
        print("Min, Quartiles, Max")
        print(", ".join(tuple(map(lambda x: str(x), min_quartiles_max))))
        
        KMeansByMinEPS = self.KMS[min_EPS_index]
        return KMeansByMinEPS
