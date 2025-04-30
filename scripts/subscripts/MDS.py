import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import euclidean_distances
from matplotlib.pyplot import subplots

from sklearn.manifold import MDS

class TimeSeriesMDS:
    """
    TimeSeriesMDS class implements various methods for performing Multidimensional Scaling (MDS)
    on time series data using different approaches.
    """

    def __init__(self, dissimilarity: np.ndarray) -> None:
        """
        Initialize the TimeSeriesMDS class with a dissimilarity matrix.

        Parameters:
        - dissimilarity: Dissimilarity matrix (NxN).
        """
        self.__dissim = dissimilarity

        self.N = self.__dissim.shape[0]
        self.__H = np.eye(N = self.N) - (1/self.N)*np.full((self.N, self.N), 1.0)

    def __compute_euclidean_B(self):
        """
        Compute the double-centered matrix B from the dissimilarity matrix.
        """
        self.__B = -0.5 * (self.__H @ (self.__dissim ** 2.0) @ self.__H.T)

    def __get_smallest_eigen_value(self) -> np.float32:
        """
        Compute the smallest positive eigenvalue of the matrix B.

        Returns:
        - Smallest positive eigenvalue.
        """
        B_eigvals = np.linalg.eigvals(self.__B)
        min_B_eigen_values = B_eigvals[B_eigvals > 0].min()

        return min_B_eigen_values

    def __GetEuclideanDistances(self):
        """
        Convert the dissimilarity matrix into a Euclidean distance matrix.
        """
        self.__compute_euclidean_B()
        MinEigVal = self.__get_smallest_eigen_value()

        self.__EuclidDist = np.sqrt((self.__dissim**2.0) - 2.0 * MinEigVal * (np.full((self.N, self.N), 1.0) - np.eye(N = self.N)))

    def __VisualizeShepardPlot(self, orig_dissim:np.ndarray, red_dissim:np.ndarray):
        """
        Visualize the Shepard plot comparing original and reduced dissimilarities.

        Parameters:
        - orig_dissim: Original dissimilarity matrix.
        - red_dissim: Reduced dissimilarity matrix.
        """
        Figure, Subplot = subplots(nrows = 1, ncols = 1, figsize = (6, 6))

        upper_triu_index = np.triu_indices(n = orig_dissim.shape[0], k = 1)
        x, y = orig_dissim[upper_triu_index], red_dissim[upper_triu_index]
        max_x = x.max()

        Subplot.scatter(x, y, s = 4, c = "blue", marker = "o", alpha = 0.1)
        Subplot.plot([0.0, max_x], [0.0, max_x], "-k")
        
        Subplot.set_xlabel("Original dissimilarities")
        Subplot.set_ylabel("Reduced dissimilarities")
        Subplot.set_xlim(left = 0.0, right = max_x)
        Subplot.set_ylim(bottom = 0.0)
        
        Figure.tight_layout()

    def __compute_stress_1(self, D_orig, D_red):
        """
        Compute Kruskal's Stress-1 given original and reduced dissimilarity matrices.

        Parameters:
        - D_orig: Original dissimilarity matrix.
        - D_red: Reduced dissimilarity matrix.

        Returns:
        - Stress-1 value.
        """
        # Extract upper triangular parts (excluding diagonal) to avoid redundancy
        triu_idx = np.triu_indices_from(D_orig, k=1)

        # Compute squared differences
        num = np.sum((D_orig[triu_idx] - D_red[triu_idx]) ** 2)
        denom = np.sum(D_orig[triu_idx] ** 2)

        # Compute Stress-1
        return np.sqrt(num / denom)

    def fit(self, num_comps=2, method="Classic", max_iter=500, eps=1e-6, verbose=0, visualize_shepard=True):
        """
        Fit the MDS model using the specified method and parameters.

        Parameters:
        - num_comps: Number of components for dimensionality reduction.
        - method: MDS method to use (e.g., "classic", "SMACOF-euclidean").
        - max_iter: Maximum number of iterations for SMACOF.
        - eps: Convergence tolerance for SMACOF.
        - verbose: Verbosity level for SMACOF.
        - visualize_shepard: Whether to visualize the Shepard plot.

        Returns:
        - Reduced data matrix (Xc).
        """
        def compute_classic_embedding(dissimilarity_matrix):
            """Compute the classic MDS embedding."""
            B = -0.5 * (self.__H @ (dissimilarity_matrix ** 2.0) @ self.__H.T)
            eigvals, eigvecs = np.linalg.eigh(B)
            return np.fliplr(np.sqrt(eigvals[-num_comps:]) * eigvecs[:, -num_comps:])

        def compute_smacof_embedding(dissimilarity_matrix, init_conf=None):
            """Compute the SMACOF MDS embedding."""
            mds = MDS(
                n_components=num_comps,
                n_jobs=-1,
                dissimilarity="precomputed",
                max_iter=max_iter,
                eps=eps,
                verbose=verbose,
                n_init=1 if init_conf is not None else 4,
            )
            return mds.fit(dissimilarity_matrix, init=init_conf).embedding_

        def process_method(dissimilarity_matrix, use_euclidean=False, use_smacof=False, use_classic_init=False):
            """Process the specified method and compute the embedding."""
            if use_euclidean:
                self.__GetEuclideanDistances()
                dissimilarity_matrix = self.__EuclidDist

            if use_smacof:
                init_conf = None
                if use_classic_init:
                    init_conf = compute_classic_embedding(dissimilarity_matrix)
                self.Xc = compute_smacof_embedding(dissimilarity_matrix, init_conf)
            else:
                self.Xc = compute_classic_embedding(dissimilarity_matrix)

            distances_Xc = euclidean_distances(self.Xc)
            self.normalized_stress = self.__compute_stress_1(dissimilarity_matrix, distances_Xc)

            if visualize_shepard:
                self.__VisualizeShepardPlot(dissimilarity_matrix, distances_Xc)

        # Map methods to their configurations
        method_config = {
            "classic": {"use_euclidean": True, "use_smacof": False, "use_classic_init": False},
            "dissim": {"use_euclidean": False, "use_smacof": False, "use_classic_init": False},
            "SMACOF-euclidean": {"use_euclidean": True, "use_smacof": True, "use_classic_init": False},
            "SMACOF-dissim": {"use_euclidean": False, "use_smacof": True, "use_classic_init": False},
            "SMACOF-euclidean-classic": {"use_euclidean": True, "use_smacof": True, "use_classic_init": True},
            "SMACOF-dissim-classic": {"use_euclidean": False, "use_smacof": True, "use_classic_init": True},
        }

        if method not in method_config:
            raise ValueError(f"Unknown method: {method}")

        # Process the method
        config = method_config[method]
        process_method(
            dissimilarity_matrix=self.__dissim,
            use_euclidean=config["use_euclidean"],
            use_smacof=config["use_smacof"],
            use_classic_init=config["use_classic_init"],
        )

        print(f"{method} with {num_comps} components has a stress-1 value of {self.normalized_stress:.6f}")
        return self.Xc

    def VisualizeVectors(self, Colors=None, Centroids=None):
        """
        Visualize the reduced vectors in pairwise scatter plots and diagonal density plots.

        Parameters:
        - Colors: Optional array of colors for the scatter plots.
        - Centroids: Optional array of centroids to visualize.
        """
        num_dims = self.Xc.shape[1]

        Figure, Subplots = subplots(nrows=num_dims, ncols=num_dims, sharex="col", figsize=(10, 10))
        for n in range(num_dims):
            kde_gaussian = gaussian_kde(self.Xc[:, n].flatten(), bw_method="scott").evaluate(self.Xc[:, n])
            ordered_X_n, KDE = zip(*sorted([(x, kde_x) for x, kde_x in zip(self.Xc[:, n].flatten(), kde_gaussian)], key=lambda e: e[0]))

            Subplots[n, n].plot(ordered_X_n, KDE, "-k")
            Subplots[n, n].fill_between(ordered_X_n, KDE, alpha=0.25, color="black")
            Subplots[n, n].set_ylim(bottom=0.0)

            for m in range(num_dims):
                if n != m:
                    Subplots[n, m].axvline(linestyle="-", color="black", alpha=0.5, zorder=0)
                    Subplots[n, m].axhline(linestyle="-", color="black", alpha=0.5, zorder=0)
                    if Colors is not None:
                        Subplots[n, m].scatter(self.Xc[:, m], self.Xc[:, n], c=Colors, ec="black", s=12)
                    else:
                        Subplots[n, m].scatter(self.Xc[:, m], self.Xc[:, n], marker="o", fc="black", ec="black", s=12)

                    # Plot centroids if provided
                    if Centroids is not None:
                        Subplots[n, m].scatter(Centroids[:, m], Centroids[:, n], marker="X", c="red", s=50, label="Centroids")
                        Subplots[n, m].legend(loc="upper right")

            Subplots[num_dims - 1, n].set_xlabel(f"Coordinate {n + 1}")
        Figure.tight_layout()
