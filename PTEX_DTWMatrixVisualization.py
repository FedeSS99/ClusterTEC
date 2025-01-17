import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

def visualize_distance_matrix(dtw_matrix, downsample_factor=10, cmap='viridis', figsize=(10, 10)):
    """
    Visualize a DTW distance matrix with optional downsampling.
    
    Parameters:
        dtw_matrix (numpy.ndarray): The DTW distance matrix (NxN).
        downsample_factor (int): Factor by which to downsample the matrix for visualization.
        cmap (str): Colormap for the visualization.
        figsize (tuple): Size of the Matplotlib figure.
    """
    # Check if downsampling is needed
    if downsample_factor > 1:
        downsampled_matrix = dtw_matrix[::downsample_factor, ::downsample_factor]
    else:
        downsampled_matrix = dtw_matrix

    # Plot the matrix
    plt.figure(figsize=figsize)
    plt.imshow(
        downsampled_matrix,
        aspect='auto',
        cmap=cmap,
        norm=Normalize(vmin=np.min(downsampled_matrix), vmax=np.max(downsampled_matrix))
    )
    plt.colorbar(label="DTW Distance")
    plt.title("DTW Distance Matrix Visualization")
    plt.xlabel("Index")
    plt.ylabel("Index")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    PTEX_dissim = np.loadtxt("./data/PTEX_DTW_matrix.dat", dtype= np.float64, delimiter = ",")
    visualize_distance_matrix(PTEX_dissim, downsample_factor=1, cmap='jet', figsize=(10, 10))