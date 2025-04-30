from pywt import cwt
import numpy as np
from scipy.stats import iqr
from sklearn.cluster import KMeans
import cv2

def compute_cwt(tec_series: np.ndarray, scales: np.ndarray, dt: float, wavelet: str) -> tuple[np.ndarray]:
    """
    Compute the Continuous Wavelet Transform (CWT) of a time series.

    Parameters:
    - tec_series: Input time series (e.g., TEC data).
    - scales: Array of scales for the wavelet transform.
    - dt: Sampling period of the time series.
    - wavelet: Type of wavelet to use for the transform.

    Returns:
    - cwt_power: Power of the CWT coefficients.
    - periods: Corresponding periods for the scales.
    """
    cwt_coeffs, freqs = cwt(data=tec_series, scales=scales,
                             wavelet=wavelet, sampling_period=dt,
                             method="fft")
    cwt_power = np.abs(cwt_coeffs) ** 2.0
    periods = 1.0 / (60.0 * freqs)  # Convert frequencies to periods in minutes
    return cwt_power, periods

def get_prominent_contours(power_array: np.ndarray, labels_array: np.ndarray, time_seq: np.ndarray, period_seq: np.ndarray):
    """
    Identify the most prominent contour in the wavelet power spectrum.

    Parameters:
    - power_array: 2D array of wavelet power values.
    - labels_array: 2D array of cluster labels.
    - time_seq: Array of time values.
    - period_seq: Array of period values.

    Returns:
    - box_x: Tuple with the min and max time values of the prominent contour.
    - box_y: Tuple with the min and max period values of the prominent contour.
    """
    gray_image = 255 * labels_array.astype(np.uint8)  # Convert labels to grayscale image
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter and reshape contours
    reshaped_contours = [c.reshape(-1, 2) for c in contours if c.size // 2 > 4]
    reshaped_contours_with_maxpower = [(power_array[c[:, 1], c[:, 0]].max(), c) for c in reshaped_contours]

    # Find the contour with the maximum power
    big_contour = max(reshaped_contours_with_maxpower, key=lambda x: x[0])[1]
    big_contour_x = time_seq[big_contour[:, 0]]
    big_contour_y = period_seq[big_contour[:, 1]]

    return (big_contour_x.min(), big_contour_x.max()), (big_contour_y.min(), big_contour_y.max())

def extract_prominent_series(index, time_seq, tec_seq, wavelet, dj):
    """
    Extract the most prominent time series from the wavelet power spectrum.

    Parameters:
    - index: Identifier for the time series.
    - time_seq: Array of time values.
    - vtec_seq: Array of VTEC values.
    - wavelet: Type of wavelet to use for the transform.
    - dj: Scale resolution parameter.

    Returns:
    - index: Identifier for the time series.
    - Extracted time series and normalized values, or an empty array if no prominent series is found.
    """
    dt = np.diff(time_seq).mean().item().total_seconds()  # Compute the average sampling period
    if dt > 0.0:
        s0 = 2.0 * dt  # Smallest scale
        J = np.log2(time_seq.size * dt / s0) / dj  # Number of scales
        scales = s0 * (2 ** (np.arange(J + 1) * dj))  # Generate scales

        # Compute wavelet power and filter by periods
        cwt_power, periods = compute_cwt(tec_seq, scales, dt, wavelet)
        cwt_power = cwt_power[periods <= 60.0]  # Filter periods <= 60 minutes

        # Perform clustering on wavelet power values
        labels = KMeans(n_clusters=2, init="k-means++").fit_predict(cwt_power.flatten().reshape(-1, 1))
        if np.sum(labels == 0) <= np.sum(labels == 1):  # Ensure the larger cluster is labeled as 1
            labels = np.where(labels == 0, 1, 0)

        labels_array = labels.reshape(cwt_power.shape)
        box_x, _ = get_prominent_contours(cwt_power, labels_array, time_seq, periods)

        # Extract the time series within the prominent contour
        time_box_indexes = (box_x[0] <= time_seq) & (time_seq <= box_x[1])
        if time_box_indexes.any():
            dtec_box_values = tec_seq[time_box_indexes]
            return index, time_seq[time_box_indexes], (dtec_box_values - np.median(dtec_box_values)) / iqr(dtec_box_values)
        
    return index, np.array([])  # Return empty array if no prominent series is found
