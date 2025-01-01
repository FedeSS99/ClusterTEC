from pywt import cwt
import numpy as np
from scipy.stats import iqr
from sklearn.cluster import KMeans
import cv2

def compute_cwt(tec_series: np.ndarray, scales: np.ndarray, dt: float, wavelet: str) -> tuple[np.ndarray]:
    cwt_coeffs, freqs = cwt(data=tec_series, scales=scales,
                             wavelet=wavelet, sampling_period=dt,
                             method="fft")
    power = np.abs(cwt_coeffs) ** 2.0
    periods = 1.0 / (60.0 * freqs)
    return cwt_coeffs, power, periods

def get_prominent_contours(power_array: np.ndarray, labels_array: np.ndarray, time_seq: np.ndarray, period_seq: np.ndarray):
    gray_image = 255 * labels_array.astype(np.uint8)
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    reshaped_contours = [c.reshape(-1, 2) for c in contours if c.size // 2 > 4]
    reshaped_contours_with_maxpower = [(power_array[c[:, 1], c[:, 0]].max(), c) for c in reshaped_contours]

    big_contour = max(reshaped_contours_with_maxpower, key=lambda x: x[0])[1]
    big_contour_x = time_seq[big_contour[:, 0]]
    big_contour_y = period_seq[big_contour[:, 1]]

    return (big_contour_x.min(), big_contour_x.max()), (big_contour_y.min(), big_contour_y.max())

def extract_prominent_series(index, time_seq, vtec_seq, wavelet, dj):
    dt = np.diff(time_seq).mean().total_seconds()
    if dt > 0.0:
        s0 = 2.0 * dt
        J = np.log2(time_seq.size * dt / s0) / dj
        scales = s0 * (2 ** (np.arange(J + 1) * dj))

        _, cwt_power, periods = compute_cwt(vtec_seq, scales, dt, wavelet)
        cwt_power = cwt_power[periods <= 60.0]

        labels = KMeans(n_clusters=2, init="k-means++").fit_predict(cwt_power.flatten().reshape(-1, 1))
        if np.sum(labels == 0) <= np.sum(labels == 1):
            labels = np.where(labels == 0, 1, 0)

        labels_array = labels.reshape(cwt_power.shape)
        box_x, _ = get_prominent_contours(cwt_power, labels_array, time_seq, periods)

        time_box_indexes = (box_x[0] <= time_seq) & (time_seq <= box_x[1])
        if time_box_indexes.any():
            dtec_box_values = vtec_seq[time_box_indexes]
            return index, time_seq[time_box_indexes], (dtec_box_values - np.median(dtec_box_values)) / iqr(dtec_box_values)
        
    return index, np.array([])
