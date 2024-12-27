from scripts.subscripts.MiscellanousFunctions import tqdm_joblib
from tqdm import tqdm
from joblib import Parallel, delayed

from datetime import datetime

from tslearn.metrics import dtw
from scipy.signal import correlate
import numpy as np
from pywt import cwt
from scipy.stats import iqr

from collections import Counter
from sklearn.cluster import KMeans

import cv2

def compute_cwt(tec_series: np.ndarray, scales: np.ndarray, dt:float, wavelet:str) -> tuple[np.ndarray]:
    cwt_coeffs, freqs = cwt(data = tec_series, scales = scales,
                             wavelet = wavelet, sampling_period = dt,
                             method = "fft")
    power = np.abs(cwt_coeffs)**2.0
    periods = 1.0/(60.0*freqs)
    return cwt_coeffs, power, periods

def GetProminentContours(power_array:np.ndarray, labels_array:np.ndarray, time_seq:np.ndarray, period_seq:np.ndarray):
    gray_image = 255 * np.copy(labels_array).astype(np.uint8)

    contours = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    reshaped_contours = [contour_array.reshape(contour_array.size//2, 2) for contour_array in contours if contour_array.size//2 > 4]
    del contours
    reshaped_contours_with_maxpower = [(power_array[contour_array[:,1], contour_array[:,0]].max(), contour_array) for contour_array in reshaped_contours]
    del reshaped_contours

    BigContour = max(reshaped_contours_with_maxpower, key = lambda contour: contour[0])[1]

    BigContourX = time_seq[BigContour[:,0]] 
    BigContourY = period_seq[BigContour[:,1]] 
    BigBoxX = (BigContourX.min(), BigContourX.max())
    BigBoxY = (BigContourY.min(), BigContourY.max())

    return (BigBoxX, BigBoxY)

def extract_prominent_series(index, time_seq, vtec_seq, wavelet, dj):
    dt = np.diff(time_seq).mean().total_seconds()
    if dt > 0.0:
        s0 = 2.0 * dt
        J = np.log2(time_seq.size * dt / s0)/dj
        scales = s0 * (2 ** (np.arange(J + 1) * dj))
            
        cwt_coeffs, cwt_power, periods = compute_cwt(vtec_seq, scales, dt, wavelet)
        mstids_periods_index = np.argwhere(periods <= 60.0)[:, 0]
        del cwt_coeffs
        cwt_power = cwt_power[mstids_periods_index, :]
            
        PowerPhase_KMeans = KMeans(n_clusters = 2, init = "k-means++")
        Labels = PowerPhase_KMeans.fit_predict(cwt_power.flatten().reshape(-1,1))
        numLabels = dict(Counter(Labels.tolist()))
        if numLabels[0] <= numLabels[1]:
            Labels[Labels == 0] = 2
            Labels[Labels == 1] = 0
            Labels[Labels == 2] = 1
        LabelsArray = Labels.reshape(cwt_power.shape)
        LabelBiggestContour = GetProminentContours(cwt_power, LabelsArray, time_seq, periods)
            
        BoxX = LabelBiggestContour[0]
        time_box_indexes = (BoxX[0] <= time_seq) & (time_seq <= BoxX[1])
        if time_box_indexes.any():
            time_box_values = time_seq[time_box_indexes]
            dtec_box_values = vtec_seq[time_box_indexes]
            median_dtec_box = np.median(dtec_box_values)
            iqr_dtec_box = iqr(dtec_box_values)
            
            return (index, time_box_values, (dtec_box_values - median_dtec_box)/iqr_dtec_box)
        else:
            return (index, np.array([]))

if __name__ == "__main__":
    wavelet = "cmor1.5-1.5"
    dj = 0.0625
    dist_method = "DTW"

    with open("./data/PTEX_DTEC_series.dat", "+r") as DTECin:
        file_lines = DTECin.readlines()

        total_series = int(file_lines[0])

        dtec_series = []
        for n in tqdm(range(2, 2*total_series + 1, 2)):
            time_n_data = np.array(tuple(map(lambda x: datetime.fromisoformat(x), [time_stamp.replace(" ", "").replace("\n", "") for time_stamp in file_lines[n-1].split(",")])))
            dtec_n_data = np.array(tuple(map(lambda x: float(x), file_lines[n].split(","))))

            prominent_time_dtec_series = extract_prominent_series(n, time_n_data, dtec_n_data, wavelet, dj)

            if len(prominent_time_dtec_series) == 3:
                dtec_series.append(prominent_time_dtec_series[2])

        dtec_series = tuple(dtec_series)
    
    total_series = len(dtec_series)
    print(f"Total of series: {total_series}")
    print(f"\n--Computing {dist_method} between every pair of prominent series--")
    if dist_method == "DTW":
        def compute_dist(index_1, index_2, seq_1, seq_2):
            dtw_1_2 = dtw(seq_1, seq_2)
            return (index_1, index_2, dtw_1_2)

    with tqdm_joblib(tqdm(total = total_series*(total_series - 1)//2)) as progress_bar:
        dtw_results = Parallel(n_jobs=-1)(delayed(compute_dist)(i, j, dtec_series[i], dtec_series[j])
                                          for i in range(total_series-1) for j in range(i+1, total_series))
            
    dissimilarity_matrix = np.zeros((total_series, total_series))
    for result in dtw_results:
        dissimilarity_matrix[result[0], result[1]] = result[2]
        dissimilarity_matrix[result[1], result[0]] = result[2]

    np.savetxt(f"./data/PTEX_{dist_method}_matrix.dat", dissimilarity_matrix, delimiter=",")
