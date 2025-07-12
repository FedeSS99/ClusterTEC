import contextlib
import joblib
from tqdm import tqdm

import cv2
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold
from matplotlib.pyplot import subplots
import numpy as np

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report progress into a tqdm progress bar.

    Parameters:
    - tqdm_object: tqdm progress bar object.
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def FindDataTendency(input_data: np.ndarray, window_size: int, dt: float) -> int:
    """
    Find the tendency of the input data using the Savitzky-Golay filter.

    Parameters:
    - input_data: Input data array.
    - window_size: Window size for the filter.
    - dt: Time step for the filter.

    Returns:
    - Filtered data representing the tendency.
    """
    #Pre computed values for optimar order and error for the filter
    #are written, expecting to be changed through the next loop
    poly_orders = np.arange(1, 11)
    MSE_array = np.zeros_like(poly_orders, dtype = np.float64)
    # Only explore polynomial orders from 1 to 10
    for p, order in enumerate(poly_orders):
        data_tendency = savgol_filter(input_data, window_size, order, 
                                          delta=dt, mode="interp")

        # Compute MSE of the difference between input data and its tendency
        MSE_array[p] = ((input_data - data_tendency)**2.0).mean()

    tec_tendency = savgol_filter(input_data, window_size, poly_orders[np.argmin(MSE_array)], 
                                 delta=dt, mode = "interp")

    return tec_tendency

def GetHourMinuteSecond(time: np.ndarray) -> tuple[np.ndarray]:
    """
    Convert time in decimal hours to hours, minutes, and seconds.

    Parameters:
    - time: Array of time values in decimal hours.

    Returns:
    - List of tuples containing (hour, minute, second) for each time value.
    """
    integer_hours = time.astype(int)
    minutes_from_fraction = 60 * (time - integer_hours)
    integer_minutes = minutes_from_fraction.astype(int)
    integer_seconds = (60 * (minutes_from_fraction - integer_minutes)).astype(int)

    indexes_minus_24 = np.argwhere(integer_hours == -24)[:, 0]
    integer_hours[indexes_minus_24] = 0
    integer_minutes[indexes_minus_24] = 0
    integer_seconds[indexes_minus_24] = 0

    return [(hour, minute, second) for hour, minute, second in zip(integer_hours, integer_minutes, integer_seconds)]

def GetProminentContours(power_array: np.ndarray, labels_array: np.ndarray, time_seq: np.ndarray, period_seq: np.ndarray):
    """
    Extract the most prominent contour from a labeled array.

    Parameters:
    - power_array: Array of power values.
    - labels_array: Labeled array of regions.
    - time_seq: Time sequence corresponding to the array.
    - period_seq: Period sequence corresponding to the array.

    Returns:
    - Tuple containing bounding box coordinates of the prominent contour.
    """
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
