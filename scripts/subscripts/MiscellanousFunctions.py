import contextlib
import joblib
import cv2
from scipy.signal import savgol_filter

from numpy import ndarray, argwhere, copy, uint8

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
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

def FindDataTendency(input_data:ndarray, window_size:int, dt:float) -> int:
    #Pre computed values for optimar order and error for the filter
    #are written, expecting to be changed through the next loop
    OptimalR2 = 0.0
    OptimalOrder = 1
    # Only explore polynomial orders from 1 to 10
    for order in range(1, 11):
        data_tendency = savgol_filter(input_data, window_size, order, 
                                          delta=dt, mode="interp")

        meanData = input_data.mean()
        sr = ((input_data - data_tendency)**2.0).sum()
        st = ((input_data - meanData)**2.0).sum()

        # Computing R2 score
        R2 = 1.0 - (sr/st)
        if OptimalR2 < R2 < 1.0:
            OptimalR2  = R2
            OptimalOrder = order

    tec_tendency = savgol_filter(input_data, window_size, OptimalOrder, 
                                 delta=dt, mode="interp")

    return tec_tendency

def GetHourMinuteSecond(time:ndarray) -> tuple[ndarray]:
    integer_hours = time.astype(int)
    minutes_from_fraction = 60 * (time - integer_hours)
    integer_minutes = minutes_from_fraction.astype(int)
    integer_seconds = (60 * (minutes_from_fraction - integer_minutes)).astype(int)

    indexes_minus_24 = argwhere(integer_hours == -24)[:, 0]
    integer_hours[indexes_minus_24] = 0
    integer_minutes[indexes_minus_24] = 0
    integer_seconds[indexes_minus_24] = 0

    return [(hour, minute, second) for hour, minute, second in zip(integer_hours, integer_minutes, integer_seconds)]

def GetProminentContours(power_array:ndarray, labels_array:ndarray, time_seq:ndarray, period_seq:ndarray):
    gray_image = 255 * copy(labels_array).astype(uint8)

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