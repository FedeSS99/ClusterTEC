from os import listdir
from os.path import join
from tqdm import tqdm
from datetime import datetime

import numpy as np
from scipy.signal import savgol_filter

from matplotlib import use
import matplotlib.pyplot as plt

use('TkAgg')

def FindOptimalOrder_SGF(input_data:np.ndarray, window_size:int, dt:float) -> int:
    #Pre computed values for optimar order and error for the filter
    #are written, expecting to be changed through the next loop
    OptimalR2 = 0.0
    OptimalOrder = 1
    # Only explore polynomial orders from 1 to 10
    meanData = input_data.mean()
    for order in range(1, 11):
        data_tendency = savgol_filter(input_data, window_size, order, 
                                          delta=dt, mode="interp")

        sr = ((input_data - data_tendency)**2.0).sum()
        st = ((input_data - meanData)**2.0).sum()

        # Computing R2 score
        R2 = 1.0 - (sr/st)
        if OptimalR2 < R2 < 1.0:
            OptimalR2  = R2
            OptimalOrder = order

    return OptimalOrder


def GetHourMinuteSecond(time:np.ndarray) -> tuple[np.ndarray]:
    integer_hours = time.astype(int)
    minutes_from_fraction = 60 * (time - integer_hours)
    integer_minutes = minutes_from_fraction.astype(int)
    integer_seconds = (60 * (minutes_from_fraction - integer_minutes)).astype(int)

    indexes_minus_24 = np.argwhere(integer_hours == -24)[:, 0]
    integer_hours[indexes_minus_24] = 0
    integer_minutes[indexes_minus_24] = 0
    integer_seconds[indexes_minus_24] = 0

    return [(hour, minute, second) for hour, minute, second in zip(integer_hours, integer_minutes, integer_seconds)]


class WaveletRidgeVis:
    def __init__(self, main_dir:str, min_amplitude:float = 0.1, dj:float = 0.5) -> None:
        self.__main_dir :str = main_dir

        self.__wavelet = "cmor1.5-1.5"
        self.__psi_0 = 1.0/np.sqrt(np.pi * float(self.__wavelet.split("-")[0][4:]))
        self.dj = dj

        self.__min_amplitude = min_amplitude

        self.__list_cmn_files :list[str] = []
        for month in listdir(self.__main_dir):
            current_month_dir :str = join(self.__main_dir, month)
            for cmn_filename in listdir(current_month_dir):
                self.__list_cmn_files.append(join(current_month_dir, cmn_filename))

        self.satellite_data = []
        print("--Reading Cmn files--")
        print(f"Number of files: {len(self.__list_cmn_files)}")
        for cmn_file in tqdm(self.__list_cmn_files):
            self.satellite_data.append(self.__save_cmn_data(cmn_file))

        self.satellite_data = tuple(self.satellite_data)

        self.__separate_and_detrend_data_by_PRN()
        self.__remove_data_by_min_amplitude(self.__min_amplitude)

    def __save_cmn_data(self, filename_dir):
        filename = filename_dir.split("/")[-1].split("-")
        station_name = filename[0][:4]
        date_string = "-".join(filename[1:])
        date_string = date_string.replace(".Cmn", "")
        date_datetime = datetime.strptime(date_string, "%Y-%m-%d")
        
        with open(filename_dir, "r") as input_cmn:
            time_data = []
            PRN_data = []
            elevation_data = []
            vtec_data = []

            for num_line, line in enumerate(input_cmn):
                if num_line >= 5:
                    line_split = line.split()

                    time_data.append(float(line_split[1]))
                    PRN_data.append(int(line_split[2]))
                    elevation_data.append(float(line_split[4]))
                    vtec_data.append(float(line_split[8]))

        # Save satellite data in numpy arrays
        time_data = np.array(time_data, dtype = np.double)
        PRN_data = np.array(PRN_data, dtype = int)
        elevation_data = np.array(elevation_data, dtype = np.double)
        vtec_data = np.array(vtec_data, dtype = np.double)
        # Filter the data with elevation less than 30 degrees
        filter_elevation = np.argwhere(elevation_data > 30.0)[:,0]

        time_data = time_data[filter_elevation]
        PRN_data = PRN_data[filter_elevation]
        vtec_data = vtec_data[filter_elevation]

        return {"date":date_datetime, "PRN": PRN_data, "time":time_data, "vtec": vtec_data}

    def __detrend_tec_data(self, time, tec_data, window_size):
        sampling_time = np.diff(time).mean().astype("timedelta64[s]").item().total_seconds()
        Optimal_Savgol_order = FindOptimalOrder_SGF(tec_data, window_size, sampling_time)
        tec_tendency = savgol_filter(tec_data, window_size, Optimal_Savgol_order, 
                                     delta=sampling_time, mode="interp")

        return tec_data - tec_tendency, sampling_time
    
    def __separate_and_detrend_data_by_PRN(self) -> None:
        self.satellite_data_by_PRN = dict()

        print("--Separate each Cmn file by PRN--")
        for cmn_data in tqdm(self.satellite_data):
            AvailablePRN = tuple(set(cmn_data["PRN"]))
            cmn_date = cmn_data["date"]
            year, month, day = cmn_date.year, cmn_date.month, cmn_date.day
            

            for PRN in AvailablePRN:
                PRN_indexes = np.argwhere(cmn_data["PRN"] == PRN)[:, 0]
                hour_minute_second_by_PRN = GetHourMinuteSecond(cmn_data["time"][PRN_indexes])

                complete_date_localtime = [datetime(year, month ,day, hour, minute, second) for hour, minute, second in hour_minute_second_by_PRN]

                if PRN in self.satellite_data_by_PRN.keys():
                    self.satellite_data_by_PRN[PRN]["time"] += complete_date_localtime
                    self.satellite_data_by_PRN[PRN]["vtec"] = np.concatenate((self.satellite_data_by_PRN[PRN]["vtec"], cmn_data["vtec"][PRN_indexes]))
                else:
                    self.satellite_data_by_PRN[PRN] = dict(time = complete_date_localtime,
                                                           vtec = cmn_data["vtec"][PRN_indexes])
                    
                self.satellite_data_by_PRN[PRN]["dtec"] = self.satellite_data_by_PRN[PRN]["vtec"] 
                
                
        del self.satellite_data
        window_size = 240

        print("--Detrend time series by PRN--")
        AllAvailablePRN = tuple(self.satellite_data_by_PRN.keys()) 
        for PRN in tqdm(AllAvailablePRN):
            self.satellite_data_by_PRN[PRN]["time"] = np.array(self.satellite_data_by_PRN[PRN]["time"], dtype = np.datetime64)
            
            diff_times = np.diff(self.satellite_data_by_PRN[PRN]["time"]).astype("timedelta64[s]")
            median_diff_times = np.median(diff_times)
            jump_indexes_by_PRN = np.argwhere(diff_times > 2*median_diff_times)[:, 0] + 1

            self.satellite_data_by_PRN[PRN]["time"] = np.array_split(self.satellite_data_by_PRN[PRN]["time"], jump_indexes_by_PRN)
            self.satellite_data_by_PRN[PRN]["vtec"] = np.array_split(self.satellite_data_by_PRN[PRN]["vtec"], jump_indexes_by_PRN)
            self.satellite_data_by_PRN[PRN]["dtec"] = np.array_split(self.satellite_data_by_PRN[PRN]["dtec"], jump_indexes_by_PRN)

            indexes_to_remove_by_PRN = []
            for k, time_dtec_seq in enumerate(zip(self.satellite_data_by_PRN[PRN]["time"], self.satellite_data_by_PRN[PRN]["dtec"])):
                if time_dtec_seq[0].size >= window_size:
                    detrended_vtec, sampling_time = self.__detrend_tec_data(*time_dtec_seq, window_size)
                    self.satellite_data_by_PRN[PRN]["time"][k] = self.satellite_data_by_PRN[PRN]["time"][k]
                    self.satellite_data_by_PRN[PRN]["dtec"][k] = detrended_vtec
                else:
                    indexes_to_remove_by_PRN.append(k)

            self.satellite_data_by_PRN[PRN]["time"] = [time_seq for k, time_seq in enumerate(self.satellite_data_by_PRN[PRN]["time"]) if k not in indexes_to_remove_by_PRN]
            self.satellite_data_by_PRN[PRN]["vtec"] = [vtec_seq for k, vtec_seq in enumerate(self.satellite_data_by_PRN[PRN]["vtec"]) if k not in indexes_to_remove_by_PRN]
            self.satellite_data_by_PRN[PRN]["dtec"] = [dtec_seq for k, dtec_seq in enumerate(self.satellite_data_by_PRN[PRN]["dtec"]) if k not in indexes_to_remove_by_PRN]

        AllAvailablePRN = tuple(self.satellite_data_by_PRN.keys()) 
        self.__time_sequences = tuple([time_seq for PRN in AllAvailablePRN for time_seq in self.satellite_data_by_PRN[PRN]["time"]])
        self.__vtec_sequences = tuple([time_seq for PRN in AllAvailablePRN for time_seq in self.satellite_data_by_PRN[PRN]["vtec"]])
        self.__dtec_sequences = tuple([vtec_seq for PRN in AllAvailablePRN for vtec_seq in self.satellite_data_by_PRN[PRN]["dtec"]])
        self.__PRN_per_seq = []
        for PRN in AllAvailablePRN:
            self.__PRN_per_seq += len(self.satellite_data_by_PRN[PRN]["time"])*[PRN]
        del self.satellite_data_by_PRN

    def __remove_data_by_min_amplitude(self, min_amplitude):
        abs_max_vtec_values = np.array(tuple([max(abs(dtec_seq.min()), dtec_seq.max()) for dtec_seq in self.__dtec_sequences]))
        indexes_by_quantiles = np.argwhere(abs_max_vtec_values >= min_amplitude)[:,0]
        self.__time_sequences, self.__dtec_sequences, self.__vtec_sequences, self.__PRN_per_seq  = zip(*[time_vtec_PRN for k, time_vtec_PRN in enumerate(zip(self.__time_sequences, self.__dtec_sequences, self.__vtec_sequences, self.__PRN_per_seq)) if k in indexes_by_quantiles])

    def VisualizeSeries(self) -> None:
        total_series = len(self.__time_sequences)

        for index in range(total_series):
            time_seq, dtec_seq, vtec_seq = self.__time_sequences[index], self.__dtec_sequences[index], self.__vtec_sequences[index]
            print()

            figure, subplots = plt.subplot_mosaic([["VTEC"],
                                                   ["DTEC"]],
                                                   sharex = True,
                                                   figsize = (10, 10))
            figure.suptitle(f"{time_seq[0]}-{ time_seq[-1]}")

            subplots["VTEC"].plot(time_seq, vtec_seq, "-r", linewidth = 1.5)
            subplots["DTEC"].plot(time_seq, dtec_seq, "-b", linewidth = 1.5)

            figure.tight_layout()
            plt.show()


if __name__ == "__main__":
    TIDsWavelet = WaveletRidgeVis(main_dir = "/home/federico/Documents/FCFM/Proyecto TIDs/Data/CMN Files/PTEX/2018", dj = 0.0625)
    TIDsWavelet.VisualizeSeries()