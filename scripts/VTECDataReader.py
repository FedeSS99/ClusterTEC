from scripts.libraries import *

class VTECDataReader:
    def __init__(self, dirs:list[str], min_amplitude:float = 0.1) -> None:
        # Initialize the VTECDataReader with directories and minimum amplitude
        self.__dirs :list[str] = dirs
        self.min_amplitude = min_amplitude

    def __save_cmn_data(self, filename_dir):
        # Extract the date from the filename
        filename = filename_dir.split("/")[-1].split("-")
        date_string = "-".join(filename[1:])
        date_string = date_string.replace(".Cmn", "")
        date_datetime = datetime.strptime(date_string, "%Y-%m-%d")
        
        # Read and parse the Cmn file to extract VTEC data
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

        # Convert lists to numpy arrays
        time_data = np.array(time_data, dtype = np.double)
        PRN_data = np.array(PRN_data, dtype = int)
        elevation_data = np.array(elevation_data, dtype = np.double)
        vtec_data = np.array(vtec_data, dtype = np.double)
        
        # Filter data based on elevation threshold
        filter_elevation = np.argwhere(elevation_data > 30.0)[:,0]
        time_data = time_data[filter_elevation]
        PRN_data = PRN_data[filter_elevation]
        vtec_data = vtec_data[filter_elevation]

        return {"date":date_datetime, "PRN": PRN_data, "time":time_data, "vtec": vtec_data}

    def __detrend_tec_data(self, time, tec_data, window_size):
        # Compute the detrended TEC data using Savitzky-Golay Filter
        sampling_time = np.diff(time).mean().astype("timedelta64[s]").item().total_seconds()
        tec_tendency = FindDataTendency(tec_data, window_size, sampling_time)

        return tec_data - tec_tendency
    
    def __separate_and_detrend_data_by_PRN(self) -> None:
        # Separate and detrend VTEC data for each PRN
        self.satellite_data_by_PRN = dict()

        print("\n--Separate each Cmn file by PRN--")
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
        del self.satellite_data
        window_size = 240

        print("\n--Detrend time series by PRN--")
        AllAvailablePRN = tuple(self.satellite_data_by_PRN.keys()) 
        for PRN in tqdm(AllAvailablePRN):
            self.satellite_data_by_PRN[PRN]["time"] = np.array(self.satellite_data_by_PRN[PRN]["time"], dtype = np.datetime64)
            
            diff_times = np.diff(self.satellite_data_by_PRN[PRN]["time"]).astype("timedelta64[s]")
            median_diff_times = np.median(diff_times)
            jump_indexes_by_PRN = np.argwhere(diff_times > 2*median_diff_times)[:, 0] + 1

            self.satellite_data_by_PRN[PRN]["time"] = np.array_split(self.satellite_data_by_PRN[PRN]["time"], jump_indexes_by_PRN)
            self.satellite_data_by_PRN[PRN]["vtec"] = np.array_split(self.satellite_data_by_PRN[PRN]["vtec"], jump_indexes_by_PRN)

            indexes_to_remove_by_PRN = []
            for k, time_vtec_seq in enumerate(zip(self.satellite_data_by_PRN[PRN]["time"], self.satellite_data_by_PRN[PRN]["vtec"])):
                if time_vtec_seq[0].size >= 2.0 * window_size:
                    detrended_vtec = self.__detrend_tec_data(*time_vtec_seq, window_size)
                    self.satellite_data_by_PRN[PRN]["vtec"][k] = detrended_vtec
                else:
                    indexes_to_remove_by_PRN.append(k)

            self.satellite_data_by_PRN[PRN]["time"] = [time_seq for k, time_seq in enumerate(self.satellite_data_by_PRN[PRN]["time"]) if k not in indexes_to_remove_by_PRN]
            self.satellite_data_by_PRN[PRN]["vtec"] = [vtec_seq for k, vtec_seq in enumerate(self.satellite_data_by_PRN[PRN]["vtec"]) if k not in indexes_to_remove_by_PRN]
