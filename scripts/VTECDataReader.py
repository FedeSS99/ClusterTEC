from scripts.libraries import *

class VTECDataReader:
    """
    Class to read, process, and extract VTEC data from input files.
    """
    def __init__(self, dirs: list[str], min_amplitude: float = 0.1, window_size: int = 120) -> None:
        """
        Initialize the VTECDataReader with directories and minimum amplitude.

        Parameters:
        - dirs: List of directories containing VTEC data files.
        - min_amplitude: Minimum amplitude threshold for filtering data.
        - window_size: size of the moving window for savitzky golay filter
        """
        self.__dirs: list[str] = dirs
        self.min_amplitude = min_amplitude
        self.window_size = window_size

    def __save_cmn_data(self, filename_dir):
        """
        Read and parse a .Cmn file to extract VTEC data.

        Parameters:
        - filename_dir: Path to the .Cmn file.

        Returns:
        - Dictionary containing date, PRN, time, and VTEC data.
        """
        filename = filename_dir.split("/")[-1].split("-")
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
        """
        Detrend TEC data using a moving window.

        Parameters:
        - time: Array of time values.
        - tec_data: Array of TEC values.
        - window_size: Size of the moving window.

        Returns:
        - Detrended TEC data.
        """
        sampling_time = np.diff(time).mean().astype("timedelta64[s]").item().total_seconds()
        tec_tendency = FindDataTendency(tec_data, window_size, sampling_time)
        dtec = tec_data - tec_tendency

        return dtec
    
    def __split_sequences(self, time_array, vtec_array, median_diff, window_size):
        """
        Split time and VTEC sequences based on time jumps and filter by minimum size.

        Parameters:
        - time_array: Array of time values.
        - vtec_array: Array of VTEC values.
        - median_diff: Median time difference for detecting jumps.
        - window_size: Minimum size for valid sequences.

        Returns:
        - Filtered and split time and VTEC sequences.
        """
        jump_indexes = np.argwhere(np.diff(time_array).astype("timedelta64[s]") > 2 * median_diff)[:, 0] + 1
        time_splits = np.array_split(time_array, jump_indexes)
        vtec_splits = np.array_split(vtec_array, jump_indexes)

        filtered_time = []
        filtered_vtec = []
        for time_seq, vtec_seq in zip(time_splits, vtec_splits):
            if time_seq.size >= 2 * window_size:
                filtered_time.append(time_seq)
                filtered_vtec.append(vtec_seq)

        return filtered_time, filtered_vtec

    def __detrend_sequences(self, time_sequences, vtec_sequences, window_size):
        """
        Detrend VTEC sequences.

        Parameters:
        - time_sequences: List of time sequences.
        - vtec_sequences: List of VTEC sequences.
        - window_size: Size of the moving window.

        Returns:
        - List of detrended VTEC sequences.
        """
        detrended_sequences = []
        for time_seq, vtec_seq in zip(time_sequences, vtec_sequences):
            detrended_sequences.append(self.__detrend_tec_data(time_seq, vtec_seq, window_size))
        return detrended_sequences

    def __separate_and_detrend_data_by_PRN(self) -> None:
        """
        Separate VTEC data by PRN and detrend the time series.
        """
        self.satellite_data_by_PRN = {}
        print("\n--Separate each Cmn file by PRN--")
        for cmn_data in tqdm(self.satellite_data):
            for PRN in set(cmn_data["PRN"]):
                PRN_indexes = np.argwhere(cmn_data["PRN"] == PRN)[:, 0]
                times = GetHourMinuteSecond(cmn_data["time"][PRN_indexes])
                complete_dates = [datetime(cmn_data["date"].year, cmn_data["date"].month, cmn_data["date"].day, *t) for t in times]

                if PRN not in self.satellite_data_by_PRN:
                    self.satellite_data_by_PRN[PRN] = {"time": [], "vtec": []}
                self.satellite_data_by_PRN[PRN]["time"].extend(complete_dates)
                self.satellite_data_by_PRN[PRN]["vtec"] = np.concatenate((self.satellite_data_by_PRN[PRN]["vtec"], cmn_data["vtec"][PRN_indexes]))

        del self.satellite_data

        print("\n--Detrend time series by PRN--")
        for PRN, data in tqdm(self.satellite_data_by_PRN.items()):
            data["time"] = np.array(data["time"], dtype=np.datetime64)
            median_diff = np.median(np.diff(data["time"]).astype("timedelta64[s]"))

            time_splits, vtec_splits = self.__split_sequences(data["time"], data["vtec"], median_diff, self.window_size)
            detrended_splits = self.__detrend_sequences(time_splits, vtec_splits, self.window_size)

            data["time"], data["vtec"], data["dtec"] = time_splits, vtec_splits, detrended_splits

        self.time_sequences = tuple([seq for data in self.satellite_data_by_PRN.values() for seq in data["time"]])
        self.vtec_sequences = tuple([seq for data in self.satellite_data_by_PRN.values() for seq in data["vtec"]])
        self.dtec_sequences = tuple([seq for data in self.satellite_data_by_PRN.values() for seq in data["dtec"]])
        self.prn_sequences = [PRN for PRN, data in self.satellite_data_by_PRN.items() for _ in data["time"]]
        del self.satellite_data_by_PRN

    def __remove_data_by_max_amplitude(self, min_amplitude):
        """
        Remove time series with maximum amplitude below the threshold.

        Parameters:
        - min_amplitude: Minimum amplitude threshold.
        """
        abs_max_vtec_values = np.array(tuple([max(abs(vtec_seq.min()), vtec_seq.max()) for vtec_seq in self.dtec_sequences]))
        indexes_by_quantiles = np.argwhere(abs_max_vtec_values >= min_amplitude)[:,0]
        self.time_sequences, self.vtec_sequences, self.dtec_sequences, self.prn_sequences  = zip(*[time_vtec_PRN for k, time_vtec_PRN in enumerate(zip(self.time_sequences, self.vtec_sequences, self.dtec_sequences, self.prn_sequences)) if k in indexes_by_quantiles])

    def read_and_extract_vtec_data(self):
        """
        Read VTEC data from files, process it, and extract relevant time series.
        """
        self.__list_cmn_files :list[str] = []
        for dir in self.__dirs:
            for month in listdir(dir):
                current_month_dir :str = join(dir, month)
                for cmn_filename in listdir(current_month_dir):
                    self.__list_cmn_files.append(join(current_month_dir, cmn_filename))

        self.satellite_data = []
        print("--Reading Cmn files--")
        print(f"Number of files: {len(self.__list_cmn_files)}")
        for cmn_file in tqdm(self.__list_cmn_files):
            self.satellite_data.append(self.__save_cmn_data(cmn_file))

        self.satellite_data.sort(key= lambda x: x["date"])

        self.satellite_data = tuple(self.satellite_data)

        self.__separate_and_detrend_data_by_PRN()
        if self.min_amplitude != 0.0:
            self.__remove_data_by_max_amplitude(min_amplitude = self.min_amplitude)
