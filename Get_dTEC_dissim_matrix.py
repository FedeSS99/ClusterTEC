from scripts.VTECDataReader import VTECDataReader
from scripts.subscripts.MiscellanousFunctions import tqdm_joblib
from scripts.WaveletExtraction import extract_prominent_series

from tqdm import tqdm
from joblib import Parallel, delayed

from tslearn.metrics import dtw
from numpy import zeros, savetxt

if __name__ == "__main__":
    PTEX_dir = ["/home/federico/Documents/FCFM/Proyecto TIDs/Data/CMN Files/PTEX/2018",
        "/home/federico/Documents/FCFM/Proyecto TIDs/Data/CMN Files/PTEX/2019"]
    wavelet = "cmor1.5-1.5"
    dj = 0.0625

    PTEX_VTEC = VTECDataReader(dirs = PTEX_dir, min_amplitude = 0.17334, window_size = 240)
    PTEX_VTEC.read_and_extract_vtec_data()
    total_series = len(PTEX_VTEC.dtec_sequences)

    print("--Extracting prominent dTEC subseries--")
    with tqdm_joblib(tqdm(total= total_series)) as progress_bar:
        dtec_subseries = Parallel(n_jobs=-1)(delayed(extract_prominent_series)(n,
                                    PTEX_VTEC.time_sequences[n],
                                    PTEX_VTEC.dtec_sequences[n],
                                    wavelet, dj) for n in range(total_series))

    dtec_subseries = [s for s in sorted(dtec_subseries, key=lambda x: x[0]) if len(s) == 3]
    prn_subseries = tuple(PTEX_VTEC.prn_sequences[s[0]] for s in dtec_subseries)
    time_subseries = tuple(s[1] for s in dtec_subseries)
    dtec_subseries = tuple(s[2] for s in dtec_subseries)
    total_series = len(dtec_subseries)
    print(f"Total of series: {total_series}")
    print("-- Saving dTEC subseries --")
    with open("./data/PTEX_dtec_subseries.dat", "+w") as PTEX_sub_out:
        PTEX_sub_out.write(f"{total_series}\n")
        for n in tqdm(range(total_series)):
            PTEX_sub_out.write(f"{prn_subseries[n]}\n")
            PTEX_sub_out.write(", ".join(list(map(lambda x: str(x), time_subseries[n]))) + "\n")
            PTEX_sub_out.write(", ".join(list(map(lambda x: str(x), dtec_subseries[n]))) + "\n")

    print(f"--Computing DTW between every pair of prominent series--")
    def compute_dist(index_1, index_2, seq_1, seq_2):
        return index_1, index_2, dtw(seq_1, seq_2)
    with tqdm_joblib(tqdm(total=total_series * (total_series - 1) // 2)) as progress_bar:
        dtw_results = Parallel(n_jobs=-1)(delayed(compute_dist)(i, j, dtec_subseries[i], dtec_subseries[j])
                                          for i in range(total_series - 1) for j in range(i + 1, total_series))

    dissimilarity_matrix = zeros((total_series, total_series))
    for i, j, dist in dtw_results:
        dissimilarity_matrix[i, j] = dist
        dissimilarity_matrix[j, i] = dist

    savetxt(f"./data/PTEX_DTW_matrix.dat", dissimilarity_matrix, delimiter=",")
