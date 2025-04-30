from scripts.subscripts.MiscellanousFunctions import tqdm_joblib
from scripts.WaveletExtraction import extract_prominent_series

from tqdm import tqdm
from joblib import Parallel, delayed

from datetime import datetime
from tslearn.metrics import dtw
from numpy import array, zeros, savetxt, datetime64

if __name__ == "__main__":
    wavelet = "cmor1.5-1.5"
    dj = 0.0625

    try:
        with open("./data/PTEX_dtec_series.dat", "r") as DTECin:
            file_lines = DTECin.readlines()
    except FileNotFoundError:
        print("Error: El archivo no fue encontrado.")
        exit()

    total_series = int(file_lines[0])
    prn_subseries = tuple([int(file_lines[n-2]) for n in range(3, 3*total_series + 1, 3)])
    with tqdm_joblib(tqdm(total= total_series)) as progress_bar:
        dtec_subseries = Parallel(n_jobs=-1)(delayed(extract_prominent_series)(n,
                                    array([datetime.fromisoformat(x.strip()) for x in file_lines[n-1].split(",")], dtype = datetime64),
                                    array(file_lines[n].split(","), dtype=float),
                                    wavelet, dj) for n in range(3, 3*total_series + 1, 3))

    dtec_subseries = [s for s in sorted(dtec_subseries, key=lambda x: x[0]) if len(s) == 3]
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
