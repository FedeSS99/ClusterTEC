from scripts.VTECDataReader import VTECDataReader
import matplotlib.pyplot as plt

if __name__ == "__main__":
    PTEX_dir = ["/home/fsamaniego/Documents/FCFM/Proyecto TIDs/Data/CMN Files/PTEX/2018",
                "/home/fsamaniego/Documents/FCFM/Proyecto TIDs/Data/CMN Files/PTEX/2019"]
    
    PTEX_VTEC = VTECDataReader(dirs = PTEX_dir, min_amplitude =  0.18695)

    PTEX_VTEC.read_and_extract_vtec_data()

    Figure, Subplots = plt.subplots(2, 1, figsize = (10, 5), sharex = True)
    Figure.suptitle("PTEX VTEC and dTEC observations")
    N = len(PTEX_VTEC.dtec_sequences)
    for n in range(N):
        Subplots[0].plot(PTEX_VTEC.time_sequences[n], PTEX_VTEC.vtec_sequences[n], "-b", linewidth = 1, alpha = 0.75)
        Subplots[1].plot(PTEX_VTEC.time_sequences[n], PTEX_VTEC.dtec_sequences[n], "-b", linewidth = 1, alpha = 0.75)
    Subplots[0].set_ylabel("VTEC [TECU]")
    Subplots[1].set_ylabel("dTEC [TECU]")
    Subplots[1].set_xlabel("Time [UTC]")
    Figure.tight_layout()
    plt.show()