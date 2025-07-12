from scripts.VTECDataReader import VTECDataReader

if __name__ == "__main__":
    PTEX_dir = ["/home/federico/Documents/FCFM/Proyecto TIDs/Data/CMN Files/PTEX/2018",
            "/home/federico/Documents/FCFM/Proyecto TIDs/Data/CMN Files/PTEX/2019"]
    
    PTEX_VTEC = VTECDataReader(dirs = PTEX_dir, min_amplitude = 0.17334, window_size = 240)

    PTEX_VTEC.read_and_extract_vtec_data()

    with open("./data/PTEX_dtec_series.dat", "+w") as PTEXout:
        N = len(PTEX_VTEC.dtec_sequences)

        PTEXout.write(f"{N}\n")

        for n in range(N):
            PTEXout.write(f"{PTEX_VTEC.prn_sequences[n]}\n")
            PTEXout.write(", ".join(list(map(lambda x: str(x), PTEX_VTEC.time_sequences[n]))) + "\n")
            PTEXout.write(", ".join(list(map(lambda x: str(x), PTEX_VTEC.dtec_sequences[n]))) + "\n")