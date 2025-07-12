import os

from tkinter.filedialog import askopenfile
from tkinter import Button, Tk

from matplotlib import use
from matplotlib.pyplot import rcParams
from matplotlib.pyplot import figure, show
import colorcet as cc

from numpy import array, where
from numpy import arange
from collections import Counter

def CreateSignalPlotFigure():
    timeTicks = arange(0, 25, 6)

    #Create main plotting figure to use for every prn number
    MainFigure = figure(1, figsize=(6, 6))
    SubFigureOrigSignalsCMN = MainFigure.add_subplot(111)

    #Setting subfigures labels and limits
    SubFigureOrigSignalsCMN.set_ylabel("VTEC [TECU]")
    SubFigureOrigSignalsCMN.set_xlabel("Universal Time [Hours]")
    SubFigureOrigSignalsCMN.set_xlim(0.0, 24.0)
    SubFigureOrigSignalsCMN.set_xticks([4*k for k in range(7)])

    return MainFigure, SubFigureOrigSignalsCMN


def CMN_SignalPlots(time_vtec, SignalsPlot):
    #Getting prn numbers to plot only the satellites in the given dataset
    prnNumbers = time_vtec.keys()

    #Create CMN_colors dictionary to save rgb colors for each prn Number plot
    CMN_colors = {}
    for prn,rgb in zip(prnNumbers, cc.glasbey_dark[:len(prnNumbers)]):
        CMN_colors[prn] = rgb

    for prn in prnNumbers:
        for interval in time_vtec[prn]:
            SignalsPlot[1].plot(interval['time'], interval['vtec'],
                linewidth=2, color=CMN_colors[prn])
            
    SignalsPlot[0].tight_layout()
    show()

use('TkAgg')
rcParams.update({
    "text.usetex": True,
    "font.size": 16
})

def select_file(window):
    window.cmn_file = askopenfile(title="Select cmn file to read", filetypes=[("Cmn", "*.Cmn")])
    #If the status of window.cmn_file
    #doesn´t change, dont do anything
    if window.cmn_file is not None:
        os.system('cls' if os.name == 'nt' else 'clear')
        cmn_file_path = window.cmn_file.name

        #------------------------------------------------------------------------------------
        with open(cmn_file_path, "+r") as cmn_file:
            cmn_data_lines = cmn_file.readlines()[5:]

            #After reading the whole .cmn file, it is needed to have saved only the
            #data that corresponds to a elevation greater than 30.0 degrees
            elevation = array([float(line.split()[4]) for line in cmn_data_lines])
            elevation_filter = where( elevation>=30.0, True, False)

            time_cmn = array([float(line.split()[1]) for line in cmn_data_lines])[elevation_filter]
            fixed_time_cmn = where(time_cmn>=0.0, time_cmn, time_cmn+abs(time_cmn))
            prn_cmn = array([float(line.split()[2]) for line in cmn_data_lines])[elevation_filter]
            vtec_cmn = array([float(line.split()[8]) for line in cmn_data_lines])[elevation_filter]

        #Then each read line is saved in different arrays on the condition
        #of being from the same prn
        prn_values = [prn_value for prn_value, count in Counter(prn_cmn).items() if count>1]
        cmn_time_vtec_readings = {}
        for prn_value in prn_values:
            prn_filter = where(prn_cmn==prn_value, True, False)
            time_series = fixed_time_cmn[prn_filter]
            vtec_series = vtec_cmn[prn_filter]
            
            # Calculate time differences
            time_diffs = time_series[1:] - time_series[:-1]
            # Find most common difference (rounded to 3 decimal places)
            common_diff = round(float(Counter(round(diff, 3) for diff in time_diffs).most_common(1)[0][0]), 3)
            
            # Find indices where gaps are larger than common difference
            gap_indices = where(time_diffs > common_diff * 1.5)[0]
            
            # Split data into intervals
            intervals = []
            start_idx = 0
            for gap_idx in gap_indices:
                intervals.append({
                    'time': time_series[start_idx:gap_idx + 1],
                    'vtec': vtec_series[start_idx:gap_idx + 1]
                })
                start_idx = gap_idx + 1
            # Add final interval
            intervals.append({
                'time': time_series[start_idx:],
                'vtec': vtec_series[start_idx:]
            })
            
            cmn_time_vtec_readings[str(prn_value)] = intervals

        SignalPlots = CreateSignalPlotFigure()
        CMN_SignalPlots(cmn_time_vtec_readings, SignalPlots)

if __name__=="__main__":
    window = Tk()
    window.geometry('360x100')
    window.resizable(width=False, height=False)
    window.title("Visualize .Cmn file's VTEC data")

    boton = Button(window, text='Select a .Cmn file', command=lambda: select_file(window))
    boton_cerrar = Button(window, text='Close window', command=lambda: window.quit())
    boton.pack()
    boton_cerrar.pack()
    window.mainloop()
