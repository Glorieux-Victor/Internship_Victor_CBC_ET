import pickle
import sys
import pandas as pd
import argparse
sys.path.append('../src')
sys.path.append('/home/victor-glorieux/Internship_Victor_CBC_ET/code_Adrian/MLE_pipeline/src')
from plot_results import convert_signal, comparison_signals, comparison_freq, qtrans_plot, gwpy_to_pycbc, pycbc_to_gwpy
sys.path.append('/home/victor-glorieux/Internship_Victor_CBC_ET')
from pycbc.types import TimeSeries as PycbcTimeSeries
from get_data import read_MDC_data
import glob

max_list = {'mass1': [], 'mass2': [], 'spin1z': [], 'spin2z': [], 'distance': [], 'polarization': [], 'inclination': [], 'tc': [], 'coa_phase': [], 'ra': [], 'dec': [], 'spin1x': [], 'spin2x': [], 'spin1y': [], 'spin2y': [], 'eccentricity': [], 'approximant': [], 'f_lower': []}

list_ = glob.glob("/home/victor-glorieux/MLE-pipeline/results/**/*.pkl", recursive=True)

for i in list_ :
    with open(i, 'rb') as f:
        model= pickle.load(f)
        max = model.maximized_params
        for key, value in max.items() :
            max_list[key].append(value)



print(max_list)