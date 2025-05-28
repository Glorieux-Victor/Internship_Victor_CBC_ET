import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
from gwpy.signal import filter_design
import matplotlib.pyplot as plt
from gwpy.plot import Plot
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from pycbc import psd as pypsd
from pycbc.inference.models import GaussianNoise
from pycbc.noise.gaussian import frequency_noise_from_psd
from pycbc.waveform.generator import (FDomainDetFrameGenerator,FDomainCBCGenerator)
from pycbc.psd import EinsteinTelescopeP1600143
from pycbc.conversions import mchirp_from_mass1_mass2, q_from_mass1_mass2, mass1_from_mchirp_q, mass2_from_mchirp_q

#====================================================
#fonctions : extraction_data, get_refs, filtre_func, find_ref
#====================================================

#======================================================================================================
#======================================================================================================
#======================================================================================================

def extraction_data(path,number,final,channel,dossier_save,save):
    def name_data(number):
        return  "E-E1_STRAIN_DATA-" + number + "-2048.gwf"
    if number == final:
        data = TimeSeries.read(path+name_data(number),channel)
    else :
        data = TimeSeries.read([path+name_data(str(int(number)+i*2048)) for i in range(int(((int(final)-int(number))/2048)+1))],channel)
    #print(data)
    if save :
        plot = Plot(data, figsize=(12, 6))
        plt.savefig(dossier_save+"OG_2")
    return data

cols = ["col1","col2","col3"]
ET = pd.read_csv("ET_data.txt",sep = '  ',engine='python')

#======================================================================================================
#======================================================================================================
#======================================================================================================

def get_refs(t_start,t_stop):

    t_lim = 2660352
    if t_start < 0 :
        print('Attention : temps initial négatif pas permis.')
    if t_stop > t_lim :
        print('Attention : temps final trop grand.')


    t_start += 1000000000
    t_stop += 1000000000

    ref_start = (t_start - 1000000000)//2048
    ref_stop = (t_stop - 1000000000)//2048

    return ref_start, ref_stop

#======================================================================================================
#======================================================================================================
#======================================================================================================

def filtre_func(data):
    bp = filter_design.bandpass(4, 200, data.sample_rate)
    #CBC 4 and 1000
    #Les notch correspondent aux fréquences du réseau électrique aux US (50Hz en Europe), pas utile dans notre étude car fréq ponctuelles et pas prises en compte dans la génération.
    notches = [filter_design.notch(line, data.sample_rate) for line in (60,120,180)]
    zpk = filter_design.concatenate_zpks(bp, *notches)
    hfilt = data.filter(zpk, filtfilt=True)

    hdata = data.crop(*data.span.contract(1))
    hfilt = hfilt.crop(*hfilt.span.contract(1))

    return hfilt

#======================================================================================================
#======================================================================================================
#======================================================================================================

def find_ref(ref_start,ref_stop):
    init_file = []
    final_file = []

    for int_i,i in enumerate(cols) : 
        for int_j,j in enumerate(ET[i]):
            if int_j + int_i*len(ET[i]) == ref_start: #On se repère avec les indices.
                init_file.append(j[17:27])
            if int_j + int_i*len(ET[i]) == ref_stop:
                final_file.append(j[17:27])

    return init_file, final_file

