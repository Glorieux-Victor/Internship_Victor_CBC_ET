import numpy as np
from scipy.interpolate import interp1d
from gwpy.timeseries import TimeSeries
import glob
from pycbc.types import FrequencySeries

path_to_MDC_data = {'E1': '/home/shared/et-mdc-frame-files/mdc1/v2/data/E1/',
                    'E2': '/home/shared/et-mdc-frame-files/mdc1/v2/data/E2/',
                    'E3': '/home/shared/et-mdc-frame-files/mdc1/v2/data/E3/'
                    }
psd_file = '../input/ET10km_columns.txt'

def read_MDC_data(t_start, t_end, ifos=['E1', 'E2', 'E3']):
    """
    Read ET MDC data.

    Parameters
    ----------
    t_start : float
        GPS start time.
    t_end : float
        GPS end time.
    ifos : list, optional
        List of detectors to read from. Default is E1, E2, E3.

    Returns
    -------
    data : dict
        Dictionary containing time series data for each requested detector sampled at 4096 Hz.
    """
    
    data = {}
    for ifo in ifos:
        channel = ifo + ':STRAIN'
        files = glob.glob(path_to_MDC_data[ifo] + '*.gwf')
        data[ifo] = TimeSeries.read(files, start=t_start, end=t_end, channel=channel)
        data[ifo] = data[ifo].resample(4096)

    return data

def get_psd_frequency_series(freq_array, delta_f):

    psd = np.loadtxt(psd_file, usecols=(0,3))
    psd_of_f = interp1d(psd[:,0], psd[:,1], bounds_error=False, fill_value=(psd[0,1], psd[-1,1]))
    psd_pycbc = psd_of_f(freq_array)
    psd_pycbc = FrequencySeries(psd_pycbc, delta_f=delta_f)

    return psd_pycbc

def convert_to_frequency_series_with_psd(time_series, return_psd=True):
    """
    Convert a time series to a frequency series by taking the inverse FFT.
    If return_psd is True, also return the corresponding PSD as a frequency series with the same format.

    Parameters
    ----------
    time_series : gwpy.timeseries.TimeSeries
        Time series to convert.
    return_psd : bool, optional.
        Return a PSD frequency series. Default is True.
    
    Returns
    -------
    fft_pycbc : pycbc.types.FrequencySeries
        Inverse FFT of the original time series.
    psd_pycbc : pycbc.types.FrequencySeries
        PSD frequency series.

    Note
    ----
    For now the PSD is set from the file "ET10km_columns.txt. At a later stage we may try to estimate the PSD directly from the data.
    """
    
    tsd_pycbc = time_series.to_pycbc()
    fft_pycbc = tsd_pycbc.to_frequencyseries()
    psd_pycbc = get_psd_frequency_series(fft_pycbc.get_sample_frequencies(), fft_pycbc.get_delta_f())

    return fft_pycbc, psd_pycbc

def convert_data_dict_to_frequency_series_dict(data_dict):
    """
    Generate dictionaries contaning data and PSD in the correct format for the GaussianNoise model.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing gwpy TimeSeries from each detector.
    Returns
    -------
    data_freq : dict
        Dictionary containing pycbc FrequencySeries from each detector.
    psds : dict
        Dictionary containing psd for each detector.
    """
    data_freq = {}
    psds = {}
    for ifo in data_dict.keys():
        fft, psd = convert_to_frequency_series_with_psd(data_dict[ifo], return_psd=True)
        data_freq[ifo] = fft
        psds[ifo] = psd

    return data_freq, psds

