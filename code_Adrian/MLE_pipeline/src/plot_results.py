from pycbc.conversions import mchirp_from_mass1_mass2, mass1_from_mchirp_q, mass2_from_mchirp_q
import matplotlib.pyplot as plt
from generate_data import generate_frequency_domain_signal
from gwpy.timeseries import TimeSeries
from pycbc.types import TimeSeries as PycbcTimeSeries
import pandas as pd


#======================================================================================================
#======================================================================================================
#======================================================================================================


def convert_signal(file_name, epoch):

    """
    Generate a Pycbc TimeSeries and FrequencySeries from the optimized parameters found.

    Parameters
    ----------
    file_name : str
        Name of the file containing the optimized parameters stored with DataFrame.to_csv().
    epoch : float
        Epoch of the signal reconstructed.

    Returns
    -------
    reconstructed_signal_tdomain : Pycbc TimeSeries
    reconstructed_signal_fdomain : Pycbc FrequencySeries 

    """
    
    params_opti_file = pd.read_csv('/home/victor/Internship_Victor_CBC_ET/code_Adrian/MLE_pipeline/results_mini/' + file_name)

    para_opti = [ params_opti_file['tc'].values[0],  params_opti_file['mass1'].values[0],  params_opti_file['mass2'].values[0],
                params_opti_file['distance'].values[0], params_opti_file['ra'].values[0], params_opti_file['dec'].values[0],  params_opti_file['polarization'].values[0], params_opti_file['inclination'].values[0],
                params_opti_file['spin1z'].values[0],  params_opti_file['spin2z'].values[0], params_opti_file['coa_phase'].values[0]]
    list_params = ['tc','mass1','mass2','distance','ra','dec','polarization','inclination','spin1z','spin2z','coa_phase']
    maximized_params = dict(zip(list_params, para_opti))
    cbc_params_stat = {'spin1x': 0., 'spin2x': 0.,  'spin1y': 0., 'spin2y': 0.,
        'eccentricity': 0}
    cbc_params_stat ['approximant'] = 'IMRPhenomXPHM'
    cbc_params_stat ['f_lower'] = 5

    para_opti={**maximized_params, **cbc_params_stat}


    reconstructed_signal_fdomain = generate_frequency_domain_signal(para_opti, epoch=epoch)

    ifos=['E1', 'E2', 'E3']
    reconstructed_signal_tdomain = {}
    for ifo in ifos:
        reconstructed_signal_tdomain[ifo] = reconstructed_signal_fdomain[ifo].to_timeseries() # Just an inverse FFT

    return reconstructed_signal_tdomain, reconstructed_signal_fdomain


#======================================================================================================
#======================================================================================================
#======================================================================================================


def comparison_signals(maximized_params, reconstructed_signal_tdomain, data, residual, ifo, position = None, save_fig = False, reel_params = None, opti_params = False):
    """
    Compare the reconstructed signal and the original data in the time domain (plot).

    Parameters
    ----------
    maximized_params : DataFrame
        DataFrame from read_csv containing the maximized parameters from the maximization process.
    reel_params : list (optional)
        List containing the reel parameters if known.
    opti_params : bool (optional)
        Print the optimized parameters.
    position : str
        "Front" or "Back" depending which part of the signal we want to look.
    reconstructed_signal_tdomain : dict, Pycbc TimeSeries
    data : dict, Pycbc TimeSeries
    residual : dict, Pycbc TimeSeries
    ifo : str
        "E1", "E2" or "E3" for the Einstein Telescope.
    
    Returns
    -------
    Plot of the comparison between both signals : the reconstructed ans the real one.
    """

    mchirp_true = maximized_params['chirp'].values[0]
    q_true = maximized_params['q'].values[0]
    para_opti = [0, maximized_params['tc'].values[0],  mass1_from_mchirp_q(mchirp_true, q_true),  mass2_from_mchirp_q(mchirp_true, q_true),
                maximized_params['distance'].values[0], maximized_params['ra'].values[0], maximized_params['dec'].values[0],  maximized_params['polarization'].values[0], maximized_params['inclination'].values[0],
                maximized_params['spin1z'].values[0],  maximized_params['spin2z'].values[0], maximized_params['coa_phase'].values[0]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Tracer les deux signaux dans le premier axe :
    ax1.plot(reconstructed_signal_tdomain[ifo].get_sample_times(),reconstructed_signal_tdomain[ifo],label= 'Reconstructed signal (' + ifo + ')', zorder = 2)
    if position == "Front" :
        ax1.set_xlim(maximized_params['tc'].values[0] - 3, maximized_params['tc'].values[0] + 0.5)
    elif position == "Back" :
        ax1.set_xlim(data[ifo].end_time - 10, data[ifo].end_time - 6)

    ax1.set_xscale('seconds', epoch=(data[ifo].start_time))
    ax1.plot(data[ifo].get_sample_times(),data[ifo], color='orange',label= 'MDC data (' + ifo + ')',zorder = 1)
    ax1.set_title('Signals comparison for optimized parameters', fontsize = 30)
    ax1.set_ylabel('Amplitude', fontsize = 20)

    if opti_params :
        ax1.text(0.5, 0.2, r'Params_opti : $t_c$ : {}, $m_1$ : {}, $m_2$ : {}, $d_L$ : {}, ra : {}, dec : {}, pola : {}, incl : {}, s1z : {}, s2z : {}, coa_phase : {}.'.format(round(para_opti[1],3),
            round(para_opti[2],4), round(para_opti[3],4), round(para_opti[4],4), round(para_opti[5],4),round(para_opti[6],4), round(para_opti[7],4),round(para_opti[8],4),
            round(para_opti[9],4),round(para_opti[10],4),round(para_opti[11],4)), horizontalalignment='center',
            verticalalignment='center', transform=ax1.transAxes,fontsize=12)

    if reel_params.any() != None :
        ax1.text(0.5, 0.3, r'Params_reels : $t_c$ : {}, $m_1$ : {}, $m_2$ : {}, $d_L$ : {}, ra : {}, dec : {}, pola : {}, incl : {}, s1z : {}, s2z : {}, coa_phase : {}.'.format(round(reel_params[0],3),
            round(reel_params[1],4), round(reel_params[2],4), round(reel_params[3],4), round(reel_params[4],4), round(reel_params[5],4), round(reel_params[6],4), round(reel_params[7],4),
            round(reel_params[8],4), round(reel_params[9],4), round(reel_params[10],4)), horizontalalignment='center',
            verticalalignment='center', transform=ax1.transAxes,fontsize=12)

    ax1.legend(fontsize=20)

    # Tracer la différence dans le second axe :
    ax2.plot(residual[ifo].get_sample_times(), residual[ifo], color='black')
    ax2.set_ylabel('Résidual',fontsize = 20)
    ax2.set_xlabel('Time [s]', fontsize = 20)

    ax1.tick_params(labelsize = 18)
    ax2.tick_params(labelsize = 18)

    plt.tight_layout()
    if save_fig :
        plt.savefig('Full_loc_minim_' + ifo + '_Comparaison_signal')


#======================================================================================================
#======================================================================================================
#======================================================================================================

def pycbc_to_gwpy(Pycbc_TimseSeries):
    ifos = ['E1', 'E2', 'E3']
    Gwpy_TimeSeries = {}
    for ifo in ifos :
        t0 = Pycbc_TimseSeries[ifo].start_time
        Gwpy_TimeSeries[ifo] = TimeSeries(data = Pycbc_TimseSeries[ifo],times=Pycbc_TimseSeries[ifo].get_sample_times(),t0=t0)
    return Gwpy_TimeSeries

def gwpy_to_pycbc(Gwpy_TimeSeries):
    ifos = ['E1', 'E2', 'E3']
    Pycbc_TimeSeries = {}
    for ifo in ifos :
        val = Gwpy_TimeSeries[ifo].value
        delta_t = Gwpy_TimeSeries[ifo].dt.value
        t0 = Gwpy_TimeSeries[ifo].t0.value
        Pycbc_TimeSeries[ifo] = PycbcTimeSeries(val, delta_t=delta_t, epoch = t0)
    return Pycbc_TimeSeries


def comparison_freq(opti_cut,reel_cut,residual,ifo):

    """
    Convert the pycbc TimeSeries into gwpy TimeSeries to ease the calculation of the psds.
    Compare the reconstructed signal and the original data in the frequency domain (plot).

    Parameters
    ----------
    opti_cut : Pycbc TimeSeries
    reel_cut : Pycbc TimeSeries
    residual : Pycbc TimeSeries
    ifo : str
        "E1", "E2" or "E3" for the Einstein Telescope.

    Returns
    -------
    Plot of the comparison between both signals : the reconstructed ans the real one.
    """

    #Conversion en TimesSeries de gwpy pour le calcul du psd avec .psd().
    tsgwpy_opti_cut = pycbc_to_gwpy(opti_cut)
    tsgwpy_reel_cut = pycbc_to_gwpy(reel_cut)
    tsgwpy_res = pycbc_to_gwpy(residual)

    #Calcul des psd
    psd_opti = tsgwpy_opti_cut[ifo].psd()
    psd_reel = tsgwpy_reel_cut[ifo].psd()
    psd_res = psd_reel - psd_opti

    plt.figure()
    ax = plt.gca()
    ax.loglog(psd_opti.frequencies, psd_opti, label= 'Reconstructed signal (' + ifo + ')',zorder=3)
    ax.loglog(psd_reel.frequencies, psd_reel, label= 'MDC data (' + ifo + ')',zorder=2)
    ax.loglog(psd_res.frequencies, psd_res, label= 'Residual',zorder = 1)
    ax.set_ylim(1e-53, 1e-44)
    ax.set_xlim(4, 2048)
    ax.legend()
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD [1/Hz]')


#======================================================================================================
#======================================================================================================
#======================================================================================================


def qtrans_plot(tsgwpy,frange,qrange,fres=0.1,tres = 0.01,colorbar_limits = None):
    """
    Plot q-Transform of gwpy TimeSeries.

    Parameters
    ----------
    tsgwpy : Gwpy TimeSeries
    frange : tuple
        Range of frequencies analysed.
    qrange : tuple
        Range of q analysed.
    colorbar_limits : dict (optional)
        Dictionary containing the limits "inf" and "sup" of the colorbar.
    fres : float (optional)
        Frequency resoluation of the q_tranform.
    tres : float (optional)
        Time resolution of the q_transform.
    """
    qtrans = tsgwpy.q_transform(frange=frange, qrange=qrange, fres=fres, tres=tres)

    plot = qtrans.plot(figsize=[8, 4])

    ax = plot.gca()
    #ax.set_ylim(5, 100)
    #ax.set_xlim(10, 12)
    ax.set_xscale('seconds')
    ax.set_yscale('log')
    ax.grid(True, axis='y', which='both')
    if colorbar_limits != None :
        ax.colorbar(cmap='viridis', label='Normalized energy', clim=(colorbar_limits['inf'], colorbar_limits['sup']))
    else :
        ax.colorbar(cmap='viridis', label='Normalized energy')