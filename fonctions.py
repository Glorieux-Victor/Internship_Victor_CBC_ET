from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from matplotlib import pyplot as plt
from gwpy.plot import Plot
from gwpy.signal import filter_design
from IPython.display import clear_output
import pandas as pd
import numpy as np
import copy
import pickle
from scipy.optimize import curve_fit
import pycbc.psd.analytical as detector_psd
from pycbc.detector import Detector
from pycbc.conversions import mchirp_from_mass1_mass2, q_from_mass1_mass2, mass1_from_mchirp_q, mass2_from_mchirp_q
import sys
from pycbc.types import timeseries as pycbcTimeSerie
from plot_results import convert_signal, comparison_signals, comparison_freq, qtrans_plot, gwpy_to_pycbc, pycbc_to_gwpy
from params_calculation import spinSz_from_sz_mass, spinAz_from_sz_mass, s1z_from_spinSz_spinAz, s2z_from_spinSz_spinAz

sys.path.append('/home/victor-glorieux/Internship_Victor_CBC_ET/code_Adrian/MLE_pipeline/src')
from get_data import read_MDC_data
from likelihood import subtract_signal

# =========================================
#instruments_PSD, antenna_factors, comparison_signals_params, likelihood_visualisation, extract_best_SNR
# =========================================


def puissance_seglen(seglen):
    k=seglen
    q = 0
    while k > 1 :
        k=k//2
        q +=1
    if seglen == 2**q :
        return 2**q
    else :
        ecart_inf = seglen - 2**q
        ecart_sup = 2**(q+1) - seglen
        if ecart_inf > ecart_sup :
            return 2**(q+1)
        else :
            return 2**q
        
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

#======================================================================================================
#======================================================================================================
#======================================================================================================

def PSD_func(data,dossier_save,save):
    PSD = data.psd(20,5)
    plot = Plot(PSD, figsize=(12, 6))
    ax = plot.gca()
    ax.set_xscale('log')
    ax.set_xlim(xmin=10,xmax=500)
    ax.set_ylim(ymax=0.3e-47)
    if save :
        plt.savefig(dossier_save+"OG_2_PSD")

#======================================================================================================
#======================================================================================================
#======================================================================================================

def ASD_func(data,dossier_save,save):
    ASD = data.asd(4)
    plot = Plot(ASD, figsize=(12, 6))
    ax = plot.gca()
    ax.set_xscale('log')
    if save :
        plt.savefig(dossier_save+"OG_2_ASDbis")

#======================================================================================================
#======================================================================================================
#======================================================================================================

def spectro_func(path,number,final,channel,t0_spectro,tf_spectro,dossier_save,save):
    data = extraction_data(path,number,final,channel,dossier_save,save=False)
    spectro = data.spectrogram(500, fftlength=500)
    if save :
        plot = spectro.plot(figsize=[8, 4])
        ax = plot.gca()
        ax.set_ylim(2, 70)
        ax.set_xlim(t0_spectro, tf_spectro)
        ax.set_xscale('seconds')
        ax.set_yscale('log')
        ax.grid(True, axis='y', which='both')
        ax.colorbar(cmap='viridis', label='Normalized energy')
        plt.savefig(dossier_save+"OG_2_spectro")
    plt.close(fig='all')
    return spectro

#======================================================================================================
#======================================================================================================
#======================================================================================================

def multi_spectro_func_chat(path, number, final, channel, t0_spectro, tf_spectro, dossier_save):
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=[20, 18])
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            idx = i * 2 + j
            data = extraction_data(path, number[idx], final[idx], channel, dossier_save,save=False)
            spectro = data.spectrogram(100, fftlength=100)
            # Extraire les données nécessaires pour pcolormesh
            times = spectro.times.value
            freqs = spectro.frequencies.value
            power = spectro.value.T  # Transposer pour correspondre à (Y, X)
            # Tracer le spectrogramme
            pcm = ax.pcolormesh(times, freqs, power, shading='auto', cmap='viridis')
            ax.set_ylim(2, 70)
            ax.set_xlim(t0_spectro[idx], tf_spectro[idx])
            ax.set_yscale('log')
            ax.set_xlabel("Temps (s)")
            ax.set_ylabel("Fréquence (Hz)")
            ax.set_title(f"Spectrogramme {number[idx]}")
            ax.grid(True, axis='y', which='both')
            # Ajouter une barre de couleur pour chaque subplot
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(pcm, label="Énergie normalisée",cax=cbar_ax)
    plt.tight_layout()
    plt.savefig(f"{dossier_save}spectrograms_bestSNR.png")
    plt.close(fig)

#======================================================================================================
#======================================================================================================
#======================================================================================================

def multi_spectro_func(path,number,final,channel,t0_spectro,tf_spectro,dossier_save):
    for i,name in enumerate(number) :
        data = extraction_data(path,number[i],final[i],channel,dossier_save,save=False)
        spectro = data.spectrogram(3, fftlength=3)
        plot = spectro.plot(figsize=[8, 4])
        ax = plot.gca()
        ax.set_ylim(2, 70)
        ax.set_xlim(t0_spectro[i], tf_spectro[i])
        ax.set_xscale('seconds')
        ax.set_yscale('log')
        ax.grid(True, axis='y', which='both')
        ax.colorbar(cmap='viridis', label='Normalized energy')
        plt.savefig(dossier_save+name+"_spectre")

#======================================================================================================
#======================================================================================================
#======================================================================================================

def filtre_func(path,number,final,channel,dossier_save,save):
    data = extraction_data(path,number,final,channel,dossier_save,save=False)
    bp = filter_design.bandpass(4, 200, data.sample_rate)
    #CBC 4 and 1000
    #Les notch correspondent aux fréquences du réseau électrique aux US (50Hz en Europe), pas utile dans notre étude car fréq ponctuelles et pas prises en compte dans la génération.
    notches = [filter_design.notch(line, data.sample_rate) for line in (60,120,180)]
    zpk = filter_design.concatenate_zpks(bp, *notches)
    hfilt = data.filter(zpk, filtfilt=True)

    hdata = data.crop(*data.span.contract(1))
    hfilt = hfilt.crop(*hfilt.span.contract(1))

    if save:
        plot = Plot(hdata, hfilt, figsize=[12, 6], separate=True, sharex=True,
                    color='gwpy:ligo-hanford')
        ax1, ax2 = plot.axes
        ax1.set_title('MDC-ET strain data ' + number + '-2048')
        ax1.text(1.0, 1.01, 'Unfiltered data', transform=ax1.transAxes, ha='right')
        ax1.set_ylabel('Amplitude [strain]', y=-0.2)
        ax2.set_ylabel('')
        ax2.text(1.0, 1.01, r'1-1000\,Hz bandpass, notches at 60, 120, 180 Hz',
                transform=ax2.transAxes, ha='right')
        plt.savefig(dossier_save+"OG_2_comparaison_filtre")

    return hdata, hfilt

#======================================================================================================
#======================================================================================================
#======================================================================================================

def signal_GW(hfilt,number,dossier_save,t_start,t_stop,save):
    plot = hfilt.plot(color='gwpy:ligo-hanford')
    ax = plot.gca()
    ax.set_title('MDC-ET strain data ' + number + '-2048')
    ax.set_ylabel('Amplitude [strain]')
    ax.set_xlim(t_start, t_stop)
    ax.set_xscale('seconds', epoch=t_start)
    if save:
        plt.savefig(dossier_save+"OG_2_filtre")
    return plot

#======================================================================================================
#======================================================================================================
#======================================================================================================


def extraction_temps(indexes,type,source = 'Internship_Victor_CBC_ET',print_ = False):

    """
    Extract time parameters (tc, t0) of the GW signals using the file "list_mdc1_v2.txt".
    Indices are classified by decreasing SNR. The index 0 is the highest SNR for the type considered.
    In addition, return the time references to extract the signals from the MDC data on the IJCLab server.

    Parameters
    ----------
    indexes : list
        List containing the indices of the signals we want to analyse.
    type : int
        1 : NS/NS, 2 : BH/NS, 3 : BH/BH coalescence (Neutron Star and Black Hole).
    source: str
        Environment where the code is executed, "local" or "IJCLab_server".

    Returns
    -------
    init, stop : start and stop file references.
    t0_list, tc_list
    interval
    params_list
    """

    #Extraction du fichier qui contient le nom des observations du MDC présentes sur le serveur de l'IJCLab
    cols = ["col1","col2","col3"]

    if source == "IJCLab_server" :
        ET = pd.read_csv("/home/victor-glorieux/Internship_Victor_CBC_ET/code_Adrian/MLE_pipeline/data/loudest_BBH/ET_data.txt",sep = '  ',engine='python')
    else :
        ET = pd.read_csv('/home/victor/Internship_Victor_CBC_ET/code_Adrian/MLE_pipeline/data/loudest_BBH/ET_data.txt',sep = '  ',engine='python')

    #Ce code permet d'extraire les refs, t0 et tc des événements d'onde GW que nous voulons regarder.
    #Nous listons les indices des évenements dans "indexes" et nous regardons en priorité les événements avec le meilleur SNR.
    #Une "ref" correspond à l'indice du fichier "ET_data" qui nous permet d'y trouver le nom du fichier contenant les données que nous souhaitons regarder.
    #Une "ref_sup" est l'indice de fin de nos événement.
    def temps_ref(indexes,source=source):
        if source == 'IJCLab_server' :
            ET_params = pd.read_csv("/home/victor-glorieux/Internship_Victor_CBC_ET/code_Adrian/MLE_pipeline/data/loudest_BBH/list_mdc1_v2.txt",sep = ' ',engine='python', index_col = False)
        else :
            ET_params = pd.read_csv("/home/victor/Internship_Victor_CBC_ET/code_Adrian/MLE_pipeline/data/loudest_BBH/list_mdc1_v2.txt",sep = ' ',engine='python', index_col = False)

        ET_params = ET_params.sort_values('snr',ascending=False) #Sélectionne les events avec le meilleur SNR pour les indices les plus faibles.
        ET_params = ET_params[ET_params['type'] == type] #Sélectionne un type particulier d'événements.
        #print(ET_params)
        ref_list=[]
        t0_list=[]
        tc_list=[]
        ref_sup=[]
        for i,ind in enumerate(indexes) :
            #print(ET_params.iloc[ind])
            t0 = ET_params.iloc[ind].t0
            #print(t0)
            tc = ET_params.iloc[ind].tc
            delta_t = tc-t0
            ref = (t0 - 1000000000)//2048 #la référence en temps
            ref_s = ((tc - 1000000000)//2048)
            ref_list.append(ref)
            ref_sup.append(ref_s)
            t0_list.append(t0)
            tc_list.append(tc)

            params_list = ET_params.iloc[ind]
        return ref_list,t0_list,tc_list, ref_sup, params_list

    interval = [] #Contient True : signal sur un seul fichier, ou False : signal sur plusieurs fichiers.
    init=[] #Contient le nom des fichiers
    final=[]

    def find_ref(ref,ref_sup,t0,tc):
        for int_i,i in enumerate(cols) : 
            for int_j,j in enumerate(ET[i]):
                if int_j + int_i*len(ET[i]) == ref: #On se repère avec les indices.
                    init.append(j[17:27])
                if int_j + int_i*len(ET[i]) == ref_sup:
                    final.append(j[17:27])
                    #print(j[17:27])
                if int_j + int_i*len(ET[i]) - 1  == ref:
                    #print(j[17:27])
                    if float(tc) < float(j[17:27]):
                        interval.append(True)
                        #print('Signal compris dans l\'intervale de temps.')
                    else :
                        interval.append(False)
                        #print('Il faut prendre un plus grand intervale.')

    ref_list,t0_list,tc_list,ref_sup,params_list = temps_ref(indexes)
    for i in range(len(ref_list)):
        if print_ ==True:
            print('t0 :', t0_list[i])
            print('tc :', tc_list[i])
        find_ref(ref_list[i],ref_sup[i],t0_list[i],tc_list[i])

    t0_list = [float(t0_list[i]) for i in range(len(t0_list))]
    tc_list = [float(tc_list[i]) for i in range(len(tc_list))]
    return init, final, t0_list, tc_list, interval,params_list

#======================================================================================================
#======================================================================================================
#======================================================================================================

#========================================================================
#Permet de plot une double figure contenant le signal en temporel et le spectrogram, aux bonnes échelles de temps récupérées sur le fichier "list_mdc1.txt".
#Il se base sur la fonction "extraction_temps" pour récupérer les fichiers d'intérêt.
def single_plot_spec_GW(path,channel,dossier_save,save,i,ind,type):
    indexes = np.arange(ind)
    GW_init, GW_final, t0_list, tc_list, interval = extraction_temps(indexes,type,print_=False)

    spectro = spectro_func(path,GW_init[i],GW_final[i],channel,t0_list[i],tc_list[i],dossier_save,save=False)
    print("Spectro done")
    hdata, hfilt = filtre_func(path,GW_init[i],GW_final[i],channel,dossier_save,save=False)
    print("Filtre done")

    #t_stop = np.array([tc_list[i] -2.7 for i in range(len(tc_list))])
    #t_start = t_stop - 2
    t_stop = tc_list
    t_start = t0_list
    #GW_signal = fct.signal_GW(hfilt,GW_init[i],dossier_save,t_start[i],t_stop[i],save=False)
    
    
    fig, axs = plt.subplots(2, 1, figsize=(20, 12), sharex=False)

    # Spectrogramme
    times = spectro.times.value
    freqs = spectro.frequencies.value
    power = spectro.value.T
    pcm = axs[0].pcolormesh(times, freqs, power, shading='auto', cmap='viridis')
    axs[0].set_ylim(2, 70)
    axs[0].set_xlim(t0_list[i], tc_list[i])
    axs[0].set_xscale('seconds')
    axs[0].set_yscale('log')
    axs[0].set_ylabel("Fréquence (Hz)")
    axs[0].grid(True, axis='y', which='both')
    fig.colorbar(pcm, ax=axs[0], label="Énergie normalisée")

    # TimeSeries
    t0 = hfilt.t0.value
    dt = hfilt.dt.value
    times = np.arange(len(hfilt.value)) * dt + t0
    axs[1].plot(times, hfilt.value, color='red')
    # ax.set_title('MDC-ET strain data ' + GW_init[i] + '-2048')
    axs[1].set_ylabel('Amplitude [strain]')
    axs[1].set_xlim(t_start[i], t_stop[i])
    #+5600+1130+37
    #+5600+1130+45
    axs[1].set_xscale('seconds', epoch=t_start[i])
    plt.tight_layout(pad=2)
    if save:
        plt.savefig(dossier_save+'T'+str(type)+'_'+GW_init[i]+"OG_signal_spectro")
    #plt.close(fig='all')
    print("done")

    return hfilt, spectro



# from gwpy.io.gwf import get_channel_names
# channels = get_channel_names("/home/shared/et-mdc-frame-files/mdc1/v2/data/E1/E-E1_STRAIN_DATA-1000000000-2048.gwf")
# print(channels)

#======================================================================================================
#======================================================================================================
#======================================================================================================

def extract_mchirp_tc_spectro(tsgwpy_reel,ifo,q_lim,path,init,colorbar_limits = None,frange=(4, 150),qrange=(5, 50),
                              fres=0.1, tres=0.1,show_fit=False, save_qtrans = False, show_slices = False, save_fig = False):

    """
    Plot of the spectrogram of a signal to find the approximate chirpm and tc.

    Parameters
    ----------
    tsgwpy_reel : Gwpy TimeSeries 
        Gwpy TimeSeries of a reel signal.
    ifo : str
        "E1", "E2" or "E3" for the Einstein Telescope.
    q_lim : int
        The plot of the q-trasform is made only with the point over this limit.
    colorbar_limits : dict (optional)
        Dictionary containing the limits "inf" and "sup" of the colorbar.
    show_fit : bool (optional)
        Print the plot of the spectrogram fit.
    frange, qrange, fres, tres :
        Parameters of a traditional q-transform, set initialy to the best ones for a BBH.
    
    Returns
    -------
    Dictionary containing the "mchrip" (in solar mass) and "tc" from the fit.
    """

    #constants
    G = 6.674e-11
    c = 299792e3
    M = 1.9884 * 10**30

    def qtrans_plot(frange,qrange,fres,tres,colorbar_limits = colorbar_limits,q_lim = q_lim):
        qtrans = tsgwpy_reel[ifo].q_transform(frange=frange, qrange=qrange, fres=fres, tres=tres)

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

        if save_qtrans :
            plt.savefig(path + 'qtrans')
        
        range_t = len(qtrans.times.value)
        range_f = len(qtrans.frequencies.value)

        if show_slices : 
            plt.figure()
            print('cols : ',len(qtrans.value[0,:]))
            print('rows : ',len(qtrans.value[:,0]))
            x = np.arange(range_t/10, step=0.1)
            for i in range(range_f) :
                plt.plot(x,qtrans.value[:,i])
            plt.ylabel("Normalized energy")
            plt.xlabel("Time [seconds] from 1001620439.0")
            plt.savefig('/home/victor/Internship_Victor_CBC_ET/images/step1_qfit.svg', format = 'svg')
            
            plt.figure()
            for i in range(range_f) :
                if qtrans.value[:,i].max() > q_lim :
                    plt.plot(x,qtrans.value[:,i])
            plt.xlabel("Time [seconds] cut from 1001620439.0")
            plt.xlim(12,21.3)
            plt.savefig('/home/victor/Internship_Victor_CBC_ET/images/step2_qfit.svg', format = 'svg')

            plt.figure()
            max = []
            x_list = []
            for i in range(range_f) :
                if qtrans.value[:,i].max() > q_lim :
                    index = np.where(qtrans.value[:,i] == qtrans.value[:,i].max())[0][0]
                    max.append(qtrans.value[:,i].max())
                    x_list.append(x[index])
            plt.scatter(x_list,max,label = 'Maximums')
            print(x_list)
            print(max)
            plt.legend()
            plt.xlabel("Time [seconds] cut from 1001620439.0")
            plt.savefig('/home/victor/Internship_Victor_CBC_ET/images/step3_qfit.svg', format = 'svg')


        plt.figure()
        y_ = []
        x_ = []
        freq_list = qtrans.frequencies.value
        time_list = qtrans.times.value
        for i in range(range_t) :
            if qtrans.value[i,:].max() > q_lim :
                y_.append(freq_list[np.where(qtrans.value[i,:] == qtrans.value[i,:].max())[0][0]])
                x_.append(time_list[i] + tres/2)

        x_scaled = [i - x_[0] for i in x_] #Rescale of the time values to ease the fit.
    
        return x_scaled, x_, y_


    def function_fit(t, mchirp, tc): #def function to plot the spectrogram with mchirp and tc
        tau = tc - t
        return (1/np.pi) * (5/(256*tau))**(3/8) * ((G*mchirp)/(c**3))**(-5/8)
    
    x_scaled, x_, y_ = qtrans_plot(frange,qrange,fres,tres)

    try :
        popt, pcov = curve_fit(function_fit, x_scaled, y_, p0 = init)
    except RuntimeError: 
        print('Failed to fit the spectrogram. Test of another method ...')

        x_scaled, x_, y_ = qtrans_plot(frange,qrange = (4,25),fres=0.01,tres=0.01,colorbar_limits = {'inf' : 18, 'sup' : None}, q_lim=18)

        try :
            popt, pcov = curve_fit(function_fit, x_scaled, y_, p0 = init)
        except RuntimeError: 
            result = {"mchirp" : 'ERROR', "tc" : 'ERROR'}
            plt.figure()
            plt.scatter(x_,y_,label = 'Extracted points',s=4)
            plt.legend()
            if save_fig :
                plt.savefig(path + 'fit_qtrans_error01')
            
            x_scaled, x_, y_ = qtrans_plot(frange,qrange = (4,30),fres=0.005,tres=0.001,colorbar_limits = {'inf' : 14, 'sup' : None}, q_lim=14)

            try :
                popt, pcov = curve_fit(function_fit, x_scaled, y_, p0 = init)
            except RuntimeError:
                plt.figure()
                plt.scatter(x_,y_,label = 'Extracted points',s=4)
                plt.legend()
                if save_fig :
                    plt.savefig(path + 'fit_qtrans_error02')
                
                y_fit = None
                return result, x_, y_, y_fit

    result = {"mchirp" : popt[0], "tc" : popt[1]+x_[0]}
    print('cov :',pcov)

    if show_fit :
        y_fit = []
        for i,x in enumerate(x_):
            y_fit.append(function_fit(x,result["mchirp"],result["tc"]))
        plt.figure()
        plt.scatter(x_,y_,label = 'Extracted points',s=4)
        plt.plot(x_,y_fit,label = 'Fit curve',c='black')
        plt.legend()
    
    result["mchirp"] = popt[0]/M
    result["u_mchirp"] = np.sqrt(pcov[0,0])/M
    result["u_tc"] = np.sqrt(pcov[1,1])

    if save_fig :
        plt.savefig(path + 'q_trans_fit')
    

    return result, x_, y_, y_fit


#======================================================================================================
#======================================================================================================
#======================================================================================================

def instruments_PSD(instruments_dict,ET_MDC=False):

    """
    PSD plots of the different GW detection instruments from pycbc.psd.analytical.

    Parameters
    ----------
    instruments_dict : dict
        Dictionary containing the chosen name of the instrument and the method
        e.g. 'aLigo' : 'aLIGOAPlusDesignSensitivityT1800042'
    ET_MDC : bool
        Plot the ET nominal noise PSD used for the MDC.
    
    """

    for key, value in instruments_dict.items() :
        data = getattr(detector_psd, value)(length=200000, delta_f=1./100, low_freq_cutoff=4)
        plt.loglog(data.get_sample_frequencies(),data,label = key)

    if ET_MDC :
        ET10km = pd.read_csv('../input/ET10km_columns.txt',sep = ' ',names=["frequencies", "A", "B", "C"])
        plt.loglog(ET10km['frequencies'],ET10km['C'],label = 'ET (MDC)')

    plt.xlim(4,1000)
    plt.ylim(10e-51,10e-35)
    plt.xlabel(r'Frequency [Hz]')
    plt.ylabel(r'PSD [1/Hz]')
    plt.legend()
    plt.tight_layout()

    plt.savefig('/home/victor-glorieux/Internship_Victor_CBC_ET/images/instruments_PSD_comparison.svg', format='svg')

    plt.close('all')

#======================================================================================================
#======================================================================================================
#======================================================================================================

#from pycbc.detector import Detector
def antenna_factors(detectors,params):
    """
    Calculation of the antenna factors of a detector for a given signal.

    Parameters
    ----------
    detectors : list, str
        List of the detectors e.g. 'E1', 'H1', ...
    params : dict
        Dictionary containing the parameters of the signal.
        'ra', 'dec', 'polarization' and 'tc' are required.
    
    Returns
    -------
    Dictionary with fp and another with fx for each detector.
    """
    
    fp = {}
    fx = {}
    for ifo in detectors : 
        fp[ifo], fx[ifo] = Detector(ifo).antenna_pattern(params['ra'], params['dec'], params["polarization"], params['tc'])
    
    return fp, fx

#======================================================================================================
#======================================================================================================
#======================================================================================================

def comparison_signals_params(model,dict_param, cbc_params,domain,label,x_time_lim = {'inf' : 1, 'sup' : 0.2},spectroplot = False, ifos = ['E1','E2','E3'], save_fig = False):
    """
    PLot : Comparison of a same signal with different parameters.

    Parameters
    ----------
    model : MDCGaussianNoise
        MDCGaussianNoise model of GW.
    dict_param : dict
        Dictionary containing the parameter which is changed with the wanted values.
        e.g. dict_params = {'param' : 'mass1', 'A' : 100, 'B' : 20, 'C' : 5}.
    cbc_params : dict
        Dictionary containing the parameters of the cbc.
    domain : str
        'freq' or 'time' depending on the domain we want to plot the signals.
    
    """

    if domain == 'time' :
        fig_ts = plt.figure(figsize = (8,4))
        ax_ts = fig_ts.gca()
    elif domain == 'freq' :
        fig_fs = plt.figure()
        ax_fs = fig_fs.gca()

    for key, val in dict_param.items():
        if key == 'param' :
            continue
        else :
            cbc_params[dict_param['param']] = val
            model.maximized_params = cbc_params
            reconstructed_signal_fdomain, reconstructed_signal_tdomain = model.reconstruct_signal()
            for ifo in ifos :
                tc=cbc_params['tc']
                t_end = reconstructed_signal_tdomain[ifo].get_sample_times()[-1]
                reconstructed_signal_tdomain[ifo] = reconstructed_signal_tdomain[ifo].cyclic_time_shift(t_end - tc - 0.2)
                reconstructed_signal_fdomain[ifo] = reconstructed_signal_tdomain[ifo].to_frequencyseries()
            
            if domain == 'time' :
                ax_ts.plot(reconstructed_signal_tdomain['E1'].get_sample_times(),reconstructed_signal_tdomain['E1'],label = label[key])

            else :
                # for ifo in ifos :
                #     reconstructed_signal_tdomain[ifo] = reconstructed_signal_tdomain[ifo].time_slice(tc - 5, tc + 0.2)
                tsgwpy = pycbc_to_gwpy(reconstructed_signal_tdomain)
                if domain == 'freq' :
                    psd_gwpy = tsgwpy['E1'].psd()
                    ax_fs.loglog(psd_gwpy.frequencies,psd_gwpy,label = label[key])
                else : 
                    # Gwpy_TimeSeries = {}
                    # for ifo in ifos :
                    #     Gwpy_TimeSeries[ifo] = TimeSeries(data = reconstructed_signal_tdomain[ifo],times=reconstructed_signal_tdomain[ifo].get_sample_times() + 100)
                    qtrans_plot(tsgwpy['E1'],frange = (5,100),qrange = (5,15),fres=0.1,tres = 0.1,colorbar_limits = {'inf' : 0, 'sup' : 10000})

                    plt.figure()
                    plt.plot(tsgwpy['E1'].times, tsgwpy['E1'])
                    plt.xlim(tc-5, tc + 0.2)

            if spectroplot :
                spectro = tsgwpy['E1'].spectrogram(1, fftlength=1)
                plot = spectro.plot(figsize=[8, 4])
                ax = plot.gca()
                ax.set_ylim(4, 150)
                #ax.set_xlim(tc - 50, tc + 0.5)
                ax.set_xscale('seconds')
                ax.set_yscale('log')
                ax.grid(True, axis='y', which='both')
                ax.colorbar(cmap='viridis', label='Normalized energy')



    if domain == 'time' :
        ax_ts.set_xlabel('Time [s]')
        ax_ts.set_ylabel('Strain')
        ax_ts.set_xlim(tc - x_time_lim['inf'],tc + x_time_lim['sup'])
        #ax_ts.legend()

    elif domain == 'freq' :
        ax_fs.set_xlabel('Frequency [Hz]')
        ax_fs.set_ylabel('PSD [1/Hz]')
        ax_fs.set_xlim(4,1000)
        ax_fs.set_ylim(10e-55,10e-44)
    
        ET10km = pd.read_csv('../input/ET10km_columns.txt',sep = ' ',names=["frequencies", "A", "B", "C"])
        ax_fs.loglog(ET10km['frequencies'],ET10km['C'],label = 'Nominal noise PSD')
        ax_fs.legend()
        

    plt.tight_layout

    if save_fig :
        plt.savefig('/home/victor-glorieux/Internship_Victor_CBC_ET/images/' + dict_param['param'] + '_unique_' + domain + '.svg', format = 'svg')
        #plt.savefig('/home/victor-glorieux/Internship_Victor_CBC_ET/images/' + dict_param['param'] + '_comp_' + domain + '.svg', format = 'svg')


#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================


def likelihood_visualisation(model,true_params,params = 'all',fig_name = None,save_fig = False):
    """
    Plot of the likelihood ratio for all the parameters used during the maxmimization process, giving a GW model.

    Parameters
    ----------
    model : MDCGaussianNoise
        MDCGaussianNoise model of GW.
    true_params : dict
        Dictionary containing the true parameter of the signal : the best loglr parameters.
    params : list (optional)
        List of the params we want to plot. Default : 'all' for all params.
    fig_name : str (optional)
    save_fig : bool (optional)
    
    """

    mchirp_true = mchirp_from_mass1_mass2(true_params['mass1'],true_params['mass2'])
    q_true = q_from_mass1_mass2(true_params['mass1'],true_params['mass2'])

    spinSz_true = spinSz_from_sz_mass(true_params['spin1z'],true_params['spin2z'],true_params['mass1'],true_params['mass2'])
    spinAz_true = spinAz_from_sz_mass(true_params['spin1z'],true_params['spin2z'],true_params['mass1'],true_params['mass2'])


    def plot_lik(axs_list,label_x,data_x,true_params,param_min,param_max,echantill,nb_graphs,q):
        clear_output(wait=True)
        ax = axs_list[data_x]

        model.update(**true_params)
        params_modif =  copy.deepcopy(true_params)

        x_grid = np.arange(param_min[data_x],param_max[data_x],echantill[data_x])
        y_grid = np.zeros(len(x_grid))
        print("Iterations totales : ",len(y_grid))
        k=0
        for i, x_ in enumerate(x_grid):
            if data_x == 'mass1' :
                mass1 = mass1_from_mchirp_q(mchirp=x_,q=q_true)
                mass2 = mass2_from_mchirp_q(mchirp=x_,q=q_true)
                params = {'mass1' : mass1, 'mass2' : mass2}
                params_modif.update(params)
                model.update(**params_modif)
                y_grid[i]=-model.loglr
            elif data_x == 'mass2' :
                mass1 = mass1_from_mchirp_q(mchirp=mchirp_true,q=x_)
                mass2 = mass2_from_mchirp_q(mchirp=mchirp_true,q=x_)
                params = {'mass1' : mass1, 'mass2' : mass2}
                params_modif.update(params)
                model.update(**params_modif)
                y_grid[i]=-model.loglr
            elif data_x == 'spinSz' :
                spin1z = s1z_from_spinSz_spinAz(spinSz = x_,spinAz = spinAz_true,mass1 = true_params['mass1'],mass2 = true_params['mass2'])
                spin2z = s2z_from_spinSz_spinAz(spinSz = x_,spinAz = spinAz_true,mass1 = true_params['mass1'],mass2 = true_params['mass2'])
                params = {'spin1z' : spin1z, 'spin2z' : spin2z}
                params_modif.update(params)
                model.update(**params_modif)
                y_grid[i]=-model.loglr
            elif data_x == 'spinAz' :
                spin1z = s1z_from_spinSz_spinAz(spinSz = spinSz_true,spinAz = x_,mass1 = true_params['mass1'],mass2 = true_params['mass2'])
                spin2z = s2z_from_spinSz_spinAz(spinSz = spinSz_true,spinAz = x_,mass1 = true_params['mass1'],mass2 = true_params['mass2'])
                params = {'spin1z' : spin1z, 'spin2z' : spin2z}
                params_modif.update(params)
                model.update(**params_modif)
                y_grid[i]=-model.loglr
            else :
                params = {data_x : x_} #Les paramètres que l'on souhaite modifier sur le modèle de notre GW
                params_modif.update(params)
                model.update(**params_modif) #Modification du modèle 
                y_grid[i]=-model.loglr

            k +=1
            print ("Plot : {}/{}, iteration : {}".format(q,nb_graphs,k), end="\r")

        ax.plot(x_grid,y_grid,label = r"-log($\mathcal{L}$)")

        if data_x == 'mass1' :
            ax.set_xlabel(r'$M_\text{chirp}$',fontsize = 30)
            ax.axvline(mchirp_true,color = 'red',label = 'True param')
        elif data_x == 'mass2' :
            ax.set_xlabel('q',fontsize = 30)
            ax.axvline(q_true,color = 'red',label = 'True param')
            ax.axvline(1/q_true,color = 'red',label = '1/q',ls = '--')
        elif data_x == 'spinSz' :
            ax.set_xlabel(label_x[data_x],fontsize = 30)
            ax.axvline(spinSz_true,color = 'red',label = 'True param')
        elif data_x == 'spinAz' :
            ax.set_xlabel(label_x[data_x],fontsize = 30)
            ax.axvline(spinAz_true,color = 'red',label = 'True param')
        else :
            ax.set_xlabel(label_x[data_x],fontsize = 30)
            ax.axvline(true_params[data_x],color = 'red',label = 'True param')
        ax.tick_params(labelsize = 20)
        ax.legend(fontsize = 25)
    
    fig_lik, axs = plt.subplots(nrows=3, ncols=4, figsize = (40,20))

    axs_list = {'tc' : axs[0,0], 'mass1' : axs[0,1], 'mass2' : axs[0,2], 'distance'  : axs[0,3], 'ra'    : axs[1,0], 'dec' : axs[1,1], 'polarization' : axs[1,2], 'inclination' : axs[1,3], 'spin1z' : axs[2,0], 'spin2z'  : axs[2,1], 'coa_phase'  : axs[2,2], 'spinSz' : axs[2,0], 'spinAz' : axs[2,1]}
    label_x = {'tc'  : r'$t_c$', 'mass1' : r'$m_1$', 'mass2' : r'$m_2$',  'distance' : r'$r$', 'ra' : r'$\alpha$', 'dec'    : r'$\delta$', 'polarization'   : r'$\psi$', 'inclination'  : r'$\iota$', 'spin1z'  : r'$\text{spin}_{1z}$', 'spin2z' : r'$\text{spin}_{2z}$', 'coa_phase' : r'$\phi_c$', 'spinSz' : r'$\text{spin}_{Sz}$', 'spinAz' : r'$\text{spin}_{Az}$'}
    data_x = ['tc',   'mass1',    'mass2',    'distance',    'ra',       'dec',      'polarization', 'inclination', 'spin1z',   'spin2z', 'coa_phase', 'spinSz', 'spinAz']
    param_min = {'tc' : true_params['tc'] - 0.5,'mass1' :   mchirp_true - 2,'mass2' :   0.2,'distance'  :     true_params['distance'] - 200,'ra' :        0,'dec' :   -np.pi/2,'polarization' :        0,'inclination' :       0,'spin1z' :    -1,'spin2z' :   -1,'coa_phase' : 0, 'spinSz' : -0.8, 'spinAz' : -0.8}
    param_max = {'tc' : true_params['tc'] + 0.5,'mass1' :  mchirp_true + 2,'mass2' :      3,'distance'  :   true_params['distance'] + 200,'ra' :  2*np.pi,'dec' :    np.pi/2,'polarization' :  2*np.pi,'inclination' :   np.pi,'spin1z' :     1,'spin2z' :    1,'coa_phase' : 2*np.pi, 'spinSz' : 0.8, 'spinAz' : 0.8}
    echantill = {'tc' : 0.0005,'mass1' :   0.01,'mass2' : 0.005,'distance'  :     1,'ra' :    0.01,'dec' :     0.01,'polarization' :     0.01,'inclination' :    0.01,'spin1z' :  0.01,'spin2z' : 0.01,'coa_phase' : 0.01, 'spinSz' : 0.005, 'spinAz' : 0.005}

    nb_graphs = len(data_x)

    q=0
    if params == 'all' :
        for i in range(nb_graphs ):
            q += 1
            plot_lik(axs_list,label_x,data_x[i],true_params,param_min,param_max,echantill,nb_graphs = nb_graphs,q=q)
    else :
        for data_x in params :
            q += 1
            plot_lik(axs_list,label_x,data_x,true_params,param_min,param_max,echantill,nb_graphs = len(params),q=q)
    
    fig_lik.tight_layout()

    if save_fig :
        plt.savefig(fig_name + '.svg', format='svg')

#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================

def extract_best_SNR(SNR_lower_limit, source = 'local'):
    """
    Find the best SNR signals from the list_mdc1_v2.txt.

    Parameters
    ----------
    SNR_lower_limit : dict
        {"type_1" : SNR_lower_limit,"type_2" : SNR_lower_limit,"type_3" : SNR_lower_limit}
    
    Returns
    ----------
    Dictionary with lists of indexes for best SNR signals for each type of CBC.

    """ 
    if source == 'IJCLab_server' :
        ET_params = pd.read_csv("/home/victor-glorieux/Internship_Victor_CBC_ET/code_Adrian/MLE_pipeline/data/loudest_BBH/list_mdc1_v2.txt",sep = ' ',engine='python', index_col = False)
    elif source == 'local' :
        ET_params = pd.read_csv("/home/victor/Internship_Victor_CBC_ET/code_Adrian/MLE_pipeline/data/loudest_BBH/list_mdc1_v2.txt",sep = ' ',engine='python', index_col = False)

    dict_best_SNR = {}
    ET_params = ET_params.sort_values('snr',ascending=False)
    for i, type in enumerate(['type_1','type_2','type_3']) : 
        ET_params_ = ET_params[ET_params['type'] == i+1]
        ET_params_ = ET_params_[ET_params_['snr'] > SNR_lower_limit[type]]
        number_high_SNR = len(ET_params_['snr'])
        list_best_SNR = np.arange(number_high_SNR)
        dict_best_SNR[type] = list_best_SNR
    
    return dict_best_SNR



#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================



def read_pickle_from_file(folder_output,pickle_file,study_type,infos,compare_sig = True, compare_freq=True, q_transform=True):

    '''
    info : dict {'t_start_signal' : ,'t_end_signal' : }
    
    '''

    ifos = ['E1','E2','E3']

    path_folder = folder_output

    with open(path_folder + '/' + pickle_file, 'rb') as f:
        model= pickle.load(f)
    print(' - Lecture du pickle : done')

    params = model.maximized_params

    tc = params['tc']

    #Lecture du signal complet du MDC avec read_MDC_data de get_data.py ==================================
    original_tsd = read_MDC_data(infos['t_start_signal'], infos['t_end_signal'] + 1)
    original_tsd = gwpy_to_pycbc(original_tsd)
    #original_tsd = read_MDC_data(signal_reconstructed_time['E1'].start_time, signal_reconstructed_time['E1'].end_time)
    print(' - Lecture des données complètes du MDC : done')

    signal_reconstructed_freq, signal_reconstructed_time = model.reconstruct_signal()
    for ifo in ifos :
        #original_tsd[ifo] = original_tsd[ifo].to_pycbc()
        t_end = signal_reconstructed_time[ifo].get_sample_times()[-1]
        signal_reconstructed_time[ifo] = signal_reconstructed_time[ifo].cyclic_time_shift(t_end - tc - 1)
    print(' - Conversion des données en séries temporelles : done')
    

    signal_reconstructed_time_cut = {}
    signal_reconstructed_time_freq = {}
    original_tsd_cut = {}
    original_tsd_freq = {}

    for ifo in ifos :
        if study_type == 3 :
            signal_reconstructed_time_cut[ifo] = signal_reconstructed_time[ifo].time_slice(tc - 2,tc+0.2)
            original_tsd_cut[ifo] = original_tsd[ifo].time_slice(tc - 2,tc+0.2)

            original_tsd_freq[ifo] = original_tsd[ifo].time_slice(tc - 3 , tc+0.2)
            signal_reconstructed_time_freq[ifo] = signal_reconstructed_time[ifo].time_slice(tc - 3 , tc+0.2)
        
        else :
            signal_reconstructed_time_cut[ifo] = signal_reconstructed_time[ifo].time_slice(tc - 0.5, tc+0.05)
            original_tsd_cut[ifo] = original_tsd[ifo].time_slice(tc - 0.5, tc+0.05)

            original_tsd_freq[ifo] = original_tsd[ifo].time_slice(tc - 20, tc+0.1)
            signal_reconstructed_time_freq[ifo] = signal_reconstructed_time[ifo].time_slice(tc - 20, tc+0.1)
            
    print(' - Découpage des données : done')

    residual_time_cut = subtract_signal(original_tsd_cut, signal_reconstructed_time_cut)
    residual_time_freq = subtract_signal(original_tsd_freq, signal_reconstructed_time_freq)
    print(' - Calcul des résidus : done')

    if compare_sig :

        comparison_signals(params,signal_reconstructed_time_cut,original_tsd_cut,residual_time_cut,ifo = 'E1',position = "Front",source="MLE_pipeline",save_fig = True,infos=infos)
        print(' - [PLOT] Comparaison signal : done')

    if compare_freq :

        average_noise = {'status' : True, 'ech' : 15}
        if study_type == 3 :
            original_tsd = read_MDC_data(signal_reconstructed_time[ifo].start_time, signal_reconstructed_time[ifo].end_time)
            original_tsd = gwpy_to_pycbc(original_tsd)

            residual_time = subtract_signal(original_tsd, signal_reconstructed_time)
            comparison_freq(signal_reconstructed_time,original_tsd,residual_time,ifo = 'E1',noisePSD=True,average_noise=average_noise,save_fig = True,infos=infos)
        else :
            comparison_freq(signal_reconstructed_time_freq,original_tsd_freq,residual_time_freq,ifo = 'E1',noisePSD=True,average_noise=average_noise,save_fig = True,infos=infos)
        print(' - [PLOT] Comparaison fréquence : done')

    if q_transform :

        tsgwpy_real = pycbc_to_gwpy(original_tsd_freq)
        tsgwpy_res = pycbc_to_gwpy(residual_time_freq)
        #colorbar_limits = {'inf' : 0, 'sup' :1500}
        if study_type == 3 :
            frange = (4, 100)
            qrange = (12, 30)
        else :
            frange = (10, 250)
            qrange = (30, 150)
        qtrans_plot(tsgwpy_real['E1'],fres = 0.01,tres = 0.01,frange = frange,qrange = qrange, name = '_MDC', save_fig = True, infos = infos)
        qtrans_plot(tsgwpy_res['E1'],tres = 0.01,frange = frange,colorbar_limits = {'inf' : 0, 'sup' :500},qrange = qrange,name = '_res', save_fig = True, infos = infos)
        print(' - [PLOT] q-transforms : done')