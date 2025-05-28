from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from matplotlib import pyplot as plt
from gwpy.plot import Plot
from gwpy.signal import filter_design
import pandas as pd
import numpy as np

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

#==============================================================================
#Permet de déterminer, à partir du fichier de données "list_mdc1.txt", le nom de l'observations (temps dans les données présentes sur le serveur IJCLab) associé
#à l'événement de type sélectionné qui a le SNR le plus grand. La liste indexes nons permet de délectionner plusieurs signaux qui ont le plus grand SNR.
def extraction_temps(indexes,type,print_):

    #Extraction du fichier qui contient le nom des observations du MDC présentes sur le serveur de l'IJCLab
    cols = ["col1","col2","col3"]
    ET = pd.read_csv("ET_data.txt",sep = '  ',engine='python')

    #Ce code permet d'extraire les refs, t0 et tc des événements d'onde GW que nous voulons regarder.
    #Nous listons les indices des évenements dans "indexes" et nous regardons en priorité les événements avec le meilleur SNR.
    #Une "ref" correspond à l'indice du fichier "ET_data" qui nous permet d'y trouver le nom du fichier contenant les données que nous souhaitons regarder.
    #Une "ref_sup" est l'indice de fin de nos événement.
    def temps_ref(indexes):
        ET_params = pd.read_csv("list_mdc1.txt",sep = ' ',engine='python')
        ET_params = ET_params.sort_values('snrET_Opt',ascending=False) #Sélectionne les events avec le meilleur SNR pour les indices les plus faibles.
        ET_params = ET_params[ET_params['type'] == type] #Sélectionne un type particulier d'événements.
        #print(ET_params.head())
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
        return ref_list,t0_list,tc_list, ref_sup

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

    ref_list,t0_list,tc_list,ref_sup = temps_ref(indexes)
    for i in range(len(ref_list)):
        if print_ ==True:
            print('t0 :', t0_list[i])
            print('tc :', tc_list[i])
        find_ref(ref_list[i],ref_sup[i],t0_list[i],tc_list[i])

    t0_list = [float(t0_list[i]) for i in range(len(t0_list))]
    tc_list = [float(tc_list[i]) for i in range(len(tc_list))]
    return init, final, t0_list, tc_list, interval

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