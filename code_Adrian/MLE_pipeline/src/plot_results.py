from pycbc.conversions import mchirp_from_mass1_mass2, mass1_from_mchirp_q, mass2_from_mchirp_q
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from pycbc.types import TimeSeries as PycbcTimeSeries

def comparison_signals(model,reconstructed_signal_tdomain,data,residual,save_fig):
    # Calcul de la différence
    #residual = reconstructed_signal_tdomain['E1'] - trunc_data['E1']

    mchirp_true = model.maximized_params['chirp'].values[0]
    q_true = model.maximized_params['q'].values[0]
    para_opti = [0, model.maximized_params['tc'].values[0],  mass1_from_mchirp_q(mchirp_true, q_true),  mass2_from_mchirp_q(mchirp_true, q_true),
                model.maximized_params['distance'].values[0], model.maximized_params['ra'].values[0], model.maximized_params['dec'].values[0],  model.maximized_params['polarization'].values[0], model.maximized_params['inclination'].values[0],
                model.maximized_params['spin1z'].values[0],  model.maximized_params['spin2z'].values[0], model.maximized_params['coa_phase'].values[0]]
    #24.386689 23.987984 0.0756 -0.023644 0.037543 0.061211 0.1384 -0.080838 0.007908 0.112059 0 0 0.1137 545.6354 -1.418225 0.795954 1.295625 2.002555 2.949147 587.553918 653.770127 3
    para_reels = [1.001620460259506941e+09, 24.38, 23.98, 545.63, -1.42, 0.80, 'NAN', 2.00, 0.061,0.11]

    # Création des subplots : 2 lignes, 1 colonne, partagent l'axe X
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Tracer les deux signaux dans le premier axe
    ax1.plot(reconstructed_signal_tdomain['E1'].get_sample_times(),reconstructed_signal_tdomain['E1'],label='E1_opti', zorder = 2)
    ax1.set_xlim(1001620399.26659 + 45, 1001620463.11925 - 2.5)
    #ax1.set_xlim(1001620399.26659 + 45, 1001620463.11925 - 14)
    #ax1.set_xlim(1001620399.26659 + 60, 1001620463.11925 - 2.6)
    ax1.set_xscale('seconds', epoch=(1001620399.26659 + 45))
    ax1.plot(data['E1'].times,data['E1'], color='orange',label='E1_reel',zorder = 1)
    ax1.set_title('Signals comparison for optimized parameters', fontsize = 30)
    ax1.set_ylabel('Amplitude')
    ax1.text(1.7, 1.1, r'Params_opti : $t_c$ : {}, $m_1$ : {}, $m_2$ : {}, $d_L$ : {}, ra : {}, dec : {}, pola : {}, incl : {}, s1z : {}, s2z : {}, coa_phase : {}.'.format(round(para_opti[1],3),
        round(para_opti[2],4), round(para_opti[3],4), round(para_opti[4],4), round(para_opti[5],4),round(para_opti[6],4), round(para_opti[7],4),round(para_opti[8],4),
        round(para_opti[9],4),round(para_opti[10],4),round(para_opti[11],4)), horizontalalignment='center',
        verticalalignment='center', transform=ax1.transAxes,fontsize=12)
    ax1.text(1.7, 1.2, r'Params_reels : $t_c$ : {}, $m_1$ : {}, $m_2$ : {}, $d_L$ : {}, ra : {}, dec : {}, pola : {}, incl : {}, s1z : {}, s2z : {}.'.format(round(para_reels[0],3),
        para_reels[1], para_reels[2], para_reels[3], para_reels[4], para_reels[5], para_reels[6], para_reels[7], para_reels[8], para_reels[9]), horizontalalignment='center',
        verticalalignment='center', transform=ax1.transAxes,fontsize=12)
    ax1.legend(fontsize=20)

    # Tracer la différence dans le second axe
    ax2.plot(residual.get_sample_times(), residual, color='black')
    ax2.set_ylabel('Résidual')
    ax2.set_xlabel('Time [s]')


    plt.tight_layout()
    if save_fig :
        plt.savefig("Full_loc_minim_L1_Comparaison_signal")


def comparison_freq(opti_cut,reel_cut,residual,psd_opti,psd_reel,psd_res):
    #Conversion en GwpyTimeseries ======================
    #tsgwpy_opti_cut = TimeSeries(data = opti_cut,times=opti_cut.get_sample_times())
    #tsgwpy_reel_cut = TimeSeries(data = reel_cut,times=reel_cut.get_sample_times())
    #tsgwpy_res = TimeSeries(data = residual,times=residual.get_sample_times())

    #Calcul des psd
    # psd_opti = tsgwpy_opti_cut.psd()
    # psd_reel = data['E1'].psd()
    # psd_res = psd_reel - psd_opti

    plt.figure()
    ax = plt.gca()
    ax.loglog(psd_opti.frequencies, psd_opti, label='E1 optimized, PSD',zorder=3)
    ax.loglog(psd_reel.frequencies, psd_reel, label='E1 reel, PSD',zorder=2)
    ax.loglog(psd_res.frequencies, psd_res, label='PSD Residual E1',zorder = 1)
    ax.set_ylim(1e-53, 1e-44)
    ax.set_xlim(4, 2048)
    ax.legend()
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD [1/Hz]')

def qtrans(tsgwpy):
    qtrans = tsgwpy.q_transform(frange=(4, 100), qrange=(5, 30))

    plot = qtrans.plot(figsize=[8, 4])
    ax = plot.gca()
    ax.set_ylim(5, 100)
    #ax.set_xlim(t0_spectro, tf_spectro)
    ax.set_xscale('seconds')
    ax.set_yscale('log')
    ax.grid(True, axis='y', which='both')
    ax.colorbar(cmap='viridis', label='Normalized energy')