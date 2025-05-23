import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
import matplotlib.pyplot as plt
from gwpy.plot import Plot
from pycbc import psd as pypsd
from pycbc.inference.models import GaussianNoise
from pycbc.noise.gaussian import frequency_noise_from_psd
from pycbc.waveform.generator import (FDomainDetFrameGenerator,FDomainCBCGenerator)
from pycbc.psd import EinsteinTelescopeP1600143
from pycbc.conversions import mchirp_from_mass1_mass2, q_from_mass1_mass2, mass1_from_mchirp_q, mass2_from_mchirp_q


#====================================================
#fonctions : plot_dual_local_minimisation, plot_mnimisation, plot_correlation, plot_correlation_2_params, plot_correlation_chirp_q
#====================================================


def plot_dual_local_minimisation(params_dataFrame,x_data,y_data,x_label,y_label,save_fig):

    x_list=params_dataFrame[x_data]
    y_list=params_dataFrame[y_data]
    ll_ratio_test=params_dataFrame['mloglik']

    data_tail = params_dataFrame.loc[params_dataFrame['mloglik'].idxmin()]

    plt.scatter(x_list, y_list, c=ll_ratio_test, cmap='viridis', s=100)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.axhline(float(data_tail[x_data]),linestyle = 'dashed',color='r')
    plt.axvline(float(data_tail[x_data]),linestyle = 'dashed',color='r')
    colorbar = plt.colorbar(orientation='vertical')
    colorbar.set_label(r'-log($\mathcal{L}$)', labelpad=10)

    if save_fig :
        plt.savefig("minimisation/"+ x_data +"_"+ y_data +"_minimisation_locale")





def plot_mnimisation(params_dataFrame,para_opti,para_reels,initial_params,log_noise_likelihood_from_SNR,mini,save_fig):

    tc_list=params_dataFrame['tc']
    mass1_list=params_dataFrame['mass1']
    mass2_list=params_dataFrame['mass2']
    distance_list=params_dataFrame['distance']
    ra_list=params_dataFrame['ra']
    dec_list=params_dataFrame['dec']
    pola_list=params_dataFrame['polarization']
    incl_list=params_dataFrame['inclination']
    s1z_list=params_dataFrame['spin1z']
    s2z_list=params_dataFrame['spin2z']
    ll_ratio_test=params_dataFrame['mloglik']

    data_tail = params_dataFrame.loc[params_dataFrame['mloglik'].idxmin()]

    nb_params = len(para_reels)
    fig, axs = plt.subplots(nrows=nb_params,ncols=nb_params,figsize=(40,40))


    def plot_corr(x_list,y_list,label_x,label_y,data_x,data_y,ax,data_tail):
        ax.scatter(x_list, y_list, c=ll_ratio_test, cmap='viridis', s=100)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.axhline(float(data_tail[data_y]),linestyle = 'dashed',color='r', label=label_y + r'$_{,opt}$ = ' + str(round(float(data_tail[data_y]),2)))
        ax.axvline(float(data_tail[data_x]),linestyle = 'dashed',color='r', label=label_x + r'$_{,opt}$ = ' + str(round(float(data_tail[data_x]),2)))
        ax.legend()

    axs_list = [axs[0,1], axs[0,2],   axs[0,3],      axs[0,4],   axs[0,5],   axs[0,6],       axs[0,7],      axs[0,8],   axs[0,9],   axs[1,2],    axs[1,3],     axs[1,4], axs[1,5], axs[1,6],        axs[1,7],      axs[1,8],  axs[1,9],  axs[2,3],      axs[2,4],   axs[2,5],   axs[2,6],       axs[2,7],      axs[2,8],   axs[2,9],   axs[3,4],      axs[3,5],      axs[3,6],       axs[3,7],      axs[3,8],      axs[3,9],      axs[4,5], axs[4,6],       axs[4,7],      axs[4,8],  axs[4,9],  axs[5,6],       axs[5,7],      axs[5,8],  axs[5,9],  axs[6,7],      axs[6,8],       axs[6,9],       axs[7,8],      axs[7,9],      axs[8,9]]
    x_list = [tc_list,    mass2_list, distance_list, ra_list,    dec_list,   pola_list,      incl_list,     s1z_list,   s2z_list,   mass2_list, distance_list, ra_list,  dec_list,  pola_list,      incl_list,     s1z_list,  s2z_list,  distance_list, ra_list,    dec_list,   pola_list,      incl_list,     s1z_list,   s2z_list,   ra_list,       dec_list,      pola_list,      incl_list,     s1z_list,      s2z_list,      dec_list, pola_list,      incl_list,     s1z_list,  s2z_list,  pola_list,      incl_list,     s1z_list,  s2z_list,  incl_list,     s1z_list,       s2z_list,       s1z_list,      s2z_list,      s2z_list]
    y_list = [mass1_list, mass1_list, mass1_list,    mass1_list, mass1_list, mass1_list,     mass1_list,    mass1_list, mass1_list, tc_list,    tc_list,       tc_list,  tc_list,   tc_list,        tc_list,       tc_list,   tc_list,   mass2_list,    mass2_list, mass2_list, mass2_list,     mass2_list,    mass2_list, mass2_list, distance_list, distance_list, distance_list,  distance_list, distance_list, distance_list, ra_list,  ra_list,        ra_list,       ra_list,   ra_list,   dec_list,       dec_list,      dec_list,  dec_list,  pola_list,     pola_list,      pola_list,      incl_list,     incl_list,     s1z_list]
    label_x = [r'$t_c$',  r'$m_2$',   r'distance',   r'ra',      r'dec',     r'pola',        r'incl',       r's_{1z}',  r's_{2z}',  r'$m_2$',   r'distance',   r'ra',    r'dec',    r'pola',        r'incl',       r's_{1z}', r's_{2z}', r'distance',   r'ra',      r'dec',     r'pola',        r'incl',       r's_{1z}',  r's_{2z}',  r'ra',         r'dec',        r'pola',        r'incl',       r's_{1z}',     r's_{2z}',     r'dec',   r'pola',        r'incl',       r's_{1z}', r's_{2z}', r'pola',        r'incl',       r's_{1z}', r's_{2z}', r'incl',       r's_{1z}',      r's_{2z}',      r's_{1z}',     r's_{2z}',     r's_{2z}']
    label_y = [r'$m_1$',  r'$m_1$',   r'$m_1$',      r'$m_1$',   r'$m_1$',   r'$m_1$',       r'$m_1$',      r'$m_1$',   r'$m_1$',   r'$t_c$',   r'$t_c$',      r'$t_c$', r'$t_c$',  r'$t_c$',       r'$t_c$',      r'$t_c$',  r'$t_c$',  r'$m_2$',      r'$m_2$',   r'$m_2$',   r'$m_2$',       r'$m_2$',      r'$m_2$',   r'$m_2$',   r'distance',   r'distance',   r'distance',    r'distance',   r'distance',   r'distance',   r'ra',    r'ra',          r'ra',         r'ra',     r'ra',     r'dec',         r'dec',        r'dec',    r'dec',    r'pola',       r'pola',        r'pola',        r'incl',       r'incl',       r's_{1z}']
    data_x = ['tc',       'mass2',    'distance',    'ra',       'dec',      'polarization', 'inclination', 'spin1z',   'spin2z',   'mass2',    'distance',    'ra',     'dec',     'polarization', 'inclination', 'spin1z',  'spin2z',  'distance',    'ra',       'dec',      'polarization', 'inclination', 'spin1z',   'spin2z',   'ra',          'dec',         'polarization', 'inclination', 'spin1z',      'spin2z',      'dec',    'polarization', 'inclination', 'spin1z',  'spin2z',  'polarization', 'inclination', 'spin1z',  'spin2z', 'inclination',  'spin1z',       'spin2z',       'spin1z',      'spin2z',      'spin2z']
    data_y = ['mass1',    'mass1',    'mass1',       'mass1',    'mass1',    'mass1',        'mass1',       'mass1',    'mass1',    'tc',       'tc',          'tc',     'tc',      'tc',           'tc',          'tc',      'tc',      'mass2',       'mass2',    'mass2',    'mass2',        'mass2',       'mass2',    'mass2',    'distance',    'distance',    'distance',     'distance',    'distance',    'distance',    'ra',     'ra',           'ra',          'ra',      'ra',      'dec',          'dec',         'dec',     'dec',    'polarization', 'polarization', 'polarization', 'inclination', 'inclination', 'spin1z']

    for i in range(len(axs_list)):
        plot_corr(x_list[i],y_list[i],label_x[i],label_y[i],data_x[i],data_y[i],axs_list[i],data_tail)

    fig.tight_layout()



    axs_off = [axs[1,0],axs[2,0],axs[3,0],axs[4,0],axs[5,0],axs[6,0],axs[7,0],axs[8,0],axs[9,0],
            axs[2,1],axs[3,1],axs[3,2],axs[4,1],axs[4,2], axs[4,3],
            axs[5,1],axs[5,2], axs[5,3], axs[5,4],
            axs[6,1],axs[6,2], axs[6,3], axs[6,4],axs[6,5],
            axs[7,1],axs[7,2], axs[7,3], axs[7,4],axs[7,5],axs[7,6],
            axs[8,1],axs[8,2], axs[8,3], axs[8,4],axs[8,5],axs[8,6],axs[8,7],
            axs[9,1],axs[9,2], axs[9,3], axs[9,4],axs[9,5],axs[9,6],axs[9,7],axs[9,8]]
    for i in axs_off:
        i.axis('off')
    ax=axs[7,2]
    mass1_init = round(mass1_from_mchirp_q(mchirp=initial_params[1],q=initial_params[2]),2)
    mass2_init = round(mass2_from_mchirp_q(mchirp=initial_params[1],q=initial_params[2]),2)
    mass1_opti = round(mass1_from_mchirp_q(mchirp=para_opti['chirp'],q=para_opti['q']),2)
    mass2_opti = round(mass2_from_mchirp_q(mchirp=para_opti['chirp'],q=para_opti['q']),2)
    ax.text(0.5, 0.95, r'Params_opti : $t_c$ : {}, $m_1$ : {}, $m_2$ : {}, $d_L$ : {}, ra : {}, dec : {}, pola : {}, incl : {}, s1z : {}, s2z : {}.'.format(round(para_opti['tc'],2),
        mass1_opti, mass2_opti, round(para_opti['distance'],2), round(para_opti['ra'],2), round(para_opti['dec'],2),round(para_opti['polarization'],2),round(para_opti['inclination'],2),round(para_opti['spin1z'],2),
        round(para_opti['spin2z'],2)), horizontalalignment='center',verticalalignment='center',fontsize=20)
    ax.text(0.5, 0.80, r'Params_reels : $t_c$ : {}, $m_1$ : {}, $m_2$ : {}, $d_L$ : {}, ra : {}, dec : {}, pola : {}, incl : {}, s1z : {}, s2z : {}.'.format(para_reels[0],
        para_reels[1], para_reels[2], para_reels[3], para_reels[4], para_reels[5],para_reels[6],para_reels[7],para_reels[8],para_reels[9]), horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes,fontsize=20)
    ax.text(0.5, 0.65, r'Params_init : $t_c$ : {}, $m_1$ : {}, $m_2$ : {}, $d_L$ : {}, ra : {}, dec : {}, pola : {}, incl : {}, s1z : {}, s2z : {}.'.format(initial_params[0],
        mass1_init, mass2_init, initial_params[3], initial_params[4], initial_params[5],initial_params[6],initial_params[7],initial_params[8],
        initial_params[9]), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes,fontsize=20)
    ax.text(0.5, 0.35, r'min_m2loglik : {}'.format(para_opti['mloglik']), horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes,fontsize=20)
    ax.text(0.5, 0.20, r'min_m2loglik attendu : {}'.format(log_noise_likelihood_from_SNR), horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes,fontsize=20)

    if save_fig == True:
        plt.savefig("minimisation/Full_Minimisation_chirp_"+ mini +"_signal_noise")




def plot_correlation(params_dataFrame,model,para_reels,cbc_params,save_fig):

    tc_list=params_dataFrame['tc']
    mass1_list=params_dataFrame['mass1']
    mass2_list=params_dataFrame['mass2']
    distance_list=params_dataFrame['distance']
    ra_list=params_dataFrame['ra']
    dec_list=params_dataFrame['dec']
    pola_list=params_dataFrame['polarization']
    incl_list=params_dataFrame['inclination']
    s1z_list=params_dataFrame['spin1z']
    s2z_list=params_dataFrame['spin2z']
    ll_ratio_test=params_dataFrame['mloglik']

    nb_params = len(para_reels)
    fig_3, axs = plt.subplots(nrows=nb_params,ncols=nb_params,figsize=(40,40))


    def plot_grid_correlation(id,data_x,data_y,axs_list,label_x,label_y,param_min,param_max,params_dataFrame):

        model.update(tc=cbc_params['tc'], mass1 = cbc_params['mass1'], mass2 = cbc_params['mass2'],
                    distance = cbc_params['distance'], ra = cbc_params['ra'], dec = cbc_params['dec'])


        index_x = params_dataFrame.columns.get_loc(data_x) - 1
        index_y = params_dataFrame.columns.get_loc(data_y) - 1


        x_grid = np.arange(param_min[index_x],param_max[index_x],echantill[index_x])
        y_grid = np.arange(param_min[index_y],param_max[index_y],echantill[index_y])


        print("Iterations totales : ",len(x_grid)*len(y_grid))
        k=0

        ll_ratio_grid = np.zeros((len(x_grid), len(y_grid)))
        print(ll_ratio_grid.shape)

        for i, x_ in enumerate(x_grid):
            for j, y_ in enumerate(y_grid):
                params = {data_x : x_ ,  data_y : y_} #Les paramètres que l'on souhaite modifier sur le modèle de notre GW
                model.update(**params) #Modification du modèle 
                ll_ratio_grid[i,j] = model.loglr #calcul du likelihood ratio
                k +=1 #Compteur du nombre d'itérations
                print ("Iteration : {}, likelihood : {}".format(k,ll_ratio_grid[i,j]), end="\r")
                #time.sleep(0.1)
        max_index = np.unravel_index(np.argmax(ll_ratio_grid), ll_ratio_grid.shape)

        # Extract corresponding m1 and m2 values
        x_max = x_grid[max_index[0]]
        y_max = y_grid[max_index[1]]

        print("Maximum log-likelihood ratio at:")
        print(data_x + ' = ' + str(round(x_max,2)))
        print(data_y + " = " + str(round(y_max,2)))

        axs_list.imshow(ll_ratio_grid.T,  # Transpose to align axes correctly
                origin='lower',   # Make sure lower m1/m2 is at bottom-left
                extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                aspect='auto',    # Or use 'equal' if square pixels are desired
                cmap='viridis')   # You can change colormap as desired
        axs_list.scatter(x_max, y_max, marker='x', color='red', label='Maximum')
        axs_list.legend()
        axs_list.set_xlabel(label_x)
        axs_list.set_ylabel(label_y)
        #axs_list.axhline(x_max,y_grid[0],y_grid[-1],linestyle = 'dashed',color='r') #, label=label_y + r'$_{,opt}$ = ' + str(round(float(data_tail[data_y]),2)))
        #axs_list.axvline(y_max,x_grid[0],x_grid[-1],linestyle = 'dashed',color='r') #, label=label_x + r'$_{,opt}$ = ' + str(round(float(data_tail[data_x]),2)))

        #axs_list.title('Log-Likelihood Ratio as a Function of Mass 1 and Mass 2')

    #indexes_params = pd.DataFrame(data={'tc': [0], 'mass1': [1],
    #                                     'mass2': [2], 'distance': [3], 'ra' : [4], 'dec' : [5]})

    #ind_fig = 1          2           3              4           5           6               7              8           9           10           11            12        13        14               15             16         17         18             19          20          21              22             23          24          25             26             27              28             29             30             31        32              33             34         35         36              37             38         39         40             41              42              43             44             45
    axs_list = [axs[0,1], axs[0,2],   axs[0,3],      axs[0,4],   axs[0,5],   axs[0,6],       axs[0,7],      axs[0,8],   axs[0,9],   axs[1,2],    axs[1,3],     axs[1,4], axs[1,5], axs[1,6],        axs[1,7],      axs[1,8],  axs[1,9],  axs[2,3],      axs[2,4],   axs[2,5],   axs[2,6],       axs[2,7],      axs[2,8],   axs[2,9],   axs[3,4],      axs[3,5],      axs[3,6],       axs[3,7],      axs[3,8],      axs[3,9],      axs[4,5], axs[4,6],       axs[4,7],      axs[4,8],  axs[4,9],  axs[5,6],       axs[5,7],      axs[5,8],  axs[5,9],  axs[6,7],      axs[6,8],       axs[6,9],       axs[7,8],      axs[7,9],      axs[8,9]]
    x_list = [tc_list,    mass2_list, distance_list, ra_list,    dec_list,   pola_list,      incl_list,     s1z_list,   s2z_list,   mass2_list, distance_list, ra_list,  dec_list,  pola_list,      incl_list,     s1z_list,  s2z_list,  distance_list, ra_list,    dec_list,   pola_list,      incl_list,     s1z_list,   s2z_list,   ra_list,       dec_list,      pola_list,      incl_list,     s1z_list,      s2z_list,      dec_list, pola_list,      incl_list,     s1z_list,  s2z_list,  pola_list,      incl_list,     s1z_list,  s2z_list,  incl_list,     s1z_list,       s2z_list,       s1z_list,      s2z_list,      s2z_list]
    y_list = [mass1_list, mass1_list, mass1_list,    mass1_list, mass1_list, mass1_list,     mass1_list,    mass1_list, mass1_list, tc_list,    tc_list,       tc_list,  tc_list,   tc_list,        tc_list,       tc_list,   tc_list,   mass2_list,    mass2_list, mass2_list, mass2_list,     mass2_list,    mass2_list, mass2_list, distance_list, distance_list, distance_list,  distance_list, distance_list, distance_list, ra_list,  ra_list,        ra_list,       ra_list,   ra_list,   dec_list,       dec_list,      dec_list,  dec_list,  pola_list,     pola_list,      pola_list,      incl_list,     incl_list,     s1z_list]
    label_x = [r'$t_c$',  r'$m_2$',   r'distance',   r'ra',      r'dec',     r'pola',        r'incl',       r's_{1z}',  r's_{2z}',  r'$m_2$',   r'distance',   r'ra',    r'dec',    r'pola',        r'incl',       r's_{1z}', r's_{2z}', r'distance',   r'ra',      r'dec',     r'pola',        r'incl',       r's_{1z}',  r's_{2z}',  r'ra',         r'dec',        r'pola',        r'incl',       r's_{1z}',     r's_{2z}',     r'dec',   r'pola',        r'incl',       r's_{1z}', r's_{2z}', r'pola',        r'incl',       r's_{1z}', r's_{2z}', r'incl',       r's_{1z}',      r's_{2z}',      r's_{1z}',     r's_{2z}',     r's_{2z}']
    label_y = [r'$m_1$',  r'$m_1$',   r'$m_1$',      r'$m_1$',   r'$m_1$',   r'$m_1$',       r'$m_1$',      r'$m_1$',   r'$m_1$',   r'$t_c$',   r'$t_c$',      r'$t_c$', r'$t_c$',  r'$t_c$',       r'$t_c$',      r'$t_c$',  r'$t_c$',  r'$m_2$',      r'$m_2$',   r'$m_2$',   r'$m_2$',       r'$m_2$',      r'$m_2$',   r'$m_2$',   r'distance',   r'distance',   r'distance',    r'distance',   r'distance',   r'distance',   r'ra',    r'ra',          r'ra',         r'ra',     r'ra',     r'dec',         r'dec',        r'dec',    r'dec',    r'pola',       r'pola',        r'pola',        r'incl',       r'incl',       r's_{1z}']
    data_x = ['tc',       'mass2',    'distance',    'ra',       'dec',      'polarization', 'inclination', 'spin1z',   'spin2z',   'mass2',    'distance',    'ra',     'dec',     'polarization', 'inclination', 'spin1z',  'spin2z',  'distance',    'ra',       'dec',      'polarization', 'inclination', 'spin1z',   'spin2z',   'ra',          'dec',         'polarization', 'inclination', 'spin1z',      'spin2z',      'dec',    'polarization', 'inclination', 'spin1z',  'spin2z',  'polarization', 'inclination', 'spin1z',  'spin2z', 'inclination',  'spin1z',       'spin2z',       'spin1z',      'spin2z',      'spin2z']
    data_y = ['mass1',    'mass1',    'mass1',       'mass1',    'mass1',    'mass1',        'mass1',       'mass1',    'mass1',    'tc',       'tc',          'tc',     'tc',      'tc',           'tc',          'tc',      'tc',      'mass2',       'mass2',    'mass2',    'mass2',        'mass2',       'mass2',    'mass2',    'distance',    'distance',    'distance',     'distance',    'distance',    'distance',    'ra',     'ra',           'ra',          'ra',      'ra',      'dec',          'dec',         'dec',     'dec',    'polarization', 'polarization', 'polarization', 'inclination', 'inclination', 'spin1z']
    param_min = [cbc_params['tc']-1.5,cbc_params['mass1']-3,cbc_params['mass2']-3,cbc_params['distance']-3,cbc_params['ra']-1.5,cbc_params['dec']-1.5,cbc_params['polarization']-1.5,cbc_params['inclination']-1.5,cbc_params['spin1z']-1.5,cbc_params['spin2z']-1.5]
    param_max = [cbc_params['tc']+1.5,cbc_params['mass1']+3,cbc_params['mass2']+3,cbc_params['distance']+3,cbc_params['ra']+1.5,cbc_params['dec']+1.5,cbc_params['polarization']+1.5,cbc_params['inclination']+1.5,cbc_params['spin1z']+1.5,cbc_params['spin2z']+1.5]
    echantill = [0.1               ,0.2                   ,0.2                   ,0.2                     ,0.1                 ,0.1                  ,0.1                           ,0.1                          ,0.1                     ,0.1                     ]

    indices_interet = [6]
    #,7,8,9,14,15,16,17,21,22,23,24,27,28,29,30,32,33,34,35,36,37,38,39,40,41,42,43,44,45]
    p=0
    #for i in range(len(axs_list)):
    for i in indices_interet:
        p +=1
        print ("Plots total : {}, plot : {}".format(len(axs_list),p))
        plot_grid_correlation(i,data_x[i],data_y[i],axs_list[i],label_x[i],label_y[i],param_min,param_max,params_dataFrame)

    axs_off = [axs[1,0],axs[2,0],axs[3,0],axs[4,0],axs[5,0],axs[6,0],axs[7,0],axs[8,0],axs[9,0],
            axs[2,1],axs[3,1],axs[3,2],axs[4,1],axs[4,2], axs[4,3],
            axs[5,1],axs[5,2], axs[5,3], axs[5,4],
            axs[6,1],axs[6,2], axs[6,3], axs[6,4],axs[6,5],
            axs[7,1],axs[7,2], axs[7,3], axs[7,4],axs[7,5],axs[7,6],
            axs[8,1],axs[8,2], axs[8,3], axs[8,4],axs[8,5],axs[8,6],axs[8,7],
            axs[9,1],axs[9,2], axs[9,3], axs[9,4],axs[9,5],axs[9,6],axs[9,7],axs[9,8]]

    for i in axs_off:
        i.axis('off')
    fig_3.tight_layout()
    #fig_3.colorbar(label='Log-Likelihood Ratio')

    if save_fig :
        plt.savefig('correlation/Others_params_correlation')


def plot_correlation_2_params(model,cbc_params,x_data,y_data,borne_x,borne_y,ech_x,ech_y,x_label,y_label,save_fig):

    model.update(tc=cbc_params['tc'], mass1 = cbc_params['mass1'], mass2 = cbc_params['mass2'], inclination = 0,
                    distance = cbc_params['distance'], ra = cbc_params['ra'], dec = cbc_params['dec'],
                    polarization = cbc_params['polarization'],declination = cbc_params['declination'],
                    spin1z = cbc_params['spin1z'],spin2z = cbc_params['spin2z'])

    #Params =============================
    Nom = 'V4'
    param_x_name = x_data
    param_y_name = y_data
    # borne_x = 500
    # ech_x = 10
    # borne_y = 3
    # ech_y = 0.2
    #====================================

    x_grid = np.arange(cbc_params[param_x_name] - borne_x, cbc_params[param_x_name] + borne_x, ech_x)
    y_grid = np.arange(cbc_params[param_y_name] - borne_y, cbc_params[param_y_name] + borne_y, ech_y)

    print("Iterations totales : ",len(x_grid)*len(y_grid))
    k=0

    ll_ratio_grid_unit = np.zeros((len(x_grid), len(y_grid)))
    print(ll_ratio_grid_unit.shape)

    for i, x_ in enumerate(x_grid):
        for j, y_ in enumerate(y_grid):
            params = {param_x_name : x_ ,  param_y_name : y_} #Les paramètres que l'on souhaite modifier sur le modèle de notre GW
            model.update(**params) #Modification du modèle 
            ll_ratio_grid_unit[i,j] = model.loglr #calcul du likelihood ratio
            k +=1 #Compteur du nombre d'itérations
            print ("Iteration : {}, likelihood : {}".format(k,ll_ratio_grid_unit[i,j]), end="\r")
            #time.sleep(0.1)

    max_index = np.unravel_index(np.argmax(ll_ratio_grid_unit), ll_ratio_grid_unit.shape)

    # Extract corresponding m1 and m2 values
    x_max = x_grid[max_index[0]]
    y_max = y_grid[max_index[1]]

    print("Maximum log-likelihood ratio at:")
    print(param_x_name + ' = ' + str(round(x_max,2)))
    print(param_y_name + " = " + str(round(y_max,2)))

    plt.figure(figsize=(8, 6))
    plt.imshow(ll_ratio_grid_unit.T,  # Transpose to align axes correctly
            origin='lower',   # Make sure lower m1/m2 is at bottom-left
            extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
            aspect='auto',    # Or use 'equal' if square pixels are desired
            cmap='viridis')   # You can change colormap as desired
    plt.scatter(x_max, y_max, marker='x', color='red', label='Maximum')
    plt.legend()
    plt.colorbar(label='Log-Likelihood Ratio')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if save_fig:
        plt.savefig(Nom + "_" + param_x_name + "_" + param_x_name)


def plot_correlation_chirp_q(model,cbc_params,save_fig):

    model.update(tc=cbc_params['tc'], mass1 = cbc_params['mass1'], mass2 = cbc_params['mass2'], inclination = 0,
                    distance = cbc_params['distance'], ra = cbc_params['ra'], dec = cbc_params['dec'],
                    polarization = cbc_params['polarization'],declination = cbc_params['declination'],
                    spin1z = cbc_params['spin1z'],spin2z = cbc_params['spin2z'])

    mchirp_true = mchirp_from_mass1_mass2(cbc_params["mass1"], cbc_params["mass2"])
    q_true = q_from_mass1_mass2(cbc_params["mass1"], cbc_params["mass2"])


    x_grid = np.arange(mchirp_true-10,mchirp_true+10, 0.2)
    y_grid = np.arange(q_true-1,q_true+1,0.2)

    print("Iterations totales : ",len(x_grid)*len(y_grid))
    k=0

    ll_ratio_grid_unit = np.zeros((len(x_grid), len(y_grid)))
    print(ll_ratio_grid_unit.shape)

    for i, m in enumerate(x_grid):
        for j, q in enumerate(y_grid):
            m1 = mass1_from_mchirp_q(m, q)
            m2 = mass2_from_mchirp_q(m, q)
            model.update(mass1=m1, mass2=m2) #Modification du modèle 
            ll_ratio_grid_unit[i,j] = model.loglr #calcul du likelihood ratio
            k +=1 #Compteur du nombre d'itérations
            print ("Iteration : {}, likelihood : {}".format(k,ll_ratio_grid_unit[i,j]), end="\r")
            #time.sleep(0.1)

    max_index = np.unravel_index(np.argmax(ll_ratio_grid_unit), ll_ratio_grid_unit.shape)

    # Extract corresponding m1 and m2 values
    x_max = x_grid[max_index[0]]
    y_max = y_grid[max_index[1]]

    print("Maximum log-likelihood ratio at:")
    print("Chirp mass = " + str(round(x_max,2)))
    print("Mass ratio = " + str(round(y_max,2)))

    plt.figure(figsize=(8, 6))
    plt.imshow(ll_ratio_grid_unit.T,  # Transpose to align axes correctly
            origin='lower',   # Make sure lower m1/m2 is at bottom-left
            extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
            aspect='auto',    # Or use 'equal' if square pixels are desired
            cmap='viridis')   # You can change colormap as desired
    plt.scatter(x_max, y_max, marker='x', color='red', label='Maximum')
    plt.legend()
    plt.colorbar(label='Log-Likelihood Ratio')
    plt.xlabel('Chirp mass')
    plt.ylabel('Mass ratio')
    plt.title('Log-Likelihood Ratio as a Function of Mass 1 and Mass 2')

    if save_fig:
        plt.savefig('Chirp_mass_Mass_ratio')
