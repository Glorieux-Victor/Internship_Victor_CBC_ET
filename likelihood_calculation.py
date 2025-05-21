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

from scipy.optimize import basinhopping
from scipy.optimize import minimize

p=0
k=0

def minimisation_globale(model,method,tol,log_noise_likelihood_from_SNR,save_data):

    print('Expected log likelihood noise: {:.2f}'.format(log_noise_likelihood_from_SNR))

    params_dataFrame_glob = pd.DataFrame(data={'mloglik': [],'tc': [], 'mass1': [],
                                        'mass2': [], 'distance': [], 'ra' : [], 'dec' : [],
                                        'polarization': [], 'inclination': [], 'spin1z' : [], 'spin2z' : [],
                                        'chirp' : [], 'q' : []})
    k=0
    p=0

    def likelihood_calculation_glob(params):

        mass1 = mass1_from_mchirp_q(mchirp=params[1],q=params[2])
        mass2 = mass2_from_mchirp_q(mchirp=params[1],q=params[2])

        global params_dataFrame_glob, k
        model.update(tc=params[0],mass1=mass1,mass2=mass2,distance=params[3],ra=params[4],dec=params[5],
                    polarization=params[6],inclination=params[7],spin1z=params[8],spin2z=params[9])
        mloglik = - model.loglr

        # add = pd.DataFrame(data={'mloglik': mloglik, 'tc': params[0], 'mass1': mass1,
        #                          'mass2': mass2, 'distance': params[3], 'ra' : params[4],
        #                          'dec' : params[5], 'polarization': params[6], 'inclination': params[7],
        #                          'spin1z' : params[8], 'spin2z' : params[9], 'chirp' : params[1],
        #                         'q' : params[2]},index=[k])
        # params_dataFrame_glob = pd.concat([params_dataFrame_glob,add])
        # k +=1

        print (mloglik, end="\r")
        #time.sleep(0.1)

        return mloglik

    #réels : m1=38.6, m2=29.3
    mass1_init = 30
    mass2_init = 30
    mchirp = mchirp_from_mass1_mass2(mass1_init,mass2_init)
    q = q_from_mass1_mass2(mass1_init,mass2_init)
    #true params : (tc=3.1, chirp_mass, q, dist = 1000, ra = 1.37, dec = -1.26, pola=2.76, incl = 0, s1z=0, s2z=0)
    initial_params = [2, mchirp, q,   5000,       4,      0,    4,       2,    0,       0]

    #Bounds ==========================
    bounds=((0,10),(1,500),(0.1,20),(10,10000),(0,7.5),(-4,4),(0,7.5),(0,4),(-1,1),(-1,1))

    #Nelder-Mead,  Powell, L-BFGS-B
    # 'tol' : 10e-2
    minimizer_kwargs={ "method": method,"bounds":bounds,'tol':tol}
    def print_fun(x, f, accepted):
            global p
            p+=1
            print("at minimum %.4f accepted %d" % (f, int(accepted)),end="\r")
            if int(accepted) == 1:
                print("min : {}, it : {}".format(f,p))
    result_glob = basinhopping(likelihood_calculation_glob, x0=initial_params, minimizer_kwargs=minimizer_kwargs,niter = 100,stepsize=1,callback=print_fun)

    if save_data :
        params_dataFrame_glob.to_csv("data_files/params_glob_dataFrame_file_chirp_NMmethod.txt")

    return result_glob






def minimisation_locale(model,initial_params,save_data,follow_lik):

    params_dataFrame = pd.DataFrame(data={'mloglik': [],'tc': [], 'mass1': [],
                                     'mass2': [], 'distance': [], 'ra' : [], 'dec' : [],
                                     'polarization': [], 'inclination': [], 'spin1z' : [], 'spin2z' : []})
    k=0

    def likelihood_calculation(params):

        global params_dataFrame, k

        mass1 = mass1_from_mchirp_q(mchirp=params[1],q=params[2])
        mass2 = mass2_from_mchirp_q(mchirp=params[1],q=params[2])

        model.update(tc=params[0],mass1=mass1,mass2=mass2,distance=params[3],ra=params[4],dec=params[5],
                    polarization=params[6],inclination=params[7],spin1z=params[8],spin2z=params[9])
        mloglik = - model.loglr
        if save_data :
            add = pd.DataFrame(data={'mloglik': mloglik, 'tc': params[0], 'mass1': mass1,
                                'mass2': mass2, 'distance': params[3], 'ra' : params[4],
                                'dec' : params[5], 'polarization': params[6], 'inclination': params[7],
                                'spin1z' : params[8], 'spin2z' : params[9]},index=[k])
            params_dataFrame = pd.concat([params_dataFrame,add])
        k +=1
        if follow_lik :
            print (mloglik, end="\r")

        return mloglik


    #true params : (tc=3.1, chirp_mass, q, dist = 1000, ra = 1.37, dec = -1.26, pola=2.76, incl = 0, s1z=0, s2z=0)

    #Nelder-Mead
    bounds=((0,10),(1,500),(0.1,20),(10,10000),(0,7.5),(-4,4),(0,7.5),(0,4),(-1,1),(-1,1))
    result = minimize(likelihood_calculation, method = 'L-BFGS-B', bounds=bounds, x0=initial_params, tol=10)

    if save_data :
        params_dataFrame.to_csv("data_files/params_dataFrame_file_chirp_noise.txt")

    return result, initial_params

def print_results(result,para_reels,initial_params):
    para_opti = result.x
    mass1_init = round(mass1_from_mchirp_q(mchirp=initial_params[1],q=initial_params[2]),2)
    mass2_init = round(mass2_from_mchirp_q(mchirp=initial_params[1],q=initial_params[2]),2)
    mass1_opti = round(mass1_from_mchirp_q(mchirp=para_opti[1],q=para_opti[2]),2)
    mass2_opti = round(mass2_from_mchirp_q(mchirp=para_opti[1],q=para_opti[2]),2)
    para_opti_rd = [round(i,2) for i in para_opti]
    print('Paramètres initiaux :')
    print(r'$t_c$ : {}, $m_1$ : {}, $m_2$ : {}, $d_L$ : {}, ra : {}, dec : {}, pola : {}, incl : {}, s1z : {}, s2z : {}.'.format(initial_params[0],
        mass1_init, mass2_init, initial_params[3], initial_params[4], initial_params[5],initial_params[6],initial_params[7],initial_params[8],initial_params[9]))
    print('\nParamètres d\'optimisation trouvés (local) :')
    print(r'$t_c$ : {}, $m_1$ : {}, $m_2$ : {}, $d_L$ : {}, ra : {}, dec : {}, pola : {}, incl : {}, s1z : {}, s2z : {}.'.format(para_opti_rd[0],
        mass1_opti, mass2_opti, para_opti_rd[3], para_opti_rd[4], para_opti_rd[5],para_opti_rd[6],para_opti_rd[7],para_opti_rd[8],para_opti_rd[9]))
    print('\nParamètres réels :')
    print(r'$t_c$ : {}, $m_1$ : {}, $m_2$ : {}, $d_L$ : {}, ra : {}, dec : {}, pola : {}, incl : {}, s1z : {}, s2z : {}.'.format(para_reels[0],
        para_reels[1], para_reels[2], para_reels[3], para_reels[4], para_reels[5],para_reels[6],para_reels[7],para_reels[8],para_reels[9]))


# #Calcul de - likelihood_ratio
# def likelihood_calculation_glob(params):

#     mass1 = mass1_from_mchirp_q(mchirp=params[1],q=params[2])
#     mass2 = mass2_from_mchirp_q(mchirp=params[1],q=params[2])

#     global params_dataFrame_glob, k
#     model.update(tc=params[0],mass1=mass1,mass2=mass2,distance=params[3],ra=params[4],dec=params[5],
#                  polarization=params[6],inclination=params[7],spin1z=params[8],spin2z=params[9])
#     mloglik = - model.loglr

#     add = pd.DataFrame(data={'mloglik': mloglik, 'tc': params[0], 'mass1': mass1,
#                              'mass2': mass2, 'distance': params[3], 'ra' : params[4],
#                              'dec' : params[5], 'polarization': params[6], 'inclination': params[7],
#                              'spin1z' : params[8], 'spin2z' : params[9], 'chirp' : params[1],
#                             'q' : params[2]},index=[k])
#     params_dataFrame_glob = pd.concat([params_dataFrame_glob,add])
#     k +=1

#     print (mloglik, end="\r")
#     time.sleep(0.1)

#     return mloglik
