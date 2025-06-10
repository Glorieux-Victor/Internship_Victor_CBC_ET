import pandas as pd
import numpy as np
import time
from IPython.display import clear_output
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

from scipy.optimize import basinhopping,differential_evolution
from scipy.optimize import minimize

#====================================================
#fonctions : minimisation_globale, minimisation_unique_param_glob (pas finie), minimisation_locale, print_results, likelihood_visualisation
#====================================================


params_dataFrame_glob = pd.DataFrame(data={'mloglik': [],'tc': [], 'mass1': [],
                                        'mass2': [], 'distance': [], 'ra' : [], 'dec' : [],
                                        'polarization': [], 'inclination': [], 'spin1z' : [], 'spin2z' : [],
                                         'coa_phase': [], 'chirp' : [], 'q' : []})

p=0
k=0

def minimisation_globale(model,initial_params,bounds,minimisation,method,tol,nb_iter,maxiter,stepsize,log_noise_likelihood_from_SNR,normalisation,save_data,file_name):

    print('Expected log likelihood noise: {:.2f}'.format(log_noise_likelihood_from_SNR))

    global params_dataFrame_glob, k, p

    params_dataFrame_glob = pd.DataFrame(data={'mloglik': [],'tc': [], 'mass1': [],
                                        'mass2': [], 'distance': [], 'ra' : [], 'dec' : [],
                                        'polarization': [], 'inclination': [], 'spin1z' : [], 'spin2z' : [],
                                        'coa_phase': [], 'chirp' : [], 'q' : []})
    k=0
    p=0

    def likelihood_calculation_glob(params_):

        params=params_

        if normalisation :
            param_min = np.array([0,     1, 0.2,    200,   0,  -np.pi,   0,   0,   -1,   -1])
            step = np.array([   1.5,    5,   0.2,     5,   3,   3,   3,   2,  0.1,    0.1])
            for i in range(len(params_)):
                params[i] = param_min[i] + params_[i]*(step[i]/0.1)


        mass1 = mass1_from_mchirp_q(mchirp=params[1],q=params[2])
        mass2 = mass2_from_mchirp_q(mchirp=params[1],q=params[2])

        cbc_params = {
            # Paramètres intrinsèques à la source
            'mass1': mass1,
            'mass2': mass2,
            'spin1x': 0., 'spin2x': 0.,  'spin1y': 0., 'spin2y': 0.,  'spin1z': params[8], 'spin2z': params[9],
            'eccentricity': 0,
            # Paramètres extrinsèques
            'ra': params[4], 'dec': params[5], 'distance': params[3],
            'polarization': params[6], 'inclination': params[7],
            'tc': params[0] , 'coa_phase': params[10]}

        global params_dataFrame_glob, k
        model.update(tc=params[0],mass1=mass1,mass2=mass2,distance=params[3],ra=params[4],dec=params[5],
                    polarization=params[6],inclination=params[7],spin1z=params[8],spin2z=params[9],coa_phase=params[10],)


        if not hasattr(model, "_loglr_cache"):
            model._loglr_cache = {}

        def rounded_key(params, decimals=2):
            return tuple((k, round(v, decimals)) for k, v in sorted(params.items()))

        key = rounded_key(cbc_params)

        if key not in model._loglr_cache:
            try:
                model.loglr
            except Exception as e:
                print(f"Erreur dans loglr : {e}")
                return 1e100
            model._loglr_cache[key] = model.loglr

        mloglik = - model._loglr_cache[key]

        if save_data :
            add = pd.DataFrame(data={'mloglik': mloglik, 'tc': params[0], 'mass1': mass1,
                                  'mass2': mass2, 'distance': params[3], 'ra' : params[4],
                                  'dec' : params[5], 'polarization': params[6], 'inclination': params[7],
                                  'spin1z' : params[8], 'spin2z' : params[9], 'coa_phase': params[10],
                                  'chirp' : params[1], 'q' : params[2]},index=[k])
            params_dataFrame_glob = pd.concat([params_dataFrame_glob,add])
            k +=1

        print (mloglik,end="\r")

        return mloglik


    def negative_loglr(x):

        variable_params = ['tc','mass1', 'mass2','distance', 'ra', 'dec', 'polarization', 'inclination','spin1z', 'spin2z', 'coa_phase']
        static_params = {'spin1x': 0., 'spin2x': 0.,  'spin1y': 0., 'spin2y': 0.,'approximant' : 'IMRPhenomXPHM', 'f_lower' : 5.}
        # Unwrap parameters
        params = dict(zip(variable_params, x))
        
        # Convert chirp mass and mass ratio into mass1 and mass2
        if 'mass1' in variable_params:
            m1 = mass1_from_mchirp_q(params['mass1'], params['mass2'])
            m2 = mass2_from_mchirp_q(params['mass1'], params['mass2'])
            params['mass1'] = m1
            params['mass2'] = m2
        # Update model with the given vector of parameters
        updated_params = {**params, **static_params}
        #print(updated_params)
        model.update(**updated_params)
        mloglr = -model.loglr
        #print(mloglr,end="\r")
    
        return mloglr  # Negate for maximization


    #réels : m1=38.6, m2=29.3
    mass1_init = 30
    mass2_init = 30
    mchirp = mchirp_from_mass1_mass2(mass1_init,mass2_init)
    q = q_from_mass1_mass2(mass1_init,mass2_init)
    #true params : (tc=3.1, chirp_mass, q, dist = 1000, ra = 1.37, dec = -1.26, pola=2.76, incl = 0, s1z=0, s2z=0)
    #initial_params = [2, mchirp, q,   5000,       0,      0,    0,       0,    0,       0]

    #Bounds ==========================
    if normalisation:
        param_min = np.array([0,     1,   0.2,   200,     0,  -np.pi,    0,   0,   -1,   -1])
        param_max = np.array([10,  50,     5, 10000,   2*np.pi,   np.pi,  2*np.pi,   np.pi,    1,    1])
        borne_min_norm = np.zeros(len(param_min))
        borne_max_norm = borne_min_norm
        step = np.array([   1.5,    5,   0.2,     5,     3,   3,    3,   2,  0.1,    0.1])
        bornes_inf_sup_scaled=()
        for i in range(len(param_min)):
            delta = param_max[i] - param_min[i]
            points = delta/step[i] #nombre de points
            bornes_inf_sup_scaled += ((0,points*0.1),) #bornes à donner.
        bounds=bornes_inf_sup_scaled
    else :
        bounds=bounds

    #Nelder-Mead,  Powell, L-BFGS-B
    # 'tol' : 10e-2
    minimizer_kwargs={ "method": method,"bounds":bounds,'tol':tol}

    def print_fun(x, f, accepted):
            clear_output(wait=True)
            global p
            p+=1
            #print("at minimum %.4f accepted %d" % (f, int(accepted)),end="\r")
            if int(accepted) == 1:
                print("min : {}, it : {}".format(f,p))
    
    def print_fun_DE(x, f):
            clear_output(wait=True)
            global p
            p+=1
            #print("at minimum %.4f accepted %d" % (f, int(accepted)),end="\r")
            print("min : {}, it : {}".format(f,p))


    iteration_counter = {"count": 0}
    def status_callback(x, f, accept=False):
        clear_output(wait=True)
        iteration_counter["count"] += 1
        current_value = negative_loglr(x)
        print(f"Iteration {iteration_counter['count']}: negative_loglr = {current_value}")


    if minimisation == 'basinhopping':
        result_glob = basinhopping(likelihood_calculation_glob, x0=initial_params, minimizer_kwargs=minimizer_kwargs,niter = nb_iter,stepsize=stepsize,callback=print_fun)

    elif minimisation == 'differential_evolution':
        result_glob = differential_evolution(likelihood_calculation_glob,bounds,x0=initial_params,maxiter=maxiter,tol=tol,callback=status_callback)

    #params_glob_dataFrame_file_chirp.txt
    if save_data :
        params_dataFrame_glob.to_csv("/home/victor/Internship_Victor_CBC_ET/data_files/"+ file_name ,index=False)

    return result_glob



#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================

params_dataFrame_glob_unique = pd.DataFrame(data={'mloglik': [],'param': []})


def minimisation_unique_param_glob(param,true_value,model,method,tol,nb_iter,stepsize,log_noise_likelihood_from_SNR,save_data,file_name):

    #liste des paramètres, utile pour remonter aux indices.
    list_params = ['tc','chrip','q','distance','ra','dec','polarization','inclination','spin1z','spin2z']

    global params_dataFrame_glob_unique, k, p

    params_dataFrame_glob_unique = pd.DataFrame(data={'mloglik': [],'param': []})
    k=0
    p=0
    
    def likelihood_calculation_glob(params):

        mass1 = mass1_from_mchirp_q(mchirp=params[1],q=params[2])
        mass2 = mass2_from_mchirp_q(mchirp=params[1],q=params[2])

        global params_dataFrame_glob_unique, k

        cbc_params_unique = {
            # Paramètres intrinsèques à la source
            'mass1': mass1,
            'mass2': mass2,
            'spin1x': 0., 'spin2x': 0.,  'spin1y': 0., 'spin2y': 0.,  'spin1z': params[8], 'spin2z': params[9],
            'eccentricity': 0,
            # Paramètres extrinsèques
            'ra': params[4], 'dec': params[5], 'distance': params[3],
            'polarization': params[6], 'inclination': params[7],
            'tc': params[0] , 'coa_phase': 0}

        model.update(tc=params[0],mass1=mass1,mass2=mass2,distance=params[3],ra=params[4],dec=params[5],
                    polarization=params[6],inclination=params[7],spin1z=params[8],spin2z=params[9])
        
        if not hasattr(model, "_loglr_cache"):
            model._loglr_cache = {}

        def rounded_key(params, decimals=3):
            return tuple((k, round(v, decimals)) for k, v in sorted(params.items()))

        key = rounded_key(cbc_params_unique)

        if key not in model._loglr_cache:
            try:
                model.loglr
            except Exception as e:
                print(f"Erreur dans loglr : {e}")
                return 1e100
            model._loglr_cache[key] = model.loglr

        mloglik = - model._loglr_cache[key]

        if save_data :
            add = pd.DataFrame(data={'mloglik': mloglik, 'param': params[0]})
            params_dataFrame_glob_unique = pd.concat([params_dataFrame_glob_unique,add])
            k +=1

        print (mloglik, end="\r")

        return mloglik

    mass1_init = 30
    mass2_init = 30
    mchirp = mchirp_from_mass1_mass2(mass1_init,mass2_init)
    q = q_from_mass1_mass2(mass1_init,mass2_init)

    initial_params = [2, mchirp, q,   5000,       0,      0,    0,       0,    0,       0]


    #On récupère les bonne contraintes pour le paramètres considéré.
    pi2 = np.pi/2
    dpi = 2*np.pi
    bounds_list=((0,10),(1,50),(0.1,5),(10,10000),(0,dpi),(-pi2,pi2),(0,dpi),(0,np.pi),(-1,1),(-1,1))
    bound = bounds_list(list_params.index(param))

    #Liste des méthodes utiles : Nelder-Mead,  Powell, L-BFGS-B
    minimizer_kwargs={ "method": method,"bounds":bound,'tol':tol}
    def print_fun(x, f, accepted): #fonction qui renvoie un message à chaque fin d'itération (de minimisation locale)
            global p
            p+=1
            print("at minimum %.4f accepted %d" % (f, int(accepted)),end="\r")
            if int(accepted) == 1:
                print("min : {}, it : {}".format(f,p))


    if save_data :
        params_dataFrame_glob_unique.to_csv("data_files/"+ file_name ,index=False)

    result = basinhopping(likelihood_calculation_glob, x0=initial_params, minimizer_kwargs=minimizer_kwargs,niter = nb_iter,stepsize=stepsize,callback=print_fun)

    return result




#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================




params_dataFrame = pd.DataFrame(data={'mloglik': [],'tc': [], 'mass1': [],
                                        'mass2': [], 'distance': [], 'ra' : [], 'dec' : [],
                                        'polarization': [], 'inclination': [], 'spin1z' : [], 'spin2z' : [],
                                        'chirp' : [], 'q' : []})


def minimisation_locale(model,initial_params,tol,log_noise_likelihood_from_SNR,follow_lik,save_data):

    print('Expected log likelihood noise: {:.2f}'.format(log_noise_likelihood_from_SNR))

    global params_dataFrame

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
                                    'spin1z' : params[8], 'spin2z' : params[9], 'chirp' : params[1],
                                    'q' : params[2]},index=[k])
            params_dataFrame = pd.concat([params_dataFrame,add])
            k +=1
        if follow_lik :
            print (mloglik, end="\r")

        return mloglik


    #true params : (tc=3.1, chirp_mass, q, dist = 1000, ra = 1.37, dec = -1.26, pola=2.76, incl = 0, s1z=0, s2z=0)

    #Nelder-Mead
    bounds=((0,10),(1,500),(0.1,20),(10,10000),(0,7.5),(-4,4),(0,7.5),(0,4),(-1,1),(-1,1))
    result = minimize(likelihood_calculation, method = 'L-BFGS-B', bounds=bounds, x0=initial_params, tol=tol)

    if save_data :
        params_dataFrame.to_csv("data_files/params_dataFrame_file_chirp_noise.txt",index=False)

    return result, initial_params


#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================


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


#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================


def likelihood_visualisation(model,params_dataFrame_glob,para_reels,fig_name,save_fig):
    fig_lik, axs = plt.subplots(nrows=3, ncols=4, figsize = (40,20))

    true_param = para_reels


    def plot_lik(ax,data_x,label_x,true_param,param_min,param_max,echantill,params_dataFrame):

        model.update(tc=true_param[0], mass1 = true_param[1], mass2 = true_param[2], distance = true_param[3],
                    ra = true_param[4], dec = true_param[5], polarization = true_param[6], inclination = true_param[7],
                    spin1z = true_param[8], spin2z = true_param[9])
        
        dico = {'tc' : true_param[0], 'mass1' : true_param[1], 'mass2' : true_param[2], 'distance' : true_param[3],
                'ra' : true_param[4], 'dec' : true_param[5], 'polarization' : true_param[6], 'inclination' : true_param[7],
                'spin1z' : true_param[8], 'spin2z' : true_param[9]}
        
        mchirp_true = mchirp_from_mass1_mass2(true_param[1],true_param[2])
        q_true = q_from_mass1_mass2(true_param[1],true_param[2])

        index_x = params_dataFrame.columns.get_loc(data_x) - 1
        x_grid = np.arange(param_min[index_x],param_max[index_x],echantill[index_x])
        y_grid = np.zeros(len(x_grid))
        for i, x_ in enumerate(x_grid):
            if data_x == 'mass1' :
                mass1 = mass1_from_mchirp_q(mchirp=x_,q=q_true)
                mass2 = mass2_from_mchirp_q(mchirp=x_,q=q_true)
                params = {'mass1' : mass1, 'mass2' : mass2}
                dico.update(params)
                model.update(**dico)
                y_grid[i]=-model.loglr
            elif data_x == 'mass2' :
                mass1 = mass1_from_mchirp_q(mchirp=mchirp_true,q=x_)
                mass2 = mass2_from_mchirp_q(mchirp=mchirp_true,q=x_)
                params = {'mass1' : mass1, 'mass2' : mass2}
                dico.update(params)
                model.update(**dico)
                y_grid[i]=-model.loglr
            else :
                params = {data_x : x_} #Les paramètres que l'on souhaite modifier sur le modèle de notre GW
                dico.update(params)
                model.update(**dico) #Modification du modèle 
                y_grid[i]=-model.loglr
        ax.plot(x_grid,y_grid,label = r"-log($\mathcal{L}$)")
        if data_x == 'mass1' :
            ax.set_xlabel('M_chirp')
            ax.axvline(mchirp_true,color = 'red',label = 'True param')
        elif data_x == 'mass2' :
            ax.set_xlabel('q')
            ax.axvline(q_true,color = 'red',label = 'True param')
        else :
            ax.set_xlabel(label_x)
            ax.axvline(true_param[index_x],color = 'red',label = 'True param')
            ax.legend()

    axs_list = [axs[0,0],axs[0,1],axs[0,2],axs[0,3],axs[1,0],axs[1,1],axs[1,2],axs[1,3],axs[2,0],axs[2,1]]
    label_x = [r'$t_c$', r'$m_1$', r'$m_2$',  r'distance',   r'ra',      r'dec',     r'pola',        r'incl',       r's_{1z}',  r's_{2z}']
    data_x = ['tc',   'mass1',    'mass2',    'distance',    'ra',       'dec',      'polarization', 'inclination', 'spin1z',   'spin2z']
    param_min = [0,     1,   0.2,     200,        0,   -np.pi,        0,       0,    -1,   -1]
    param_max = [10,  50,      5,   10000,  2*np.pi,    np.pi,  2*np.pi,   np.pi,     1,    1]
    echantill = [0.1,   0.2, 0.05,     50,     0.05,     0.05,     0.05,    0.05,  0.01, 0.01]

    q=0
    for i in range(len(label_x)):
        plot_lik(axs_list[i],data_x[i],label_x[i],true_param,param_min,param_max,echantill,params_dataFrame_glob)
        q += 1
        print('graph {} fini'.format(q),end='\r')
    
    fig_lik.tight_layout()

    if save_fig :
        plt.savefig(fig_name)