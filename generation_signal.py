import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
import matplotlib.pyplot as plt
from gwpy.plot import Plot
from pycbc import psd as pypsd
from pycbc.inference.models import GaussianNoise
from pycbc.conversions import mchirp_from_mass1_mass2, q_from_mass1_mass2, mass1_from_mchirp_q, mass2_from_mchirp_q, tau0_from_mass1_mass2,snr_from_loglr
from fonctions import puissance_seglen
from pycbc.noise.gaussian import frequency_noise_from_psd
from pycbc.waveform.generator import (FDomainDetFrameGenerator,FDomainCBCGenerator)
from pycbc.psd import EinsteinTelescopeP1600143

#comments : génération de signal LIGO pas fonctionnelle, juste besoin de l'adapter de la même façon que la génération de signal ET.


para_reels = np.array([3.1, 38.6, 29.3, 1000, 1.37, -1.26,2.76,0,0,0])

class Signal_GW:

    # cbc_params_stat = {
    #         'spin1x': 0., 'spin2x': 0.,  'spin1y': 0., 'spin2y': 0.,
    #         'eccentricity': 0, 'coa_phase': 0}

    def __init__(self,seglen,sample_rate,fmin,cbc_params,approximant):

        N = int(seglen * sample_rate / 2 + 1) # Number of samples in the frequency series

        cbc_params_stat = {
            'spin1x': 0., 'spin2x': 0.,  'spin1y': 0., 'spin2y': 0.,
            'eccentricity': 0, 'coa_phase': 0}

        cbc_params_var = [
            'mass1','mass2',  'spin1z', 'spin2z', 'ra', 'dec', 'distance',
            'polarization', 'inclination', 'tc']
        
        #cbc_params['approximant'] = approximant
        cbc_params_stat['approximant'] = approximant
        #IMRPhenomXAS (modèle plus simple)
        #cbc_params['f_lower'] =  fmin
        cbc_params_stat['f_lower'] =  fmin

        # Paramètres du signal
        self.tc = cbc_params['tc']
        self.m1 = cbc_params['mass1']
        self.m2 = cbc_params['mass2']
        self.s1 = cbc_params['spin1z']
        self.s2 = cbc_params['spin2z']
        self.ra = cbc_params['ra']
        self.dec = cbc_params['dec']
        self.dist = cbc_params['distance']
        self.pola = cbc_params['polarization']
        self.incl = cbc_params['inclination']

        self.params = cbc_params
        self.seglen = seglen
        self.var = cbc_params_var
        self.stat = cbc_params_stat
        self.fmin = fmin
        self.N = N
    
    def signal_ET(self):

        generator = FDomainDetFrameGenerator(
                        FDomainCBCGenerator, 0., detectors=['E1', 'E2', 'E3'], variable_args=self.var,
                        delta_f=1./self.seglen, **self.stat)
        dico = {
            'mass1': self.m1, 'mass2': self.m2,  'spin1z': self.s1, 'spin2z': self.s1,
            'ra': self.ra, 'dec': self.dec, 'distance': self.dist,
            'polarization': self.pola, 'inclination': self.incl,
            'tc': self.tc }
        signal = generator.generate(**dico)

        return signal


    def signal_LIGO(self):
        generator_H1L1 = FDomainDetFrameGenerator(
                        FDomainCBCGenerator, 0., detectors=['H1', 'L1'],
                        delta_f=1./self.seglen, **self.cbc_params_stat)
    
        signal = generator_H1L1.generate()

        return signal



    def signal_simple(self,signal):

        signal_modif = copy.deepcopy(signal) #Nécessité de copier pour éviter de modifier signal (l'égalité simple alloue seulement un autre chemin d'accès à la variable)

        psd = EinsteinTelescopeP1600143(self.N, 1./self.seglen, self.fmin)
        psds = {'E1': psd, 'E2': psd, 'E3': psd}
        low_frequency_cutoff = {'E1': self.fmin, 'E2': self.fmin, 'E3': self.fmin}

        model = GaussianNoise(self.var, signal_modif, low_frequency_cutoff,
                                psds=psds, static_params=self.stat)

        return model, signal_modif
    
    def signal_noise(self,signal):

        signal_modif = copy.deepcopy(signal)

        psd = EinsteinTelescopeP1600143(self.N, 1./self.seglen, self.fmin)
        psds = {'E1': psd, 'E2': psd, 'E3': psd}
        low_frequency_cutoff = {'E1': self.fmin, 'E2': self.fmin, 'E3': self.fmin}

        # Generate noise and add it to the signal
        noise = frequency_noise_from_psd(psd)
        signal_modif['E1'] = signal_modif['E1'] + noise
        signal_modif['E2'] = signal_modif['E2'] + noise
        signal_modif['E3'] = signal_modif['E3'] + noise
        
        model = GaussianNoise(['mass1', 'mass2', 'tc','polarization','ra','dec','inclination','spin1z','spin2z','distance'], signal_modif, low_frequency_cutoff,
                                psds=psds, static_params=self.stat)

        return model, signal_modif


def generate_time_series_from_frequency_series(frequency_series, tc):

    time_series = frequency_series.to_timeseries()
    t_end = time_series.get_sample_times()[-1]
    time_series = time_series.cyclic_time_shift(t_end - tc)

    return time_series



def generation_signal_GW(cbc_params,sample_rate,fmin,noise,print_snr):

    #mieux en prenant une puissance de 2 la plus proche
    seglen = round(tau0_from_mass1_mass2(cbc_params['mass1'],cbc_params['mass2'],fmin))
    seglen = puissance_seglen(seglen) #converti seglen en puissance de 2 la plus proche : permet de matcher les ranges des listes pour le signal et le bruit par exemple.

    approximant='IMRPhenomD'

    signal = Signal_GW(seglen,sample_rate,fmin,cbc_params,approximant)

    signalGW_ET = signal.signal_ET()

    if noise :
        model, signal_ = signal.signal_noise(signalGW_ET)
        model.update(**cbc_params)
        #print('{:.2f}'.format(model_noise.loglr))
    else :
        model, signal_ = signal.signal_simple(signalGW_ET)
        model.update(**cbc_params)
        #print('{:.2f}'.format(model_normal.loglr))

    snr_E1_sq = model.det_optimal_snrsq('E1')
    snr_E2_sq = model.det_optimal_snrsq('E2')
    snr_E3_sq = model.det_optimal_snrsq('E3')

    log_noise_likelihood_from_SNR = -0.5 * (snr_E1_sq + snr_E2_sq + snr_E3_sq)

    if print_snr :
        print('SNR E1: {:.2f}'.format(snr_E1_sq**0.5))
        print('SNR E2: {:.2f}'.format(snr_E2_sq**0.5))
        print('SNR E3: {:.2f}'.format(snr_E3_sq**0.5))

        model._current_wfs = None #Force le recalcul de la waveform lorsque l'on calcule loglr
        print('Expected minus loglr: {:.2f}'.format(log_noise_likelihood_from_SNR))
        print('loglr :',model.loglr)

    return model, log_noise_likelihood_from_SNR
