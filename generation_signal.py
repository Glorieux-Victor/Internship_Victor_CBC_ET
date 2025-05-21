import pandas as pd
import numpy as np
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




para_reels = np.array([3.1, 38.6, 29.3, 1000, 1.37, -1.26,2.76,0,0,0])

class Signal_GW:

    cbc_params_stat = {
            'spin1x': 0., 'spin2x': 0.,  'spin1y': 0., 'spin2y': 0.,
            'eccentricity': 0, 'coa_phase': 0}

    def __init__(self,seglen,sample_rate,fmin,cbc_params,approximant):

        N = int(seglen * sample_rate / 2 + 1) # Number of samples in the frequency series

        cbc_params_stat = {
            'spin1x': 0., 'spin2x': 0.,  'spin1y': 0., 'spin2y': 0.,
            'eccentricity': 0, 'coa_phase': 0}

        cbc_params['approximant'] = approximant
        cbc_params_stat['approximant'] = approximant
        #IMRPhenomXAS (modèle plus simple)
        cbc_params['f_lower'] =  fmin
        cbc_params_stat['f_lower'] =  fmin

        # Génération du signal
        self.params = cbc_params
        self.seglen = seglen
        self.stat = cbc_params_stat
        self.fmin = fmin
        self.N = N
    
    def signal_ET(self):
        generator = FDomainDetFrameGenerator(
                        FDomainCBCGenerator, 0., detectors=['E1', 'E2', 'E3'],
                        delta_f=1./self.seglen, **self.params)
        signal = generator.generate()

        return signal


    def signal_LIGO(self):
        generator_H1L1 = FDomainDetFrameGenerator(
                        FDomainCBCGenerator, 0., detectors=['H1', 'L1'],
                        delta_f=1./self.seglen, **self.params)
    
        signal = generator_H1L1.generate()

        return signal
    


    def signal_simple(self,signal):
        psd = EinsteinTelescopeP1600143(self.N, 1./self.seglen, self.fmin)
        psds = {'E1': psd, 'E2': psd, 'E3': psd}
        low_frequency_cutoff = {'E1': self.fmin, 'E2': self.fmin, 'E3': self.fmin}

        model = GaussianNoise(['mass1', 'mass2', 'tc','polarization','ra','dec','inclination','spin1z','spin2z','distance'], signal, low_frequency_cutoff,
                                psds=psds, static_params=self.stat)

        return model
    
    def signal_noise(self,signal):
        psd = EinsteinTelescopeP1600143(self.N, 1./self.seglen, self.fmin)
        psds = {'E1': psd, 'E2': psd, 'E3': psd}
        low_frequency_cutoff = {'E1': self.fmin, 'E2': self.fmin, 'E3': self.fmin}

        # Generate noise and add it to the signal
        noise = frequency_noise_from_psd(psd)
        signal['E1'] = signal['E1'] + noise
        signal['E2'] = signal['E2'] + noise
        signal['E3'] = signal['E3'] + noise

        model = GaussianNoise(['mass1', 'mass2', 'tc','polarization','ra','dec','inclination','spin1z','spin2z','distance'], signal, low_frequency_cutoff,
                                psds=psds, static_params=self.stat)

        return model