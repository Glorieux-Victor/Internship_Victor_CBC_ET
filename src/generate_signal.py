from pycbc.waveform.generator import (FDomainDetFrameGenerator,FDomainCBCGenerator)
from pycbc.conversions import mchirp_from_mass1_mass2, mass1_from_mchirp_q, mass2_from_mchirp_q, q_from_mass1_mass2, tau0_from_mass1_mass2, chi_eff
from pycbc.psd import EinsteinTelescopeP1600143
from pycbc.noise.gaussian import frequency_noise_from_psd

def generate_fd_signal_from_params(cbc_params, noise=False):

    tau0 = tau0_from_mass1_mass2(cbc_params['mass1'], cbc_params['mass2'], cbc_params['f_lower'])
    print('Tau0 (signal duration): ' + format(tau0, '.2f') + ' s')
    delta_f = 1 / tau0

    # Définition du générateur
    generator = FDomainDetFrameGenerator(
        FDomainCBCGenerator, 0., detectors=['E1', 'E2', 'E3'],
        delta_f=delta_f, **cbc_params)
    
    # Génération du signal
    signal = generator.generate()
    N = len(signal['E1']) # Number of samples in the frequency series

    if noise == True:
    
        psd = EinsteinTelescopeP1600143(N, delta_f, cbc_params['f_lower'])
        psds = {'E1': psd, 'E2': psd, 'E3': psd}
        noise1 = frequency_noise_from_psd(psd)
        noise2 = frequency_noise_from_psd(psd)
        noise3 = frequency_noise_from_psd(psd)
        
        signal['E1'] += noise1
        signal['E2'] += noise2
        signal['E3'] += noise2

    return signal

def generate_time_series_from_frequency_series(frequency_series, tc):
    
    time_series = frequency_series.to_timeseries()
    t_end = time_series.get_sample_times()[-1]
    time_series = time_series.cyclic_time_shift(t_end - tc)

    return time_series