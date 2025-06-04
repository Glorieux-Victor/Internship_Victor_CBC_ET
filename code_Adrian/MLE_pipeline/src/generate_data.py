from pycbc.waveform.generator import (FDomainDetFrameGenerator,FDomainCBCGenerator)
from pycbc.conversions import tau0_from_mass1_mass2
default_cbc_params = {
              # Paramètres intrinsèques à la source
              'mass1': 38.6,
              'mass2': 29.3,
              'spin1x': 0., 'spin2x': 0.,  'spin1y': 0., 'spin2y': 0.,  'spin1z': 0, 'spin2z': 0, 
              'eccentricity': 0,
              # Paramètres extrinsèques
              'ra': 1.37, 'dec': -1.26, 'distance': 10000, 
              'polarization': 2.76, 'inclination': 0,
              'tc': 3.1 , 'coa_phase': 0.3,
              'approximant': 'IMRPhenomD',
              'f_lower': 5
                }

def generate_frequency_domain_signal(cbc_params=default_cbc_params, ifos=['E1', 'E2', 'E3'], with_noise=False, epoch=0):
    """
    Generate a CBC signal in the frequency domain for a given list of detectors.

    Parameters
    ----------
    cbc_params : dict, optional.
        Dictionary containing signal parameters. Default is default_cbc_params.
    ifos : list, optional.
        List of detectors. Default is E1, E2, E3.
    with_noise : bool, optional.
        Add Gaussian noise from ET10km_columns.txt PSD file. Not implemented yet. Default is False.
    """
    tau0 = tau0_from_mass1_mass2(cbc_params['mass1'], cbc_params['mass2'], cbc_params['f_lower'])
    segment_duration = 1
    while segment_duration < tau0 + 2: segment_duration *=2

    print('Waveform approximate duration: ' + format(tau0, '.1f') + 's')
    print('Segment duration: ' + format(segment_duration, '.1f') + 's')
    
    generator = FDomainDetFrameGenerator(
    FDomainCBCGenerator, epoch, detectors=ifos,
    delta_f=1./segment_duration, **cbc_params)

    # Génération du signal
    signal = generator.generate()        
    
    return signal


