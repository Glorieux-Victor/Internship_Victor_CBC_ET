"""
This module contains a redefinition of the class GaussianNoise to work on the ET MDC data.
"""

import sys
from pycbc.inference.models import GaussianNoise
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
from IPython.display import clear_output
import numpy as np
from pycbc.conversions import mchirp_from_mass1_mass2, mass1_from_mchirp_q, mass2_from_mchirp_q
from generate_data import generate_frequency_domain_signal
from gwpy.timeseries import TimeSeries

all_cbc_params = ['mass1', 'mass2', 'spin1z', 'spin2z', 'distance', 'polarization', 'inclination', 'tc', 'coa_phase', 'ra', 'dec',\
                 'spin1x', 'spin2x', 'spin1y', 'spin1z', 'approximant', 'f_lower']

default_static_params = {
              'spin1x': 0., 'spin2x': 0.,  'spin1y': 0., 'spin2y': 0.,}

default_variable_params = ['mass1', 'mass2', 'spin1z', 'spin2z', 'distance', 'polarization', 'inclination', 'tc', 'coa_phase', 'ra', 'dec']

default_spinz_bound = (-0.97, 0.97)
default_angle_bound = (0, 2 * np.pi)
default_iota_bound = (0, np.pi)
default_dec_bound = (-np.pi / 2, np.pi /2)

class MDCGaussianNoise(GaussianNoise):
    """
    Represent a model consisting of stationary Gaussian noise following a known PSD and a CBC signal. Inherit from pycbc GaussianNoise class.

    Attributes
    ----------
    data : dict
        Dictionary of pycbc FrequencySeries containing the data from each detector.
    psd : dict
        Dictionary of pycbc FrequencySeries containing the PSD from each detector.
    ifos : list
        List of detectors.
    static_params : dict
        Dictionary containing static parameters (e.g spins along x and y axes).
    maximized_params : dict
        Dictionary containing maximized parameters (only available after running maximize()).
    injection_params : dict, optional.
        Dictionary containing the parameters of the injected CBC signal. May not be set if the signal is unknown.
    optimal_loglikelihood : float
        Log likelihood computed for the true injected CBC signal parameters, if they are provided.
    optimal_loglr : float
        Log likelihood ratio computed for the true injected CBC signal parameters, if they are provided.
    max_loglr : float
        Log likelihood ratio obtained after maximization.
    injection_network_snr : float
        Network SNR of the injected signal, if it is provided.
        
    Methods
    -------
    compute_optimal_likelihood : compute likelihood for the correct values of the injected parameters (assuming they are known)
    maximize : maximize the likelihood using scipy.optimize.differential_evolution
    reconstruct_signal : reconstruct the signal in the frequency domain from the maximized parameters.
    """
    def __init__(self, data, psd, approximant='IMRPhenomXPHM', fmin=5, static_params=default_static_params, variable_params=default_variable_params, injection_params=None, injection_network_snr=None, **kwargs):
        """
        Parameters
        ----------
        data : dict
            Dictionary containing the data frequency series for each detector.
        psd : dict
            Dictionary containing the PSD frequency series for each detector.
        approximant : str, optional
            Waveform approximant to use. Default is IMRPhenomD
        fmin : float, optional.
            Low frequency cutoff. Default is 5 Hz.
        static_params : dict, optional.
            Dictionary of static parameters to use. Default is default_static_params.
        static_params : list, optional.
            List of variable parameters to use. Default is default_variable_params.
        injection_params : dict, optional.
            Dictionary containing true injection parameters.
        injection_network_snr : float, optional.
            Network SNR of the injected signal (if known).
        """

        static_params['approximant'] = approximant
        static_params['f_lower'] = fmin
        ifos = data.keys()
        low_frequency_cutoff = {}
        for ifo in ifos:
            low_frequency_cutoff[ifo] = fmin
        super().__init__(variable_params, data, low_frequency_cutoff, psds=psd, normalize=False, static_params=static_params, **kwargs)
        self.injection_params = injection_params
        self.static_params = static_params
        self.maximized_params = None
        self.data = data
        self.ifos = list(self.data.keys())
        self.epoch = data[self.ifos[0]].epoch
        if injection_params != None:
            self.compute_optimal_likelihoods()
        self.injection_network_snr = injection_network_snr
        self.check_params()
    
    def check_params(self):
        """
        Check that all needed parameters are well defined.
        """
        for parameter in all_cbc_params:
            if parameter not in self.variable_params and parameter not in list(self.static_params.keys()):
                print('Error: ' + parameter + ' is not defined either as a variable or a static parameter')
                sys.exit(1)
            if parameter == 'approximant' or parameter == 'f_lower':
                if parameter not in list(self.static_params.keys()):
                    print('Error: ' + parameter + ' must be defined as a static parameter')

    def compute_optimal_likelihoods(self):
        """
        Compute likelihood for the true parameters of the signal present in the data.
        Need injection_params to be defined. If not, set the optimal likelihoods to -1.
        Returns
        -------
        optimal_loglr : float
            Log likelihood ratio (d, h) - (h, h) / 2.
        optimal_loglikelihood : float
            Log likelihood (d - h, d - h).
        """
        if self.injection_params is None:
            print('True signal parameters unknown. Cannot compute optimal likelihood.')
            self.optimal_loglr = -1
            self.optimal_loglikelihood = -1
            return -1, -1
        
        self.update(**self.injection_params)
        self.optimal_loglr = self.loglr
        self.optimal_loglikelihood = self.loglikelihood
        lognl = self.lognl

        return self.optimal_loglr, self.optimal_loglikelihood

    def maximize(self, bounds, max_iterations=1000, tol=1e-6):
        """
        Maximize the likelihood using scipy.optimize.differential_evolution over the variable parameters defined.

        Parameters
        ----------
        bounds : list of tuples
            Bounds for each parameter, e.g bounds=[(mc_min, mc_max), (q_min, q_max)]. Parameters should be in the same order as the 'variable_params' list.
        max_iterations : int, optional.
            Maximal number of iterations for the 'differential_evolution' algorithm. Default is 1000.
        tol : float, optional.
            Tolerance for the 'differential_evolution' algorithm. Default is 1e-6.
        """

        # Define the objective function to minimize (negative of log-likelihood ratio)
        def negative_loglr(x):
        
            # Unwrap parameters
            params = dict(zip(self.variable_params, x))
            
            # Convert chirp mass and mass ratio into mass1 and mass2
            if 'mass1' in self.variable_params:
                m1 = mass1_from_mchirp_q(params['mass1'], params['mass2'])
                m2 = mass2_from_mchirp_q(params['mass1'], params['mass2'])
                params['mass1'] = m1
                params['mass2'] = m2
            # Update model with the given vector of parameters
            updated_params = {**params, **self.static_params}
            #print(updated_params)
            self.update(**updated_params)
        
            return -self.loglr  # Negate for maximization
        

        # Run the optimization
        iteration_counter = {"count": 0}
        
        def status_callback(x, f=None, accept=False):
            clear_output(wait=True)
            iteration_counter["count"] += 1
            current_value = negative_loglr(x)
            print(f"Iteration {iteration_counter['count']}: negative_loglr = {current_value}")

        print('Start maximization over the following parameters:')
        print(self.variable_params)
        result = differential_evolution(negative_loglr, bounds, callback=status_callback, maxiter=max_iterations, tol=tol, polish=False)
        print('Maximization complete')
        # Extract optimized parameters
        maximized_params = dict(zip(self.variable_params, result.x))
        if 'mass1' in self.variable_params:
            m1 = mass1_from_mchirp_q(maximized_params['mass1'], maximized_params['mass2'])
            m2 = mass2_from_mchirp_q(maximized_params['mass1'], maximized_params['mass2'])
            maximized_params['mass1'] = m1
            maximized_params['mass2'] = m2

        self.maximized_params = maximized_params
        self.maximized_params = {**self.maximized_params, **self.static_params}
        
        max_loglr = -result.fun  # Undo the negation
        self.maxloglr = max_loglr
        
        print(f"Maximum log-likelihood ratio: {max_loglr}")

        return result

    def reconstruct_signal(self):
        """
        Reconstruct the signal in each detector from the set of maximized parameters.

        Returns
        -------
        reconstructed_signal_fdomain : dict
            Dictionary containing a FrequencySeries for each detector.
        reconstructed_signal_tdomain : dict
            Dictionary containing a TimeSeries for each detector.
        """
        if self.maximized_params is not None:
            reconstructed_signal_fdomain = generate_frequency_domain_signal(self.maximized_params, epoch=self.epoch)
        else:
            print('Error: run maximize() method before trying to reconstruct the signal.')

        reconstructed_signal_tdomain = {}
        for ifo in self.ifos:
            reconstructed_signal_tdomain[ifo] = reconstructed_signal_fdomain[ifo].to_timeseries() # Just an inverse FFT
        
        return reconstructed_signal_fdomain, reconstructed_signal_tdomain

def subtract_signal(original_data, reconstructed_signal_tdomain):
    """
    Subtract the reconstructed signal from the original data in the time domain.

    Parameters
    ----------
    original_data : dict
        Dictionary containing the original TimeSeries for each detector.
    reconstructed_signal_tdomain : dict
        Dictionary containing the reconstructed TimeSeries for each detector.

    Returns
    -------
    subtracted_signal_tdomain : dict 
        Dictionary containing the residual TimeSeries for each detector.
    """
    subtracted_signal_tdomain = {}
    ifos = list(original_data.keys())
    for ifo in ifos:
        tsd = original_data[ifo]

        # Interpolate reconstructed signal with time stamps of the original data
        t1 = original_data[ifo].get_sample_times().data
        t2 = reconstructed_signal_tdomain[ifo].get_sample_times().data
        
        h1 = original_data[ifo].data
        h2 = reconstructed_signal_tdomain[ifo].data
        
        h_of_t = interp1d(t2, h2, bounds_error=False, fill_value=0)
        h_new = h_of_t(t1)

        residual = TimeSeries(h1 - h_new, times=t1)
        subtracted_signal_tdomain[ifo] = residual.to_pycbc()

    return subtracted_signal_tdomain
