from pycbc.inference.models import GaussianNoise
from scipy.optimize import differential_evolution
from IPython.display import clear_output
import numpy as np
from pycbc.conversions import mchirp_from_mass1_mass2, mass1_from_mchirp_q, mass2_from_mchirp_q
from generate_data import generate_frequency_domain_signal

default_static_params = {
              # Paramètres intrinsèques à la source
              'spin1x': 0., 'spin2x': 0.,  'spin1y': 0., 'spin2y': 0., 
              'eccentricity': 0 }

default_variable_params = ['mass1', 'mass2', 'spin1z', 'spin2z', 'distance', 'polarization', 'inclination', 'tc', 'coa_phase', 'ra', 'dec']

default_chirp_mass_bound = (1, 100)
default_q_bound = (1, 100)
default_spinz_bound = (-0.97, 0.97)
default_angle_bound = (0, 2 * np.pi)
default_iota_bound = (0, np.pi)
default_dec_bound = (-np.pi / 2, np.pi /2)
default_distance_bound = (100, 100000)

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
    Methods
    -------
    compute_optimal_likelihood : compute likelihood for the correct values of the injected parameters (assuming they are known)
    maximize : maximize the likelihood using scipy.optimize.differential_evolution
    reconstruct_signal : reconstruct the signal in the frequency domain from the maximized parameters.
    """
    def __init__(self, data, psd, approximant='IMRPhenomD', fmin=5, static_params=default_static_params, variable_params=default_variable_params, injection_params=None, **kwargs):
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
        self.ifos = self.data.keys()

        self.compute_optimal_likelihoods()
        
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

    def maximize(self, max_iterations=10000, tol=1e-6, bounds_method='default', initial_guess_method='random', custom_bounds=None, custom_initial_guess=None):
        """
        Maximize the likelihood using scipy.optimize.differential_evolution over 11 parameters:
        Mc, q, s1z, s2z, ra, dec, dist, iota, psi, tc, phi_c.
        """

        # Define the objective function to minimize (negative of log-likelihood ratio)
        def negative_loglr(x):
        
            # Unwrap parameters
            mc, q, s1z, s2z, ra, dec, dist, iota, psi, tc, phi_c = x
        
            # Convert chirp mass and mass ratio into mass1 and mass2
            m1 = mass1_from_mchirp_q(mc, q)
            m2 = mass2_from_mchirp_q(mc, q)
        
            # Update model with the given vector of parameters
            self.update(mass1=m1, mass2=m2, spin1z=s1z, spin2z=s2z, ra=ra, dec=dec, distance=dist, inclination=iota, polarization=psi, tc=tc, coa_phase=phi_c)
        
            return -self.loglr  # Negate for maximization
        
        # Mc, q, chi_eff, ra, dec, dist, iota, psi, tc, phi_c

        # Manage bounds 
        if bounds_method == 'default':
            bounds = [default_chirp_mass_bound, default_q_bound, default_spinz_bound, default_spinz_bound, default_angle_bound, default_dec_bound, \
                      default_distance_bound, default_iota_bound, default_angle_bound, tc_bound, default_angle_bound]

        elif bounds_method == 'custom':
            if custom_bounds is not None:
                bounds = custom_bounds
            else:
                print('Error: please define a custom bounds list')
        elif bounds_method == 'sensible':
            if self.injection_params is not None:
                bounds = get_sensible_bounds(self.injection_params)
            else:
                print('Error: cannot use sensible bounds if injection params are not set')

        # Manage initial guess
        if initial_guess_method == 'random':
            initial_guess = None # No initial guess given, will be selected randomly by the function within the bounds
        elif initial_guess_method == 'custom':
            if custom_initial_guess is not None:
                initial_guess = custom_initial_guess
            else:
                print('Error: please define a custom initial guess')
        else:
            initial_guess = None
        # Run the optimization
        iteration_counter = {"count": 0}
        
        def status_callback(x, f=None, accept=False):
            clear_output(wait=True)
            iteration_counter["count"] += 1
            current_value = negative_loglr(x)
            print(f"Iteration {iteration_counter['count']}: negative_loglr = {current_value}")

        print('Start maximization')
        
        result = differential_evolution(negative_loglr, bounds, x0=initial_guess, callback=status_callback, maxiter=max_iterations, tol=tol)
        
        # Extract optimized parameters
        mc_opt, q_opt, s1z_opt, s2z_opt, ra_opt, dec_opt, dist_opt, iota_opt, psi_opt, tc_opt, phic_opt = result.x

        self.maximized_params = {'mass1' : mass1_from_mchirp_q(mc_opt, q_opt),
                            'mass2' : mass2_from_mchirp_q(mc_opt, q_opt),
                            'spin1z' : s1z_opt,
                            'spin2z' : s2z_opt,
                            'ra' : ra_opt,
                            'dec' : dec_opt,
                            'distance' : dist_opt,
                            'inclination' : iota_opt,
                            'polarization' : psi_opt,
                            'tc' : tc_opt, 
                            'coa_phase' : phic_opt}
        self.maximized_params = {**self.maximized_params, **self.static_params}
        
        max_loglr = -result.fun  # Undo the negation
        self.maxloglr = max_loglr
        print(f"Optimized chirp mass: {mc_opt}")
        print(f"Optimized mass ratio: {q_opt}")
        
        print(f"Maximum log-likelihood ratio: {max_loglr}")

        return result

    def reconstruct_signal(self):

        if self.maximized_params is not None:
            reconstructed_signal = generate_frequency_domain_signal(self.maximized_params)
        else:
            print('Error: run maximize() method before trying to reconstruct the signal.')
        return reconstructed_signal

## Utility functions

def get_sensible_bounds(injection_params, tol=0.1):
    """
    Generate parameter bounds for likelihood maximization with a relative tolerance radius of the true signal parameters.
    Example: if the true chirp mass is 30 and tol=0.1, the bound in chirp mass will be (0.27, 0.3).

    Parameters
    ----------
    injection_params : dict
        Dictionary containing the parameters of the injected signal.
    tol : float
        Tolerance radius to create the bounds.

    Returns
    -------
    bounds : list
        List of tuples containing bounds to be given to differential_evolution().
    """
    true_Mc = mchirp_from_mass1_mass2(injection_params['mass1'], injection_params['mass2'])
    true_q = injection_params['mass1'] / injection_params['mass2']
    
    chirp_mass_bound = (true_Mc * (1 - tol), true_Mc * (1 + tol))
    q_bound = (true_q * (1 - tol), true_q * (1 + tol))

    s1z_bound = (injection_params['spin1z'] * (1 - tol), injection_params['spin1z'] * (1 + tol))
    s2z_bound = (injection_params['spin2z'] * (1 - tol), injection_params['spin2z'] * (1 + tol))
    distance_bound =  (injection_params['distance'] * (1 - tol), injection_params['distance'] * (1 + tol))
    tc_bound =  (injection_params['tc'] * (1 - tol), injection_params['tc'] * (1 + tol))

    bounds = [chirp_mass_bound, q_bound, s1z_bound, s2z_bound, default_angle_bound, default_dec_bound, \
              distance_bound, default_iota_bound, default_angle_bound, tc_bound, default_angle_bound]
    
    return bounds
    

    
