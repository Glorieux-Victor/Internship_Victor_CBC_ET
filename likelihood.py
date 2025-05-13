import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize

def likelihood(data,sigma_mat,NOISE):
    N = len(data)
    part_log = N*np.log(2*np.pi) + np.sum( np.log( np.diag(sigma_mat) ) )
    part_exp = -(1/2)*NOISE*(1/sigma_mat)*NOISE

    loglik = part_log + part_exp

    return - loglik

def minimization(initial_params):
    result = optimize.minimize(likelihood, initial_params, method = 'SLSQP', tol= 10**(-6))

    return result

# constraints = ({'type':'eq', 'fun':m2CONSflat}, {'type':'ineq', 'fun':m2CONSflat_Vintcal2})