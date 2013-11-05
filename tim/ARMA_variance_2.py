from math import *
import numpy as np
from statsmodels.tsa.arima_process import arma_acf

"""
Calculate the variance of an ARMA(1,q) model, where the theta terms are
all = 1/(q+1)

Based on Brockwell & Davis 96 eqn 3.2.3

Modified to have theta_0 = 1/(q+1) instead of = 1
"""

def ARMAvar(q, a):

    if q>0:
        theta = 1./(q+1.)
        psi_jm1 = 1./(q+1.)
        var = psi_jm1**2
        
        for j in range(q):
            psi_j = a*psi_jm1 + theta
            var += psi_j**2
            psi_jm1 = psi_j

    else:
        var = 1

    # Now, sum up terms from q to infinity

    var += 1./(1.-a**2) - (1. - a**(2*(q+1)))/(1.-a**2)

    # var_ratio is ratio of variance to that of AR(1) model

    var_ratio = var*(1.-a**2)


    return (var, var_ratio)

def ACF_ARMA(lambda_p, lambda_avg, lambda_s, x_max):

    lambda_p = float(lambda_p)
    lambda_s = float(lambda_s)
    lambda_avg = float(lambda_avg)

    nma = round(lambda_avg/lambda_s)
    ma_coeff = np.ones((nma))
    lag_max = round(x_max / lambda_s)
    a1 = exp(-lambda_s / lambda_p)
    acf = arma_acf([1,-a1], ma_coeff, nobs=lag_max)
    x = np.arange(0, x_max, lambda_s)

    acfArray = np.empty((len(x), 2))
    acfArray[:,0] = x
    acfArray[:,1] = acf
    return acfArray

"""
Map a CAR(1) process to an AR(1), then average over lambda_avg
and calculate variance

lambda_p - physical length scale of AR process
lambda_s - physical sampling interval of AR process
lambda_avg - physical length over which AR process is to be integrated
"""

def CloudVar(lambda_p, lambda_s, lambda_avg):
    
    # q is MA order of ARMA(1,q)

    lambda_s = float(lambda_s)
    lambda_avg = float(lambda_avg)

    q = int(round(lambda_avg/lambda_s))
    a = exp(-lambda_s / lambda_p)
    
    (var, var_ratio) = ARMAvar(q, a)

    # This variance is a multiple of the variance of the noise driving the
    # AR(1) model.   This variance, in turn, is a multiple of the underlying
    # measurement variance, with the relationship given in Gillespie 96

    var = var * (1. - exp(-2*lambda_s / lambda_p))/2

#    print q, a
    return var

"""
The acf from acfFileName must have been written by the ACF_Write function of SF_ARMA.R
with the same values of lambda_p, lambda_s, lambda_avg

kappa is the average cloud layer extinction in mags
c is the multiplier that gives the sigma for the cloud extinction (0.3 < c < 0.8)
"""

def CloudSF(lambda_p, lambda_avg, lambda_s, kappa, c, acfFileName=None, x_max=1000.0):
    
    if acfFileName:
        acf = np.loadtxt(acfFileName)
    else:
        acf = ACF_ARMA(lambda_p, lambda_avg, lambda_s, x_max)
#
# convert distance x (meters) in the acf to angular scale in arcmin - clouds assumed
# at 10 km
    theta = acf[:,0]/(10000.0*pi/180.0/60.0)

    var = (c*kappa)**2 * CloudVar(lambda_p, lambda_s, lambda_avg)

    return (theta, var*(1 - acf[:,1]))
    
    
"""
Simulate an ARMA(1,q) model where all of the MA coefficients are 1/(q+1)
ts is the sequence
"""
def ARMAsim(q, a, nsamp):
    ts = np.zeros(nsamp)

    noise = np.random.normal(size=nsamp+q)
    ts[0] = 0

    for i in range(1,nsamp):
        ts[i] = a*ts[i-1] + sum(noise[i:(i+q)])/(q+1)

    ratio = (np.std(ts)/np.std(noise[0:nsamp]))**2
    print ratio
    return (ts, np.std(ts)**2)
        
        


    
    
