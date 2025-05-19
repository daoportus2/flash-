# -*- coding: utf-8 -*-
"""
Created on Mon May 19 14:46:17 2025

@author: oportus.d
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 19 14:42:49 2025

@author: oportus.d
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 19 14:10:07 2025

@author: oportus.d
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, k, m_e, c, pi
import numpy as np
from scipy.constants import pi, e,epsilon_0, h, m_e, c, k as k_B
#%% based in https://arxiv.org/pdf/2404.11540
def gaunt_factor_ei(t, Z):
    #t = kb*T[K]/me*c**2
    #Z = charge state
    c0 = 1.0
    cnr = [0.4302, 24.2255e-5, 0.7546e-5, 0.5282, 0.3301, 0.0911]
    cr = [0.55467, 2.6346, -2.277595, 1.1480, -0.36465,
          0.07451, -0.00975, 0.0007885, -3.5841e-5, 6.99834e-7]
    cz = [5.760e4, 3.440, 16.80, 0.1333]

    t = np.clip(t, 1e-6, None)  # Avoid division by zero or log(0)

    term1 = 1 - np.exp(- (cnr[1] * Z**2 / t)**cnr[3])
    term2 = (cnr[0] + cnr[5]) * np.exp(- (t / (cnr[2] * Z**2))**cnr[4])
    fnr = cnr[0] * term1 - term2

    fr = c0 * sum(cr[j] * t**(j+1) for j in range(10))

    top = (Z / 10) * cz[0] * (100 * t * np.sqrt(10 / Z))**cz[1]
    bottom = np.exp(cz[2] * (100 * t * np.sqrt(10 / Z))**cz[3]) - 1
    fz = top / bottom

    gei = c0 * (1 + fnr - fz) + fr
    return gei

def gaunt_factor_ee(t):
    #t = kb*T[K]/me*c**2

    t = np.clip(t, 1e-6, None)  # Avoid log10(0)
    logt = np.log10(t)
    Fee = 0.5 * (np.tanh(0.602 * (logt + 5.06)) + 1)
    FNR = t * (6 * np.sqrt(3) / np.sqrt(2 * np.pi)) * 0.5 * (np.tanh(-2.153 * np.log10(t / 0.43)) + 1)
    poly = 1 + 0.53 * t + 9.48 * t**2 - 0.67 * t**3 + 0.027 * t**4
    gee = Fee * FNR * poly
    return gee

#%%



def radiation_power(n_e, T_e, Z): #based in https://arxiv.org/pdf/2404.11540 pp. 9, eq (61)
    """
    Compute the total radiation power for Maxwellian quasi-neutral plasma.

    Parameters:
    - n_e: electron number density (m^-3)
    - T_e: electron temperature (K)
    - Z: average ion charge
    - t: dimensionless temperature parameter
    - g_ei_func: function g_ei(t, Z)
    - g_ee_func: function g_ee(t)

    Returns:
    - Radiation power P in Watts per cubic meter (W/m^3)
    """
    t = k_B*T_e/(m_e * c**2)
    g_ei_func = gaunt_factor_ei(t, Z)
    g_ee_func = gaunt_factor_ee(t) 
    prefactor = (32 * pi * e**6 * n_e**2) / (3 * (4 * pi * epsilon_0)**3 * h * m_e * c**3)
    sqrt_term = np.sqrt((2 * pi * k_B * T_e) / (3 * m_e))
    gaunt_sum = Z * g_ei_func + g_ee_func
    P = prefactor * sqrt_term * gaunt_sum
    #print('(gei, gee) = ('+str(round(np.max(g_ei_func),2))+', '+str(round(np.max(g_ee_func),2))+')')
    return P


#%%
# =============================================================================
# END
# =============================================================================
