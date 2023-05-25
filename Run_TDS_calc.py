import numpy as np
from matplotlib import pyplot as plt
from scipy.special import zeta
from scipy import integrate, interpolate, misc
import pandas as pd
import time
import pickle
from scipy.interpolate import interp1d, RegularGridInterpolator
import equations_and_constants as ec
from scipy.integrate import quad
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.ticker as ticker

def compute_rho_DS(T_DS, ms1, ms2):
    return 2*ec.compute_energy_density(T_DS, ms1, 1) + 2*ec.compute_energy_density(T_DS, ms2, 1)

def compute_pressure_DS(T_DS, ms1, ms2):
    return 2*ec.compute_pressure(T_DS, ms1, 1) + 2*ec.compute_pressure(T_DS, ms2, 1)

def compute_drho_injected_dt(T_SM, ms, theta, flavor, merle_simplification=True, antineutrino=False):
    y = float(ms)/T_SM
    def integrand(x):
        return ec.SID_rate(x*T_SM, T_SM, theta, ms, flavor, antineutrino, 
                        merle_simplification)*x**2*np.sqrt(x**2 + y**2)/(np.exp(np.sqrt(x**2 + y**2)) + 1)
    result, err = quad(integrand, 0.1, 20)
    return T_SM**4*result/(2*np.pi**2)

def compute_dT_DS_da(a, T_DS, ms1, ms2, theta, flavor, merle_simplification=True, antineutrino=False):
    T_SM = ec.T_SM_vs_a_conserving_entropy(a)
    drho_dT = 2*(ec.compute_drho_single_dT(T_DS, ms1, 1) + ec.compute_drho_single_dT(T_DS, ms2, 1))
    drho_inj_dt = compute_drho_injected_dt(T_SM, ms1, theta, flavor, merle_simplification, antineutrino)
    rho = compute_rho_DS(T_DS, ms1, ms2)
    pressure = compute_pressure_DS(T_DS, ms1, ms2)
    dT_dt = (-3*ec.Hubble_rate(T_SM)*(rho + pressure) + drho_inj_dt)/drho_dT
    return dT_dt/(a*ec.Hubble_rate(T_SM))

def compute_T_DS_vs_a(a_domain, T_DS_initial, ms1, ms2, theta, flavor, merle_simplification=True, antineutrino=False):
    """Compute (and optionally save) DS temperature vs scale factor using
    the covariant energy conservation equation"""
    T_SM_domain = ec.T_SM_vs_a_conserving_entropy(a_domain)
    print("starting standard model temp is {} MeV".format(T_SM_domain[0]))
    
    T_DS_domain = odeint(compute_dT_DS_da, T_DS_initial, a_domain, tfirst=True,
                    args=(ms1, ms2, theta, flavor, merle_simplification, antineutrino))
    T_DS_domain = T_DS_domain.flatten()

    return a_domain, T_DS_domain, T_SM_domain

def compute_current_DM_rho(T_SM, T_DS, m_relic):
    """Compute the dark matter relic density today. Assumes:
    - SID has finished and the other sterile neutrino has already annihilated
    - The dark matter relic sterile is still relativistic
    """
    scale_factor_ratio = (ec.Tcmb/T_SM)*(ec.compute_SM_relativistic_dof_approx(ec.Tcmb)/ec.compute_SM_relativistic_dof_approx(T_SM))**(1/3.0)
    
    # Multiplying by two to include antineutrinos
    ndens_today = 2*(3./2)*zeta(3)*(T_DS*scale_factor_ratio)**3/(2*np.pi**2)
    rho_today = m_relic*ndens_today
    omegahsq_DM = ec.rho_to_omegahsq(rho_today)
    
    return omegahsq_DM

ms1_array = np.logspace(0, 2, 7)
sinsq2theta_array = np.logspace(-10, -13, 10)
ms2_array = np.logspace(-3, -1, 5)
flavor='electron'
a_domain = np.logspace(2.032, 6, 1000)

a_domain_array = []
T_DS_array = []
T_SM_array = []
omegahsq_array = []
ms1_saved = []
sinsq2theta_saved = []
ms2_saved = []
start = time.time()
for ms1 in ms1_array:
    for ms2 in ms2_array:
        for sinsq2theta in sinsq2theta_array:
            print(ms1, sinsq2theta, ms2, start-time.time())
            theta = np.sqrt(sinsq2theta)/2.
            T_DS_initial = ms1/10.
            a_domain, T_DS, T_SM = compute_T_DS_vs_a(a_domain, T_DS_initial, ms1, ms2, theta, flavor, True, False)
            omegahsq = compute_current_DM_rho(T_SM[-1], T_DS[-1], ms2)
            a_domain_array.append(a_domain)
            T_DS_array.append(T_DS)
            T_SM_array.append(T_SM)
            omegahsq_array.append(omegahsq)
            ms1_saved.append(ms1)
            sinsq2theta_saved.append(sinsq2theta)
            ms2_saved.append(ms2)
    
data_dict = {'a_domain': a_domain_array, 'T_DS': T_DS_array, 'T_SM': T_SM_array, 
            'omegahsq': omegahsq_array, 'ms1': ms1_saved, 'ms2': ms2_saved,
            'sinsq2theta': sinsq2theta_saved}
# save the dictionary to a pickle file
with open('results/ds_temperature_dict.pkl', 'wb') as f:
    pickle.dump(data_dict, f)