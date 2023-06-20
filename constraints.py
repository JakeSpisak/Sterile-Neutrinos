import numpy as np
import constants as c
from scipy import integrate
import darksector as ds

def compute_effective_free_streaming_mass(T_DS, T_SM, ms):
    """Compute the effective free streaming mass of the dark matter sterile neutrino"""
    return ms*(T_SM/T_DS)


# Decay rates for sterile neutrino decay into SM particles, taken from 
# Fuller, Kishimoto, and Kusenko. 
decay_rate_3nu = lambda ms, theta: c.Gf**2*ms**5*np.sin(theta)**2/(192*np.pi**3)
decay_rate_nu_photon = lambda ms, theta: 9*c.Gf**2*c.fine_structure*ms**5*np.sin(theta)**2/(512*np.pi**4)
decay_rate_nu_electron_positron = lambda ms, theta: decay_rate_3nu(ms, theta)/3.
decay_rate_nu_mu_antimu = lambda ms, theta: decay_rate_nu_electron_positron(ms, theta)
decay_rate_nu_pi0 = lambda ms, theta: c.Gf**2*c.f_pion_decay**2*ms*(ms**2-c.m_pi0**2)*np.sin(theta)**2/(16*np.pi)
decay_rate_pipm_e = lambda ms, theta: 2*c.Gf**2*c.f_pion_decay**2*ms*((ms**2-(c.m_pipm+c.m_e)**2)*(ms**2-(c.m_pipm-c.m_e)**2))**(1/2.)*np.sin(theta)**2/(16*np.pi)
decay_rate_pipm_mu = lambda ms, theta: 2*c.Gf**2*c.f_pion_decay**2*ms*((ms**2-(c.m_pipm+c.m_mu)**2)*(ms**2-(c.m_pipm-c.m_mu)**2))**(1/2.)*np.sin(theta)**2/(16*np.pi)

def total_decay_rate(ms, theta):
    rate = decay_rate_3nu(ms, theta) + decay_rate_nu_photon(ms, theta)
    # Approximation:  e+/e- and mu+/mu- decay are in the relativistic limit,
    # even when the sterile neutrino mass isn't much larger than 2x the lepton mass
    # Fine because these are subdominant decay channels
    if ms > 2*c.m_e:
        rate += decay_rate_nu_electron_positron(ms, theta)
    if ms > 2*c.m_mu:
        rate += decay_rate_nu_mu_antimu(ms, theta)
    # Pion decay channels should properly incorporate the non-relativistic case
    if ms > c.m_pi0:
        rate += decay_rate_nu_pi0(ms, theta)
    if ms > c.m_pipm + c.m_e:
        rate += decay_rate_pipm_e(ms, theta)
    if ms > c.m_pipm + c.m_mu:
        rate += decay_rate_pipm_mu(ms, theta)
    return rate

total_decay_rate = np.vectorize(total_decay_rate)

def compute_elapsed_time(a_start, a_end):
    """Compute the time elapsed between two scale factors, in seconds"""
    integrand = lambda a: 1/(a*ds.hubble_rate_func(ds.T_SM_func(a))*c.MeVtoHz)
    result, err = integrate.quad(integrand, a_start, a_end)
    return result

