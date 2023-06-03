import numpy as np
from scipy.integrate import odeint
from scipy.special import zeta
from scipy import interpolate
import standardmodel as sm
import sid as sid
import constants as c
import os
import pickle
import argparse
import time

# path to the data directory
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, 'data/')

# Load the precomputed thermodynamic quantities and temperature vs scale factor data
try:
    with open("{}/SM_thermodynamic_quantities.pkl".format(data_path), 'rb') as file:
        thermodynamic_results = pickle.load(file)
    hubble_rate_func = interpolate.interp1d(thermodynamic_results['T_domain'], thermodynamic_results['hubble_rate'])
    entropy_density_func = interpolate.interp1d(thermodynamic_results['T_domain'], thermodynamic_results['entropy_density'])
    energy_density_func = interpolate.interp1d(thermodynamic_results['T_domain'], thermodynamic_results['energy_density'])
    pressure_func = interpolate.interp1d(thermodynamic_results['T_domain'], thermodynamic_results['pressure'])
    d_rho_dT_func = interpolate.interp1d(thermodynamic_results['T_domain'], thermodynamic_results['drho_dT'])

    with open("{}/T_SM_vs_a.pkl".format(data_path), 'rb') as file:
        T_SM_vs_a_results = pickle.load(file)
    T_SM_func = interpolate.interp1d(T_SM_vs_a_results['a_domain'], T_SM_vs_a_results['T_domain_entropy'])
except Exception as e:
    print(f"WARNING: scattering coefficient data files not found. Error message: {str(e)}")

def compute_T_DS_vs_a(a_domain, T_DS_initial, ms1, ms2, theta, flavor, merle_simplification=False, antineutrino=False):
    """Compute DS temperature vs scale factor using
    the covariant energy conservation equation"""
    T_DS_domain = odeint(compute_dT_DS_da, T_DS_initial, a_domain,
                    args=(ms1, ms2, theta, flavor, merle_simplification, antineutrino))
    
    return T_DS_domain.flatten()

def compute_dT_DS_da(T_DS, a, ms1, ms2, theta, flavor, 
                     merle_simplification=False, antineutrino=False):
    """Compute the rate of change of the dark sector temperature with scale factor
    - T_SM_func is a function that takes in a scale factor and returns the SM temperature
    - hubble_rate_func is a function that takes in T_SM and returns the hubble rate
    """
    drho_dT = 2*(sm.compute_drho_single_dT(T_DS, ms1, 1) + sm.compute_drho_single_dT(T_DS, ms2, 1))
    T_SM = T_SM_func(a)
    drho_inj_dt = sid.compute_drho_injected_dt(T_SM, ms1, theta, flavor, merle_simplification, antineutrino)
    # Assumes 2 DOFS for each sterile neutrino
    rho = 2*sm.compute_energy_density(T_DS, ms1, 1) + 2*sm.compute_energy_density(T_DS, ms2, 1)
    pressure = 2*sm.compute_pressure(T_DS, ms1, 1) + 2*sm.compute_pressure(T_DS, ms2, 1)
    dT_dt = (-3*hubble_rate_func(T_SM)*(rho + pressure) + drho_inj_dt)/drho_dT

    return dT_dt/(a*hubble_rate_func(T_SM))

def compute_current_DM_ndens(T_SM, T_DS):
    """Compute the dark matter relic number density today, given T_SM and T_DS
    at any point in time. Assumes:
    - SID has finished and the other sterile neutrino has already annihilated
    - The dark matter relic sterile is still relativistic 
    - The actual entropic degrees of freedom at that point in time are well-decribed by the
    approximate number
    """
    scale_factor_ratio = (c.Tcmb/T_SM)*(sm.compute_SM_relativistic_dof_approx(c.Tcmb)/sm.compute_SM_relativistic_dof_approx(T_SM))**(1/3.0)
    T_DS_today = T_DS*scale_factor_ratio
    # Assumes 2 DOFS for the sterile neutrino
    ndens_today = 2*(3./2)*zeta(3)*(T_DS_today)**3/(2*np.pi**2)

    return ndens_today

def compute_omegasq_DM(T_SM, T_DS, m_relic):
    """Compute the dark matter relic energy density today. Assumes:
    - SID has finished and the other sterile neutrino has already annihilated
    - The dark matter relic sterile is still relativistic
    """
    ndens_today = compute_current_DM_ndens(T_SM, T_DS)
    rho_today = m_relic*ndens_today
    omegahsq_DM = rho_to_omegahsq(rho_today)
    
    return omegahsq_DM

def rho_to_omegahsq(rho):
    """Input energy density in MeV^4. Return \Omega h^2"""
    return 8*np.pi*c.grav*rho/(3*c.H100**2)

def logspace_args(arg):
    return np.logspace(float(arg[0]), float(arg[1]), int(arg[2]))

def main(args):
    # convert input arguments. Make sure that T_SM_domain is largest to smallest
    T_SM_domain = np.sort(logspace_args(args.T_SM_domain))[::-1]
    sinsq2theta_values = logspace_args(args.sinsq2theta)
    theta_values = 0.5*np.sqrt(sinsq2theta_values)
    ms1_values = logspace_args(args.ms1)
    ms2_values = logspace_args(args.ms2)

    # convert to scale factor. 
    T_SM_to_a = interpolate.interp1d(T_SM_vs_a_results['T_domain_entropy'], T_SM_vs_a_results['a_domain'])
    a_domain = T_SM_to_a(T_SM_domain)

    # Initialize an empty array to store the results
    results = np.zeros((len(theta_values), len(ms1_values), len(ms2_values), len(a_domain)))

    # Nested loops to go through all combinations of theta, ms1, and ms2
    start_time = time.time()
    for i, theta in enumerate(theta_values):
        for j, ms1 in enumerate(ms1_values):
            for k, ms2 in enumerate(ms2_values):
                # Print progress
                print("Computing for theta = {}, ms1 = {}, ms2 = {}".format(theta, ms1, ms2))
                # Compute for each combination and store in the results array
                results[i, j, k] = compute_T_DS_vs_a(a_domain, args.T_DS_initial, ms1, ms2, theta, 
                                                     args.flavor, args.merle_simplification, args.antineutrino)
                print("Time elapsed: {} seconds".format(time.time() - start_time))


    # Save the data to a file
    data = {'T_SM_domain': T_SM_domain, 'sinsq2theta_values': sinsq2theta_values, 'ms1_values': ms1_values,
            'ms2_values': ms2_values, 'a_domain': a_domain, 'flavor': args.flavor,
            'merle_simplification': args.merle_simplification, 'antineutrino': args.antineutrino,
            'T_DS_initial': args.T_DS_initial, 'T_DS_domain': results}
    #save using pickle
    with open(os.path.join(dir_path, 'results/', args.output_file + '.pkl'), 'wb') as file:
       pickle.dump(data, file)


# Save a grid of runs over 3D [ms1, ms2, theta] space to a file. 
# Sample usage:
# python darksector.py --T_SM_domain 0.5 4 1000 --T_DS_initial 10 --ms1 0 2 7 --ms2 -2 -1 1 --sinsq2theta -13 -10  10 --flavor 'electron' --output_file sparse_grid --merle_simplification --antineutrino
# python darksector.py --T_SM_domain 0.5 4 10 --T_DS_initial 1 --ms1 0 2 1 --ms2 -2 -1 1 --sinsq2theta -13 -10  1 --flavor 'electron' --output_file test --merle_simplification --antineutrino 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--T_SM_domain', nargs='+', type=float, required=True)
    parser.add_argument('--T_DS_initial', type=float, required=True)
    parser.add_argument('--sinsq2theta', nargs='+', type=float, required=True)
    parser.add_argument('--ms1', nargs='+', type=float, required=True)
    parser.add_argument('--ms2', nargs='+', type=float, required=True)
    parser.add_argument('--flavor', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--merle_simplification', action='store_true')
    parser.add_argument('--antineutrino', action='store_true')
    args = parser.parse_args()
    main(args) 