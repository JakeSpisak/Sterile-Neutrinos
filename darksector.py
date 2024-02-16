import numpy as np
from scipy.integrate import odeint, cumtrapz
from scipy.special import zeta
from scipy import interpolate, optimize
import standardmodel as sm
import sid as sid
import constants as c
import os
import pickle
import argparse
import time
import sys

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
    T_SM_to_a = interpolate.interp1d(T_SM_vs_a_results['T_domain_entropy'], T_SM_vs_a_results['a_domain'])
except Exception as e:
    print(f"WARNING: scattering coefficient data files not found. Error message: {str(e)}")

def compute_T_DS_vs_loga(a_domain, T_DS_initial, ms1, ms2, theta, flavor, antineutrino=False, simplify=False):
    """Compute DS temperature vs log scale factor using
    the covariant energy conservation equation"""
    T_DS_domain = odeint(compute_dT_DS_dloga, T_DS_initial, np.log(a_domain),
                    args=(ms1, ms2, theta, flavor, antineutrino, simplify))
    
    return T_DS_domain.flatten()

def compute_dT_DS_dloga(T_DS, log_a, ms1, ms2, theta, flavor, antineutrino=False, simplify=False):
    """Compute the rate of change of the dark sector temperature with log scale factor
    - T_SM_func is a function that takes in a scale factor and returns the SM temperature
    - hubble_rate_func is a function that takes in T_SM and returns the hubble rate
    """
    drho_dT = 2*(sm.compute_drho_single_dT(T_DS, ms1, 1) + sm.compute_drho_single_dT(T_DS, ms2, 1))
    T_SM = T_SM_func(np.exp(log_a))
    drho_inj_dt = sid.compute_drho_injected_dt(T_SM, ms1, theta, flavor, antineutrino, simplify)
    # Assumes 2 DOFS for each sterile neutrino
    rho = 2*sm.compute_energy_density(T_DS, ms1, 1) + 2*sm.compute_energy_density(T_DS, ms2, 1)
    pressure = 2*sm.compute_pressure(T_DS, ms1, 1) + 2*sm.compute_pressure(T_DS, ms2, 1)
    dT_dt = (-3*hubble_rate_func(T_SM)*(rho + pressure) + drho_inj_dt)/drho_dT

    return dT_dt/(hubble_rate_func(T_SM))

def compute_drho_DS_comoving_dloga(a, ms1, theta, flavor, antineutrino=False, simplify=False):
    """Compute the rate of change of the dark sector energy density with log scale factor
    - T_SM_func is a function that takes in a scale factor and returns the SM temperature
    - hubble_rate_func is a function that takes in T_SM and returns the hubble rate
    - Approximates that rho_DS is always relativistic
    """
    T_SM = T_SM_func(a)
    drho_inj_dt = sid.compute_drho_injected_dt(T_SM, ms1, theta, flavor, antineutrino, simplify)

    return drho_inj_dt*a**4/hubble_rate_func(T_SM)

def rho_DS_solve(a_domain, ms1, theta, flavor, antineutrino=False, simplify=False):
    """Compute DS rho vs log scale factor using
    the covariant energy conservation equation. Assumes that rho_DS is always relativistic"""
    rate = compute_drho_DS_comoving_dloga(a_domain, ms1, theta, flavor, antineutrino, simplify)
    result = cumtrapz(rate, np.log(a_domain), initial=0)
    return result/a_domain**4

def compute_current_DM_ndens(T_SM, T_DS):
    """Compute the dark matter relic number density today, given T_SM and T_DS
    at any point in time. Assumes:
    - SID has finished and the other sterile neutrino has already annihilated
    - The dark matter relic sterile is still relativistic 
    - The actual entropic degrees of freedom at that point in time are well-decribed by the
    approximate number
    """
    scale_factor_ratio = T_SM_to_a(T_SM)/T_SM_to_a(2) # Ratio from T_SM to 2 MeV 
    scale_factor_ratio *= c.Tcnub/2 # Ratio from 2 MeV to today
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

def main_temperature(args):
    # convert input arguments. Make sure that T_SM_domain is largest to smallest
    sinsq2theta_values = logspace_args(args.sinsq2theta)
    theta_values = 0.5*np.sqrt(sinsq2theta_values)
    ms1_values = logspace_args(args.ms1)
    ms2_values = logspace_args(args.ms2)
    T_SM_domain = logspace_args(args.T_SM_domain)

    # convert to scale factor. 
    a_domain = T_SM_to_a(T_SM_domain)

    # Initialize an empty array to store the results
    results = np.zeros((len(theta_values), len(ms1_values), len(a_domain)))

    # Nested loops to go through all combinations of theta, ms1, and ms2
    start_time = time.time()
    for i, theta in enumerate(theta_values):
        for j, ms1 in enumerate(ms1_values):
            for k, ms2 in enumerate(ms2_values):
                # Print progress
                print("Computing for theta = {}, ms1 = {}, ms2 = {}".format(theta, ms1, ms2))
                # Compute for each combination and store in the results array
                results[i, j, k] = compute_T_DS_vs_loga(a_domain, args.T_DS_initial, ms1, ms2, theta, 
                                                        args.flavor, args.antineutrino, args.simplify)
                print("Time elapsed: {} seconds".format(time.time() - start_time))


    # Save the data to a file
    data = {'T_SM_domain': T_SM_domain, 'sinsq2theta_values': sinsq2theta_values, 'ms1_values': ms1_values,
            'ms2_values': ms2_values, 'a_domain': a_domain, 'flavor': args.flavor,
            'simplify': args.simplify, 'antineutrino': args.antineutrino,
            'T_DS_initial': args.T_DS_initial, 'T_DS_domain': results}
    #save using pickle
    with open(os.path.join(dir_path, 'results/', args.output_file + '.pkl'), 'wb') as file:
       pickle.dump(data, file)

def find_T_DS_from_rho(rho, ms1, ms2):
    guess = (240*rho/(4*7*np.pi**2))**(1/4.) # 4 relativistic DOFs
    guess_2dof = (240*rho/(2*7*np.pi**2))**(1/4.)
    xmin, xmax = 0.1, 2
    
    if guess >= xmax*ms1:
        result = guess
        return guess, guess
    
    fun = lambda T: np.abs(2*sm.compute_energy_density(T, ms1, 1) + 2*sm.compute_energy_density(T, ms2, 1) - rho)
    if guess <= xmax*ms1 and guess >= xmin*ms1:
        result = optimize.minimize(fun, 1.05*guess)
        return result.x[0], guess

    if guess <= xmin*ms1 and guess >= xmax*ms2:
        return guess_2dof, guess_2dof # 2 relativistic DOFs

    if guess <= xmax*ms2 and guess_2dof > ms2:
        result = optimize.minimize(fun, guess_2dof)
        return result.x[0], guess
    else:
        return 0, 0 # if the temperature is below ms2: can't 
    
def gamma_s_2to2(p, temp, beta, Gphi_over_Gf):
    """Compute the approximate 2 to 2 scattering rate for the dark sector"""
    return beta*0.03*(Gphi_over_Gf*c.Gf)**2*temp**4*p

def gamma_s_2to4(p, temp, beta, Gphi_over_Gf, phase_space_ratio=1):
    """Compute the approximate 2 to 4 scattering rate for the dark sector"""
    return phase_space_ratio*(Gphi_over_Gf*c.Gf)**2*temp**4*gamma_s_2to2(beta, Gphi_over_Gf, temp, p)

def sterile_decay_rate(ms, Gphi_over_Gf, numerical_factor=1):
    """Sterile neutrino decay into three light steriles. The numerical factor accounts for the difference from 3 neutrino decay"""
    return numerical_factor*(Gphi_over_Gf*c.Gf)**2*ms**5/(1024*np.pi**3)

def main_rho(args):
    # convert input arguments. Make sure that T_SM_domain is largest to smallest
    sinsq2theta_values = logspace_args(args.sinsq2theta)
    theta_values = 0.5*np.sqrt(sinsq2theta_values)
    ms1_values = logspace_args(args.ms1)

    # Nested loops to go through all combinations of theta, ms1
    results = np.zeros((len(theta_values), len(ms1_values), 500))
    T_SM_domain = np.zeros((len(theta_values), len(ms1_values), 500))
    start_time = time.time()
    for i, theta in enumerate(theta_values):
        for j, ms1 in enumerate(ms1_values):
                # Create the domain
                Ti = min(10*sid.Tmax(ms1), 149*10**3)
                Tfinal_integral = max(1, 0.02*sid.Tmax(ms1))
                T_SM = np.logspace(np.log10(Ti), np.log10(Tfinal_integral), 500)
                T_SM_domain[i, j] = T_SM
                a = T_SM_to_a(T_SM)
            
                # Print progress
                print("Computing for theta = {}, ms1 = {}".format(theta, ms1))
                # Compute for each combination and store in the results array
                results[i, j] = rho_DS_solve(a, ms1, theta, args.flavor, args.antineutrino, args.simplify)
                print("Time elapsed: {} seconds".format(time.time() - start_time))

    # Save the data to a file
    data = {'T_SM_domain': T_SM_domain, 'sinsq2theta_values': sinsq2theta_values, 'ms1_values': ms1_values,
            'flavor': args.flavor, 'simplify': args.simplify, 'antineutrino': args.antineutrino, 
            'rho_DS_domain': results}
    #save using pickle
    with open(os.path.join(dir_path, 'results/', args.output_file + '.pkl'), 'wb') as file:
       pickle.dump(data, file)

# Save a grid of runs over 3D [ms1, ms2, theta] space to a file. 
# Sample usage for rho:
# python darksector.py --ms1 -3 4 121 --sinsq2theta -20 -20  1 --flavor 'electron' --output_file rho_computation_no_theta
# Sample usage for temperature:
# python darksector.py --T_SM_domain 0.5 4 1000 --T_DS_initial 10 --ms1 0 2 7 --ms2 -2 -1 1 --sinsq2theta -13 -10  10 --flavor 'electron' --output_file sparse_grid --simplify 'merle' --antineutrino
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--computation', type=str, required=False, default='rho')
    parser.add_argument('--T_SM_domain', nargs='+', type=float, required=False)
    parser.add_argument('--T_DS_initial', type=float, required=False)
    parser.add_argument('--sinsq2theta', nargs='+', type=float, required=True)
    parser.add_argument('--ms1', nargs='+', type=float, required=True)
    parser.add_argument('--ms2', nargs='+', type=float, required=False)
    parser.add_argument('--flavor', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--simplify', required=False, default=False)
    parser.add_argument('--antineutrino', action='store_true')
    args = parser.parse_args()
    
    if args.computation == 'rho':
        main_rho(args)
    elif args.computation == 'temperature':
        main_temperature(args)
    else:
        print("Invalid computation type. Please choose either 'rho' or 'temperature'")
        sys.exit(1)