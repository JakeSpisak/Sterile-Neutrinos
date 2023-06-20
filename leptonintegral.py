import pickle
import numpy as np
import os
from scipy import integrate

def lepton_integral_number_density(x):
    """Number density integral. x = m/T, y=p/T"""
    integrand = lambda y, x: y**2/(np.exp(np.sqrt(x**2 + y**2))+1)
    # Need to distinguish between scalars and arrays
    if np.isscalar(x):
        result, _ = integrate.quad(integrand, 0, 100, args=(x))
    else:
        result = [integrate.quad(integrand, 0, 100, args=(x_val))[0] for x_val in x]
    return result

def lepton_integral_energy_density(x):
    """Energy density integral. x = m/T, y=p/T"""
    integrand = lambda y, x: y**2*np.sqrt(x**2 + y**2)/(np.exp(np.sqrt(x**2 + y**2))+1)
    # Need to distinguish between scalars and arrays
    if np.isscalar(x):
        result, _ = integrate.quad(integrand, 0, 100, args=(x))
    else:
        result = [integrate.quad(integrand, 0, 100, args=(x_val))[0] for x_val in x]
    return result

# Compute the lepton integrals and save

if __name__ == "__main__":
    data_dic = {}
    x_domain = np.logspace(-2, 2, 1000)
    data_ndens = lepton_integral_number_density(x_domain)
    data_edens = lepton_integral_energy_density(x_domain)
    x_domain_extended = np.concatenate(([0], x_domain, [np.inf]))
    data_ndens_extended = np.concatenate(([data_ndens[0]], data_ndens, [0]))
    data_edens_extended = np.concatenate(([data_edens[0]], data_edens, [0]))
    data_dic['number_density'] = {
        "x":x_domain_extended,
        "value":data_ndens_extended
    }
    data_dic['energy_density'] = {
        "x":x_domain_extended,
        "value":data_edens_extended
    }
    # Save the data
    # modify the path to allow for the data files to be found
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, 'data/lepton_integrals.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(data_dic, f)