import pickle
import numpy as np
import os
from scipy import integrate

def lepton_integral(n, x):
    """Occupation number integrals"""
    integrand = lambda y, x: y**n/(np.exp(np.sqrt(x**2 + y**2))-1)
    # Need to distinguish between scalars and arrays
    if np.isscalar(x):
        result, _ = integrate.quad(integrand, 0, 100, args=(x))
    else:
        result = [integrate.quad(integrand, 0, 100, args=(x_val))[0] for x_val in x]
    return result

# Compute the lepton integral for n=2 and 3 and save

if __name__ == "__main__":
    data_dic = {}
    for n in [2,3]:
        x_domain = np.logspace(-2, 2, 1000)
        data = lepton_integral(n, x_domain)
        x_domain_extended = np.concatenate(([0], x_domain, [np.inf]))
        data_extended = np.concatenate(([data[0]], data, [0]))
        data_dic[n] = {
            "x":x_domain_extended,
            "value":data_extended
        }
    # Save the data
    # modify the path to allow for the data files to be found
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, 'data/lepton_integrals.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(data_dic, f)