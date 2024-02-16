import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import numpy as np
import darksector as ds
import standardmodel as sm
from matplotlib import cm
import constants as c
import constraints as con
import sid
import json
from scipy import interpolate, optimize, integrate, special

# Parameters
theta = 0.5*np.sqrt(10**-15) # fig 1
ms = 1000 
flavor = 'electron'
antineutrino=False
simplify=False
Ti = min(30*sid.Tmax(ms), 149*10**3)
#Tf_integral = max(1, 0.001*sid.Tmax(ms))
Tf = 10**0
Tf_integral = Tf
T_SM_domain = np.logspace(np.log10(Ti), np.log10(Tf_integral), 1000)

# Computation
y_domain = np.linspace(0.25, 5, 50)

fs_boltzmann = []
for y in y_domain:
    rate = sid.boltzmann_integrand_T(np.log(T_SM_domain), y, Tf, theta, ms, flavor, antineutrino, simplify)
    result = [integrate.simpson(rate[:i+1], np.log(T_SM_domain[:i+1])) for i in range(len(rate)-1)]
    # prepend zero
    result.insert(0, 0)
    fs_boltzmann.append(np.array(result))

# First index is y, second index is T
poverT_SM_domain = np.transpose([y_domain*Tf*(sid.SM_entropy_density(T_SM_domain[i])/sid.SM_entropy_density(Tf))**(1./3) for i in range(len(T_SM_domain))])/T_SM_domain 
fs_boltzmann = np.array(fs_boltzmann)

# Compute beta: assumes a massless fermion, single DOF like is assumed for the fs_boltzmann computation
beta_domain = np.array([integrate.simpson(poverT_SM_domain[:,i]**2*fs_boltzmann[:,i], x=poverT_SM_domain[:,i])/(1.5*special.zeta(3)) for i in range(len(T_SM_domain))])

# Compute the energy density in the DS. Assume it is always relativistic. Multiply by 2 to get antineutrinos as well. 
rho_DS_domain = np.array([(2*T_SM_domain[i]**4/(2*np.pi**2))*integrate.simpson(poverT_SM_domain[:,i]**3*fs_boltzmann[:,i], x=poverT_SM_domain[:,i]) for i in range(len(T_SM_domain))])
T_DS_domain = (rho_DS_domain*240/(4*7*np.pi**2))**(1./4) + 1e-5 #When thermalized. Spread the energy among 4 sterile states

# Compute the time dilation factor for poverT = 3
gamma_SM = np.sqrt(1+(3*T_SM_domain)**2/ms**2)
gamma_DS = np.sqrt(1+(3*T_DS_domain)**2/ms**2)
elapsed_time = np.zeros(len(T_SM_domain))
elapsed_time[0] = 0
for i in range(1, len(T_SM_domain)):
    elapsed_time[i] = (T_SM_domain[i]-T_SM_domain[i-1])/(sid.dTemperature_dtime(T_SM_domain[i-1])) + elapsed_time[i-1]

# Save the data
np.savetxt('data/evolution_example/T_SM_domain.txt', T_SM_domain)
np.savetxt('data/evolution_example/T_DS_domain.txt', T_DS_domain)
np.savetxt('data/evolution_example/beta_domain.txt', beta_domain)
np.savetxt('data/evolution_example/gamma_SM.txt', gamma_SM)
np.savetxt('data/evolution_example/gamma_DS.txt', gamma_DS)
input_params = {'ms':ms, 'flavor':flavor}
with open('data/evolution_example/input_params.json', 'w') as f:
    json.dump(input_params, f)