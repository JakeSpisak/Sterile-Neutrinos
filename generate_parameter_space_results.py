import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})
from matplotlib.ticker import FuncFormatter
import numpy as np
import darksector as ds
import standardmodel as sm
import sid
import constants as c
import pickle
from scipy import special, interpolate, optimize, fft

##############################
# Initial Data Processing
##############################

# Load the pickle file with the parameter scan computing rho_DS across a range of m_N1 values.
# Theta is not included in the scan - the analytical dependency is used instead.
with open('results/rho_computation_no_theta.pkl', 'rb') as file:
    data_dict = pickle.load(file)
ms_idx = 51
T_SM_final = data_dict['T_SM_domain'][0,ms_idx:,-1]
rho_DS_final = data_dict['rho_DS_domain'][0,ms_idx:,-1]

# Get the final dark sector temperature for each point in the parameter space
ms1_values = data_dict['ms1_values'][ms_idx:]
sinsq2theta_base = data_dict['sinsq2theta_values'][0]

# Get the dark sector temperature today
scale_factor_ratio, T_DS_final = np.zeros(len(ms1_values)), np.zeros(len(ms1_values))
for j, ms1 in enumerate(ms1_values):
    T_DS_final[j] = ds.find_T_DS_from_rho(rho_DS_final[j], ms1, 0)[0]
    scale_factor_ratio[j] = ds.T_SM_to_a(T_SM_final[j])/ds.T_SM_to_a(2) # Ratio from T_SM_final to 2 MeV 
    scale_factor_ratio[j] *= c.Tcnub/2 # Ratio from 2 MeV to today

T_DS_interp = interpolate.interp1d(ms1_values, T_DS_final, kind='linear', bounds_error=True)
T_SM_interp = interpolate.interp1d(ms1_values, T_SM_final, kind='linear', bounds_error=True)
scale_factor_interp = interpolate.interp1d(ms1_values, scale_factor_ratio, kind='linear', bounds_error=True)

# Create a dense 2D grid with dimensions (sinsq2theta, ms1)
num = 100
ms1_dense = np.logspace(np.log10(np.min(ms1_values)), np.log10(np.max(ms1_values)), num)
sinsq2theta_dense = np.logspace(-25, -10, num)
X, Y = np.meshgrid(np.log10(ms1_dense), np.log10(sinsq2theta_dense))

T_DS_dense, scale_factor_dense, T_DS_dense_today, T_SM_dense = [], [], [], []
for ms1 in ms1_dense:
    for sinsq2theta in sinsq2theta_dense:
        theta_scaling = (sinsq2theta/sinsq2theta_base)**(1./4)
        T_DS_dense.append(T_DS_interp(ms1)*theta_scaling)
        scale_factor = scale_factor_interp(ms1)
        scale_factor_dense.append(scale_factor)
        T_DS_dense_today.append(T_DS_interp(ms1)*theta_scaling*scale_factor)
        T_SM_dense.append(T_SM_interp(ms1))

T_DS_dense = np.array(T_DS_dense).reshape((num, num))
T_SM_dense = np.array(T_SM_dense).reshape((num, num))
scale_factor_dense = np.array(scale_factor_dense).reshape((num, num))
T_DS_dense_today = np.array(T_DS_dense_today).reshape((num, num))

# Get the temperature ratio
temp_ratio = T_DS_dense_today/c.Tcmb

# Get m_N1 required today
ms2_required = [c.rho_crit_over_hsq*0.12/(ds.compute_current_DM_ndens(T_SM, T_DS)) for T_DS, T_SM in zip(T_DS_dense.flatten(), T_SM_dense.flatten())]
ms2_required = np.array(ms2_required).reshape((len(ms1_dense), len(sinsq2theta_dense)))

# Get the angle where ms1 = 3*ms2_required
ms2_func = interpolate.RegularGridInterpolator((ms1_dense, sinsq2theta_dense), ms2_required, method='linear')
ms1_diff = lambda log_sinsq2theta, log_ms1: np.abs(np.log10(3*ms2_func((10**log_ms1, 10**log_sinsq2theta))) - log_ms1)
log_sinsq2theta_equal = []
for log_ms1 in np.log10(ms1_dense):
    guess = -14-2*log_ms1
    log_sinsq2theta_equal.append(optimize.minimize(ms1_diff, guess, args=(log_ms1), bounds=[(np.log10(np.min(sinsq2theta_dense))+0.01, np.log10(np.max(sinsq2theta_dense))-0.01)]).x[0])

fs_limit = 9.7*10**-3 # Minimum free streaming mass in MeV: from Nadler et. al. 2021


##############################
# Save the data
##############################
np.savetxt('data/results_plot_data/ms1_dense.txt', ms1_dense)
np.savetxt('data/results_plot_data/log_sinsq2theta_equal.txt', log_sinsq2theta_equal)
np.savetxt('data/results_plot_data/X.txt', X)
np.savetxt('data/results_plot_data/Y.txt', Y)
np.savetxt('data/results_plot_data/ms2_required.txt', ms2_required)
np.savetxt('data/results_plot_data/temp_ratio.txt', temp_ratio)

