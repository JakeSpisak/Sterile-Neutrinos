import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy import interpolate
import darksector as ds
import constants as c
import standardmodel as sm
import sid
import json

####################
# Load the data
####################

ms1_dense = np.loadtxt('data/results_plot_data/ms1_dense.txt')
log_sinsq2theta_equal = np.loadtxt('data/results_plot_data/log_sinsq2theta_equal.txt')
X = np.loadtxt('data/results_plot_data/X.txt')
Y = np.loadtxt('data/results_plot_data/Y.txt')
ms2_required = np.loadtxt('data/results_plot_data/ms2_required.txt')
temp_ratio = np.loadtxt('data/results_plot_data/temp_ratio.txt')

T_SM_domain = np.loadtxt('data/evolution_example/T_SM_domain.txt')
T_DS_domain = np.loadtxt('data/evolution_example/T_DS_domain.txt')
beta_domain = np.loadtxt('data/evolution_example/beta_domain.txt')
gamma_SM = np.loadtxt('data/evolution_example/gamma_SM.txt')
gamma_DS = np.loadtxt('data/evolution_example/gamma_DS.txt')
with open('data/evolution_example/input_params.json', 'r') as f:
    input_params = json.load(f)
ms = input_params['ms']
flavor = input_params['flavor']


####################
# Required m_N2
####################

plt.figure(figsize=(8,8))

#Contours 

# Make all the contour lines solid
contours = plt.contour(X, Y, np.log10(ms2_required.T), colors='k', linestyles='solid', alpha=1,
                       levels=np.linspace(-7, 9, 9))
# Put the contour labels as 10^x
def format_func(value, tick_number):
    return r'{:.2g}'.format(10**(value))
plt.clabel(contours, inline=True, fontsize=16,  fmt=FuncFormatter(format_func))

# Constraints

# Annotations
plt.annotate('m$_{N1}$=3m$_{N2}$', (1.2, -19.4), fontsize=18, c='b', rotation=-42)
plt.annotate('free streaming bound', (-0.3, -13), fontsize=18, c='b', rotation=-20)

# fs contraints
fs_limit = 9.7*10**-3 # Minimum free streaming mass in MeV: from Nadler et. al. 2021
fs_contour = plt.contour(X, Y, np.log10(ms2_required.T), colors='b', linestyles='solid', levels=[np.log10(fs_limit)], alpha=0)
fs_path = fs_contour.collections[0].get_paths()[0]
fs_x_values = fs_path.vertices[:, 0]
fs_y_values = fs_path.vertices[:, 1]

# Hatched region between ms1=ms2 and fs_constraints
fs_interp = interpolate.interp1d(fs_x_values, fs_y_values, kind='linear', bounds_error=False, fill_value='extrapolate')
fs_y_interpolated = fs_interp(np.log10(ms1_dense))
plt.fill_between(np.log10(ms1_dense), log_sinsq2theta_equal, fs_y_interpolated, where=(log_sinsq2theta_equal < fs_y_interpolated), 
                 color='lightsteelblue', edgecolor='b', linewidth=2)

plt.xlabel(r'$Log_{10}(m_{N1}$/MeV)')
plt.ylabel(r'$Log_{10}(\sin^2(2 \theta))$')
plt.gca().set_aspect(0.4)
plt.tight_layout()
plt.grid()
plt.xlim(-2, 4)

plt.show()

plt.savefig("plots/mN2_required.png")

####################
# Temperature Ratio
####################

plt.figure(figsize=(8,8))

# Contours
levels = [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
contours = plt.contour(X, Y, temp_ratio.T, levels, colors='k')
plt.clabel(contours, inline=True, fontsize=12)

# Constraints

# Labels
plt.annotate('m$_{N1}$=3m$_{N2}$', (0.6, -17.9), fontsize=18, c='b', rotation=-38)
plt.annotate('free streaming bound', (-0.3, -12), fontsize=18, c='b', rotation=-17)

# fs contraints
fs_contour = plt.contour(X, Y, np.log10(ms2_required.T), colors='b', linestyles='solid', levels=[np.log10(fs_limit)], alpha=0)
fs_path = fs_contour.collections[0].get_paths()[0]
fs_x_values = fs_path.vertices[:, 0]
fs_y_values = fs_path.vertices[:, 1]

# Hatched region between ms1=3ms2 and fs_constraints
fs_interp = interpolate.interp1d(fs_x_values, fs_y_values, kind='linear', bounds_error=False, fill_value='extrapolate')
fs_y_interpolated = fs_interp(np.log10(ms1_dense))
plt.fill_between(np.log10(ms1_dense), log_sinsq2theta_equal, fs_y_interpolated, where=(log_sinsq2theta_equal < fs_y_interpolated), 
                 color='lightsteelblue', edgecolor='b', linewidth=2)

plt.xlabel(r'$Log_{10}(m_{N1}$/MeV)')
plt.ylabel(r'$Log_{10}(\sin^2(2 \theta))$')
plt.gca().set_aspect(0.4)
plt.tight_layout()
plt.grid()
plt.xlim(-2, 4)

plt.savefig("plots/temperature_ratio.png", bbox_inches='tight')

################################
# Evolution Example
################################

Gphi_over_Gf = 0.1
x_min, x_max = 5*10**4, 1

# Get the sterile densities assuming annihilation of N1.
rho_s1 = 2*np.array([sm.compute_energy_density(T, ms, 1) for T in T_DS_domain]) 
rho_s2 = 4*np.array([sm.compute_energy_density(T, 0, 1) for T in T_DS_domain])-rho_s1

fig, ax = plt.subplots(2, 1, figsize=(9, 12), gridspec_kw={'height_ratios': [1, 0.5]}, sharex=True)
fig.subplots_adjust(hspace=0)

# Hubble rate
ax[0].loglog(T_SM_domain, ds.hubble_rate_func(T_SM_domain)*c.MeVtoHz, c='k', linewidth=4, ls='dotted')
ax[0].annotate('Hubble rate', (6*10**3, 1.5*10**4), fontsize=22, c='k', rotation=0)

# Active scattering rate
ax[0].loglog(T_SM_domain, sid.active_scattering_rate(3*T_SM_domain, T_SM_domain, flavor)*c.MeVtoHz, c='y', linewidth=4)
ax[0].annotate(r'$\Gamma_\alpha$', (9*10**2, 2*10**16), fontsize=22, c='y', rotation=0)

# 2 to 2
ax[0].loglog(T_SM_domain, ds.gamma_s_2to2(3*T_SM_domain, T_SM_domain, beta_domain, Gphi_over_Gf)*c.MeVtoHz, c='r', linewidth=4)
ax[0].loglog(T_SM_domain, ds.gamma_s_2to2(3*T_DS_domain, T_DS_domain, 1, Gphi_over_Gf)*c.MeVtoHz, c='r', linewidth=4)
ax[0].annotate(r'$\Gamma_{\phi,2 \rightarrow 2}$', (3*10**4, 7*10**14), fontsize=22, c='r', rotation=0)
ax[0].fill_between(T_SM_domain, ds.gamma_s_2to2(3*T_DS_domain, T_DS_domain, 1, Gphi_over_Gf)*c.MeVtoHz, 
                   ds.gamma_s_2to2(3*T_SM_domain, T_SM_domain, beta_domain, Gphi_over_Gf)*c.MeVtoHz, color='r', alpha=0.5)

# 2 to 4
ax[0].loglog(T_SM_domain, ds.gamma_s_2to4(3*T_SM_domain, T_SM_domain, beta_domain, Gphi_over_Gf)*c.MeVtoHz, linewidth=4, c='k', alpha=0.5)
ax[0].loglog(T_SM_domain, ds.gamma_s_2to4(3*T_DS_domain, T_DS_domain, 1, Gphi_over_Gf)*c.MeVtoHz, linewidth=4, c='k', alpha=0.5)
ax[0].annotate(r'$\Gamma_{\phi,2 \rightarrow 4}$', (2*10**3, 10**1), fontsize=22, c='k', rotation=0, alpha=0.5)
ax[0].fill_between(T_SM_domain, ds.gamma_s_2to4(3*T_SM_domain, T_SM_domain, beta_domain, Gphi_over_Gf)*c.MeVtoHz, 
                   ds.gamma_s_2to4(3*T_DS_domain, T_DS_domain, 1, Gphi_over_Gf)*c.MeVtoHz, color='k', alpha=0.3)

# 1 to 3
ax[0].loglog(T_SM_domain, ds.sterile_decay_rate(ms, Gphi_over_Gf)*c.MeVtoHz/gamma_SM, linewidth=4, c='b')
ax[0].loglog(T_SM_domain, ds.sterile_decay_rate(ms, Gphi_over_Gf)*c.MeVtoHz/gamma_DS, linewidth=4, c='b')
ax[0].annotate(r'$\Gamma_{\phi, 1 \rightarrow 3}$', (5*10**2, 3*10**8), fontsize=22, c='b', rotation=0)
ax[0].fill_between(T_SM_domain, ds.sterile_decay_rate(ms, Gphi_over_Gf)*c.MeVtoHz/gamma_SM, 
                   ds.sterile_decay_rate(ms, Gphi_over_Gf)*c.MeVtoHz/gamma_DS, color='b', alpha=0.5)


ax[0].invert_xaxis()
ax[0].set_ylabel(r'Rate (Hz)')
ax[0].set_ylim(10.**-5, 10.**20)
ax[0].set_xlim(x_min, x_max)

# Lower (density) plot
ax[1].loglog(T_SM_domain, ds.energy_density_func(T_SM_domain), c='c', ls='--', linewidth=4)
ax[1].annotate(r'$\rho_{SM}$', (10**2, 10**11), fontsize=22, c='c', rotation=0)
ax[1].loglog(T_SM_domain, rho_s1, c='purple', linewidth=3.5, ls='--')
ax[1].annotate(r'$\rho_{N1}$', (6*10**2, 10**0), fontsize=22, c='purple', rotation=0)
ax[1].loglog(T_SM_domain, rho_s2, c='orange', linewidth=4, ls='--')
ax[1].annotate(r'$\rho_{N2}$', (10**2, 3*10**5), fontsize=22, c='orange', rotation=0)
ax[1].invert_xaxis()
ax[1].set_ylabel(r'$\rho$ (MeV$^4$)')
ax[1].set_xlabel(r'$T_{SM}$ (MeV)')
ax[1].set_xlim(x_min, x_max)
ax[1].set_ylim(10.**-5, 10.**25)

# Change the thickness of the plot frame
for axis in ax:
    axis.spines['top'].set_linewidth(2)     # Top spine
    axis.spines['right'].set_linewidth(2)   # Right spine
    axis.spines['bottom'].set_linewidth(2)  # Bottom spine
    axis.spines['left'].set_linewidth(2)    # Left spine
    axis.tick_params(axis='both', which='major', direction='in', width=2, length=8, top=True, right=True)
    axis.tick_params(axis='both', which='minor', direction='in', width=1, length=4, top=True, right=True)
    axis.grid(which='major', alpha=0.7)

plt.savefig("plots/thermalization_1to3.png", bbox_inches='tight')
plt.savefig("plots/thermalization_1to3.pdf", bbox_inches='tight')