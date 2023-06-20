from scipy import integrate, interpolate, special
import numpy as np
import os
import pandas as pd
import pickle
import constants as c
import standardmodel as sm
import darksector as ds

# path to the data directory
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, 'data/')

def compute_drho_injected_dt_single(T_SM, ms, theta, flavor, simplify=False, antineutrino=False):
    """Compute the rate of change of the dark sector energy density due to sterile neutrino production"""
    y = float(ms)/T_SM
    def integrand(x):
        return SID_rate(x*T_SM, T_SM, theta, ms, flavor, antineutrino, 
                        simplify)*x**2*np.sqrt(x**2 + y**2)/(np.exp(np.sqrt(x**2 + y**2)) + 1)
    result, err = integrate.quad(integrand, 0.1, 20)
    return T_SM**4*result/(2*np.pi**2)

# Allow the function to take in an array of temperatures
compute_drho_injected_dt = np.vectorize(compute_drho_injected_dt_single)

def SID_rate(p, T, theta, ms, flavor, antineutrino=False, simplify=False):
    """Compute the SID sterile production rate per active neturino. RHS of Eq. 6.6 in Abazajian 2001
    without the [f_a - f_s] term.
    
    parameters
    ----------
    p : float
        The neutrino momentum in MeV.
    T : float
        The neutrino temperature in MeV.
    theta : float
        The mixing angle between the active and sterile neutrino.
    ms : float
        The sterile neutrino mass in MeV.
    flavor : str
        The neutrino flavor. Must be 'electron', 'muon', or 'tau'.
    antineutrino : bool, optional
    simplify : bool, optional
    """
    delta = ms**2/(2*p)
    gamma_active = active_scattering_rate(p, T, flavor, simplify)
    damping = gamma_active/2
    Vthermal = matter_potential(p, T, flavor, antineutrino, simplify)
    # The conversion probability is 0.5*sin(2 theta_matter)^2
    conversion_probability = 0.5*np.sin(2*theta)**2/(np.sin(2*theta)**2 + damping**2 + (np.cos(2*theta) - Vthermal/delta)**2)
    
    return 0.5*gamma_active*conversion_probability

#Numerical Active Scattering Rate

# Load the 1 Mev to 10 GeV numerical rates
try:
    scattering_coeffs_dict = {}
    for i, flavor in enumerate(['electron', 'muon', 'tau']):
        df = pd.read_table("{}/thermal_neutrino_scattering_coefficients/hatIQ_M001_alpha{}_Fermi.dat".format(data_path, i+1),
                            skiprows=6, header=None, delim_whitespace=True, names=['T/MeV', 'q/T', 'hat{I_Q}'])
        qoverT_domain = np.array(df.loc[df['T/MeV'] == 10000.00, 'q/T'])
        T_domain = np.flip(np.array(df.loc[df['q/T'] == 1.00, 'T/MeV']))
        coefficients = np.flip(np.array(df['hat{I_Q}']).reshape((len(T_domain), len(qoverT_domain))), axis=0)
        # Append a T=0 data point with the same coefficients as T=1 MeV
        coefficients = np.insert(coefficients, 0, coefficients[0], axis=0)
        T_domain = np.insert(T_domain, 0, 0)
        scattering_coeffs_dict[flavor] = interpolate.RegularGridInterpolator((T_domain, qoverT_domain), coefficients, bounds_error=True, fill_value=None)
except Exception as e:
    print("WARNING: scattering coefficient data files not found. Error message: {str(e)}}")
    
def scattering_coeffs_1Mev_to_5_GeV(T, p, flavor):
    """Active neutrino thermal scattering coefficients. Applies from p/T from 0.25 - 12.5"""
    if np.ndim(T)==0 and np.ndim(p)==0:
        return scattering_coeffs_dict[flavor]([T, p/T])[0]
    else:
        results = []
        assert len(T) == len(p)
        for Tval, pval in zip(T, p):
            results.append(scattering_coeffs_dict[flavor]([Tval, pval/Tval])[0])
        return np.array(results)

# Load the 5 to 150 GeV numerical rates. Applies from p/T from 0.1 - 12
try:
    df = pd.read_table("{}/thermal_neutrino_scattering_coefficients/Gamma_5to150GeV.dat".format(data_path),
                        skiprows=1, header=None, delim_whitespace=True, names=['T/Gev', 'k/T', '\Gamma/T'])
    koverT_domain = np.array(df.loc[df['T/Gev'] == 150, 'k/T'])
    T_domain = np.flip(np.array(df.loc[df['k/T'] == 1.0, 'T/Gev']))*10**3 #convert to MeV
    coefficients = np.flip(np.array(df['\Gamma/T']).reshape((len(T_domain), len(koverT_domain))), axis=0)
    interp5to150Gev = interpolate.RegularGridInterpolator((T_domain, koverT_domain), coefficients, bounds_error=True, fill_value=None)
except Exception as e:
    print("WARNING: scattering coefficient data files not found. Error message: {str(e)}}")

# Write a docstring in the numpy style for this function
def active_scattering_rate(p, T, flavor, simplify=False):
    """The active neutrino scattering rate. Valid from 1 Mev to 150 GeV. 
    Applies from p/T from 0.25 - 12.5 between 1 MeV and 5 GeV, and from p/T from 0.1 - 12 between 5 and 150 GeV.  

    Parameters
    ----------
    p : float
        The neutrino momentum in MeV.
    T : float
        The neutrino temperature in MeV.
    flavor : str
        The neutrino flavor. Must be 'electron', 'muon', or 'tau'.
    simplify : bool, optional
        Whether to use 'merle' or 'DW' simplification. The default is False.
    """
    # Put in an exception to catch the case where a ValueError hits
    try:
        assert simplify in [False, 'DW', 'merle']
        if simplify == 'DW':
            return (7*np.pi/24)*c.Gf**2*p*T**4
        
        if simplify == 'merle':
          return scattering_coeffs_dict[flavor]([T, 3])[0]*c.Gf**2*p*T**4
    
        else:
            changeover = 5*10**3 
            if np.ndim(T)==0 and np.ndim(p)==0:
                if T< changeover:
                    return scattering_coeffs_dict[flavor]([T, p/T])[0]*c.Gf**2*p*T**4
                else:
                    return interp5to150Gev([T, p/T])[0]*T 
                
            else:
                # Convert either value to an array if it isn't already
                if np.ndim(T)==0:
                    T = T*np.ones(len(p))
                if np.ndim(p)==0:
                    p = p*np.ones(len(T))
                results = []
                assert len(T) == len(p)
                for Tval, pval in zip(T, p):
                    if Tval < changeover:
                        results.append(scattering_coeffs_dict[flavor]([Tval, pval/Tval])[0]*c.Gf**2*pval*Tval**4)
                    else:
                        results.append(interp5to150Gev([Tval, pval/Tval])[0]*Tval)
                return np.array(results)
    
    except ValueError as e:
        print(f"ERROR message in `active_scattering_rate': {str(e)}")
        print(f"T: {T}")
        print(f"p/T: {p/T}")
        print(f"flavor: {flavor}")
        return None

#vectorize the active neutrino scattering function
active_scattering_rate = np.vectorize(active_scattering_rate)
    
# Matter potential

def matter_potential(p, T, flavor, antineutrino=False, simplify=False):
    """The neutrino matter potential."""
    if antineutrino:
        prefactor = 1
    else:
        prefactor = -1
    assert simplify in [False, 'DW', 'merle']
    if simplify=='DW':
        return -(7*np.pi/24)*c.Gf**2*T**4*p*4*np.sin(2*c.thetaW)**2/(15*c.fine_structure)
    else:
        baryon_potential = prefactor*np.sqrt(2)*c.Gf*n_gamma(T)*(c.eta_B/4)
        if flavor == 'electron':
            baryon_potential = -1*baryon_potential
        # Assume negligble differences between leptons and antileptons VT purposes
        Z_potential = -2*n_nu(T, 1)*E_avg_nu(T)*8*np.sqrt(2)*p*c.Gf/(3*c.m_Z**2)
        W_potential = -2*n_lepton(T, flavor, 2)*E_avg_lepton(T, flavor)*8*np.sqrt(2)*p*c.Gf/(3*c.m_W**2)
        
        return baryon_potential + Z_potential + W_potential

# Various functions needed for the scattering-induced decoherence rate
def n_gamma(T):
    return 2*special.zeta(3)*T**3/(np.pi**2)

def n_nu(T, g):
    return g*3*special.zeta(3)*T**3/(4*np.pi**2)    

def E_avg_nu(T):
    return T*7*np.pi**4/(180*special.zeta(3))    
    
def n_lepton(T, flavor, g):
    """The number density of a given lepton flavor with DOFg."""
    if flavor == 'electron':
        m = c.m_e
    elif flavor == 'muon':
        m = c.m_mu
    elif flavor == 'tau':
        m = c.m_tau
    else:
        print("ERROR: Flavor must be electron, muon, or tau")
    return g*T**3*lint_number_density(m/T)/(2*np.pi**2)
        
def E_avg_lepton_single(T, m):
    """ Average lepton energy of a single temperature value."""
    return T*lint_energy_density(m/T)/lint_number_density(m/T)

def E_avg_lepton(T, flavor):
    """ Average lepton energy of a given flavor."""
    if flavor == 'electron':
        m = c.m_e
    elif flavor == 'muon':
        m = c.m_mu
    elif flavor == 'tau':
        m = c.m_tau
    else:
        print("ERROR: Flavor must be electron, muon, or tau")
        return

    E_avg_lepton_vec = np.vectorize(E_avg_lepton_single)
    return E_avg_lepton_vec(T, m)

def fermi_dirac(p, T, eta=0):
    """Fermi dirac distribution. All variables in MeV"""
    return 1/(1+np.exp(p/T - eta))

def fint_to_n(fint, T):
    """Input the integral over the distribution function and the temperature in MeV.
    Output the number density in MeV^3"""
    return T**3*fint/(2*np.pi**2)

# Use precomputed lepton integrals
try:
    with open("{}/lepton_integrals.pkl".format(data_path), 'rb') as f:
        # load the dictionary from the file
        lepton_integral_dict = pickle.load(f)
    lint_number_density = interpolate.interp1d(lepton_integral_dict['number_density']['x'], lepton_integral_dict['number_density']['value'])
    lint_energy_density = interpolate.interp1d(lepton_integral_dict['energy_density']['x'], lepton_integral_dict['energy_density']['value'])
# print the exception and a warning message
except Exception as e:
    print("WARNING: Error loading precomputed lepton integral data. Make sure you have run leptonintegral.py, Error message: {str(e)}}")
    
# Use precomputed SM thermodynamic quantities
try:
    with open("{}/SM_thermodynamic_quantities.pkl".format(data_path), 'rb') as file:
        thermodynamic_results = pickle.load(file)
    SM_entropy_density = interpolate.interp1d(thermodynamic_results['T_domain'], thermodynamic_results['entropy_density'])
    hubble_rate = interpolate.interp1d(thermodynamic_results['T_domain'], thermodynamic_results['hubble_rate'])
    SM_energy_density = interpolate.interp1d(thermodynamic_results['T_domain'], thermodynamic_results['energy_density'])
    SM_pressure = interpolate.interp1d(thermodynamic_results['T_domain'], thermodynamic_results['pressure'])
    SM_drho_dT = interpolate.interp1d(thermodynamic_results['T_domain'], thermodynamic_results['drho_dT'])
except Exception as e:
    print("WARNING: Error loading precomputed standard model data. Make sure you have run standardmodel.py, Error message: {str(e)}}")
    
def boltzmann_integrand_T(Tprime, y, Tf, theta, ms, flavor, antineutrino=False, simplify=False):
    """Integrand for the boltzmann equation with temperature 'Tprime' as the independent variable. 
    The RHS of the equation (df_s/dT)(y, T) = RHS 
    y is proportional the co-moving momentum of a free-falling particle, ie a constant of motion:
    it's normalized so that y=p/Tf at Tf.
    The corresponding proper momentum at each temperature is computed using co-moving entropy conservation."""""
    p = y*Tf*(SM_entropy_density(Tprime)/SM_entropy_density(Tf))**(1./3)
    dTdtime = sm.dTemperature_dtime(Tprime, hubble_rate, SM_energy_density, SM_pressure, SM_drho_dT)
    return SID_rate(p, Tprime, theta, ms, flavor, antineutrino, simplify)*fermi_dirac(p, Tprime, eta=0)/dTdtime


def boltzmann_solve(y, Ti, Tf, theta, ms, flavor, antineutrino=False, simplify=False):
    """Solve for the sterile neutrino distribution function by integrating the boltzmann equation"""
    result, err = integrate.quad(boltzmann_integrand_T, Ti, Tf, args=(y, Tf, theta, ms, flavor, antineutrino, simplify))
    return result, err

def compute_f_and_omegahsq(ms, theta, simplify, flavor, Ti, Tf, poverT_min=0.01, poverT_max=15, antineutrino=False):
    poverTs = np.logspace(np.log10(poverT_min), np.log10(poverT_max), 100)
    f = []
    for p in poverTs*Tf:
        result, err = boltzmann_solve(p, Ti, Tf, theta, ms, flavor, antineutrino, simplify)
        f.append(-1*result)
    f_xsq_integrand = interpolate.interp1d(poverTs, f*poverTs**2, kind='linear')
    fint, err = integrate.quad(f_xsq_integrand, poverT_min, poverT_max)
    
    # Compute number density today: multiplying by two to include antineutrinos
    ndens = fint_to_n(2*fint, c.Tcmb)
    omegahsq = ds.rho_to_omegahsq(ms*ndens)
    
    return f, poverTs, f_xsq_integrand, omegahsq 

def SID_rate_DW(p, T, theta, ms):
    """Compute the SID sterile production rate according to Dodelson-Widrow's
    original paper."""
    vaccum_rate = (7*np.pi/24)*(c.Gf**2)*p*(T**4)*np.sin(2*theta)**2
#    Veff = -4*np.sqrt(2)*17*Gf*T**4*p/(2*np.pi**2*m_W**2)
    c_DW = 4*np.sin(2*c.thetaW)**2/(15*c.fine_structure)
    sinsq2th_matter = np.sin(2*theta)**2*ms**2/(np.sin(2*theta)**2*ms**2 + (c_DW*vaccum_rate*p/ms + ms/2)**2)
    return 0.5*vaccum_rate*sinsq2th_matter

def SID_rate_integrated(T, theta, m5, flavor, antineutrino=False):
    """Compute the SID sterile production rate, integrated over momentum."""
    if np.ndim(T) == 0:
        integrand = lambda p: SID_rate(p, T, theta, m5, flavor, antineutrino)*p**2/(np.exp(p/T)+1)
        result, err = integrate.quad(integrand, 0.01*T, 10*T)
        return result/(1.5*T**3*special.zeta(3))
    else:
        rates = []
        for Ti in T:
            integrand = lambda p: SID_rate(p, Ti, theta, m5, flavor)*p**2/(np.exp(p/Ti)+1)
            result, err = integrate.quad(integrand, 0.01*Ti, 10*Ti)
            rates.append(result/(1.5*Ti**3*special.zeta(3)))
        return np.array(rates)
    
# From 2017 Kev's sterile neutrino review
def Tmax(ms):
    """The approximate maximum temperature for sterile neutrino production in MeV"""
    return 133*(ms*10**3)**(1./3)