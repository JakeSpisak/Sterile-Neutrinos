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

def compute_drho_injected_dt_single(T_SM, ms, theta, flavor, merle_simplification=False, antineutrino=False):
    """Compute the rate of change of the dark sector energy density due to sterile neutrino production"""
    y = float(ms)/T_SM
    def integrand(x):
        return SID_rate(x*T_SM, T_SM, theta, ms, flavor, antineutrino, 
                        merle_simplification)*x**2*np.sqrt(x**2 + y**2)/(np.exp(np.sqrt(x**2 + y**2)) + 1)
    result, err = integrate.quad(integrand, 0.1, 20)
    return T_SM**4*result/(2*np.pi**2)

# Allow the function to take in an array of temperatures
compute_drho_injected_dt = np.vectorize(compute_drho_injected_dt_single)

def SID_rate(p, T, theta, ms, flavor, antineutrino=False, merle_simplification=False):
    """Compute the SID sterile production rate according to a more 
    detailed perscription."""
    delta = ms**2/(2*p)
    numerator = 0.5*(delta*np.sin(2*theta))**2
    denominator = ((delta*np.sin(2*theta))**2 + (active_scattering_rate(p, T, flavor, merle_simplification)/2)**2 + 
                   (delta*np.cos(2*theta) - matter_potential(p, T, flavor, antineutrino))**2)
    conversion_probability = numerator/denominator
    
    return 0.5*active_scattering_rate(p, T, flavor, merle_simplification)*conversion_probability

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
    """Active neutrino thermal scattering coefficients"""
    if np.ndim(T)==0 and np.ndim(p)==0:
        return scattering_coeffs_dict[flavor]([T, p/T])[0]
    else:
        results = []
        assert len(T) == len(p)
        for Tval, pval in zip(T, p):
            results.append(scattering_coeffs_dict[flavor]([Tval, pval/Tval])[0])
        return np.array(results)

# Load the 5 to 150 GeV numerical rates
try:
    df = pd.read_table("{}/thermal_neutrino_scattering_coefficients/Gamma_5to150GeV.dat".format(data_path),
                        skiprows=1, header=None, delim_whitespace=True, names=['T/Gev', 'k/T', '\Gamma/T'])
    koverT_domain = np.array(df.loc[df['T/Gev'] == 150, 'k/T'])
    T_domain = np.flip(np.array(df.loc[df['k/T'] == 1.0, 'T/Gev']))*10**3 #convert to MeV
    coefficients = np.flip(np.array(df['\Gamma/T']).reshape((len(T_domain), len(koverT_domain))), axis=0)
    interp5to150Gev = interpolate.RegularGridInterpolator((T_domain, koverT_domain), coefficients, bounds_error=True, fill_value=None)
except Exception as e:
    print("WARNING: scattering coefficient data files not found. Error message: {str(e)}}")

def active_scattering_rate(p, T, flavor, merle_simplification=False):
    """The active neutrino scattering rate. Valid from 1 Mev to 150 Gev"""
    if merle_simplification:
        return scattering_coeffs_dict[flavor]([T, 3])[0]*c.Gf**2*p*T**4
    
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
    
# Matter potential

def matter_potential(p, T, flavor, antineutrino=False):
    """The neutrino matter potential."""
    if antineutrino:
        prefactor = -1
    else:
        prefactor = 1
    n_nu = 2*special.zeta(3)*T**3/(4*np.pi**2)    
    baryon_potential = prefactor*np.sqrt(2)*c.Gf*2*special.zeta(3)*T**3*c.eta_B/(4*np.pi**2)
    Z_potential = 8*np.sqrt(2)*c.Gf*2*n_nu*p*avg_p(T)/(3*c.m_Z**2)
    W_potential = 8*np.sqrt(2)*c.Gf*2*n_lepton(T, flavor)*p*E_avg_lepton(T, flavor)/(3*c.m_W**2)
    
    return baryon_potential - Z_potential - W_potential

# Various functions needed for the scattering-induced decoherence rate
    
def n_lepton(T, flavor):
    """The number density of a given lepton flavor."""
    if flavor == 'electron':
        m = c.m_e
    elif flavor == 'muon':
        m = c.m_mu
    elif flavor == 'tau':
        m = c.m_tau
    else:
        print("ERROR: Flavor must be electron, muon, or tau")
    return T**3*lepton_integral_interp(2,m/T)/(2*np.pi**2)
        
def E_avg_lepton_single(T, m):
    """ Average lepton energy of a single temperature value."""
    if m/T < 10:
        return T*lepton_integral_interp(3, m/T)/lepton_integral_interp(2, m/T)
    # When the particle is extremely nonrelativistic, energy=mass
    else:
        return m

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
    
def avg_p(T):
    """average p for a relavistic thermal particle."""
    return 7*np.pi**4*T/(180*special.zeta(3))

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
    l2 = interpolate.interp1d(lepton_integral_dict[2]['x'], lepton_integral_dict[2]['value'])
    l3 = interpolate.interp1d(lepton_integral_dict[3]['x'], lepton_integral_dict[3]['value'])
# print the exception and a warning message
except Exception as e:
    print("WARNING: Error loading precomputed lepton integral data. Make sure you have run leptonintegral.py, Error message: {str(e)}}")

def lepton_integral_interp(n,x):
    if n==2:
        return l2(x)
    elif n==3:
        return l3(x)
    else:
        print("n must equal 2 or 3 to use precomputed values")
        return None
    
# Use precomputed SM thermodynamic quantities
try:
    with open("{}/SM_thermodynamic_quantities.pkl".format(data_path), 'rb') as file:
        thermodynamic_results = pickle.load(file)
    SM_entropy_density = interpolate.interp1d(thermodynamic_results['T_domain'], thermodynamic_results['entropy_density'])
except Exception as e:
    print("WARNING: Error loading precomputed standard model data. Make sure you have run standardmodel.py, Error message: {str(e)}}")
    
def boltzmann_integrand(Tprime, p, Tf, theta, ms, flavor, antineutrino=False, merle_simplification=False):
    """Integrand for the boltzmann equation"""
    momentum = p*(SM_entropy_density(Tprime)/SM_entropy_density(Tf))**(1./3)
    return fermi_dirac(momentum, Tprime, eta=0)*0.5*SID_rate(p, Tprime, theta, ms, flavor, antineutrino, merle_simplification)/(sm.dTemperature_dtime(Tprime))

def boltzmann_solve(p, Ti, Tf, theta, ms, flavor, discontinuity_masses, antineutrino=False, merle_simplification=False):
    """Solve for the sterile neutrino distribution funciton by integrating the boltzmann equation
    Should change to eliminate discontinuity_masses and just do one integral from Ti to Tf"""
    limits = np.sort([m for m in discontinuity_masses if m < Ti and m > Tf] + [Ti, Tf])[::-1]
    total_result, total_err = 0, 0
    for i in range(len(limits)-1):
        result, err = integrate.quad(boltzmann_integrand, limits[i], limits[i+1], args=(p, Tf, theta, ms, flavor, antineutrino, merle_simplification))
        total_result += result
        total_err += err
    return total_result, total_err

def compute_f_and_omegahsq(ms, theta, merle_simplification, discontinuity_masses, flavor, Ti, Tf, poverT_min=0.01, poverT_max=15, antineutrino=False):
    poverTs = np.logspace(np.log10(poverT_min), np.log10(poverT_max), 100)
    f = []
    for p in poverTs*Tf:
        result, err = boltzmann_solve(p, Ti, Tf, theta, ms, flavor, discontinuity_masses, antineutrino, merle_simplification)
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