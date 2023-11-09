from scipy import integrate, interpolate, special, optimize
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

def compute_drho_injected_dt_single(T_SM, ms, theta, flavor, antineutrino=False, simplify=False):
    """Compute the rate of change of the dark sector energy density due to sterile neutrino production"""
    y = float(ms)/T_SM
    def integrand(x):
        return SID_rate(x*T_SM, T_SM, theta, ms, flavor, antineutrino, 
                        simplify)*x**2*np.sqrt(x**2 + y**2)/(np.exp(np.sqrt(x**2 + y**2)) + 1)
    result, err = integrate.quad(integrand, 0.25, 12)
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
    simplify : optional
    """
    delta = ms**2/(2*p)
    gamma_active = active_scattering_rate(p, T, flavor, simplify)
    damping = gamma_active/2
    V = matter_potential(p, T, flavor, antineutrino, simplify)
    if simplify == 'no theta in denom':
        conversion_probability = 0.5*(delta*np.sin(2*theta))**2/(damping**2 + (delta*np.cos(2*theta) - V)**2)        
        return 0.5*gamma_active*conversion_probability

    else:
        conversion_probability = 0.5*(delta*np.sin(2*theta))**2/((delta*np.sin(2*theta))**2 + damping**2 + (delta*np.cos(2*theta) - V)**2)        
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
        Whether to use 'merle' or 'DW' simplification.
    """
    # Put in an exception to catch the case where a ValueError hits
    try:
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

def fermi_dirac(p, T, m=0, eta=0):
    """Fermi dirac distribution. All variables in MeV"""
    e = np.sqrt(p**2 + m**2)
    if e/T > 20:
        return 0 # to avoid overflow issues in the exponent
    return 1/(np.exp(e/T - eta)+1)

fermi_dirac = np.vectorize(fermi_dirac)

def bose_einstein(p, T, m=0, eta=0):
    """Fermi dirac distribution. All variables in MeV"""
    e = np.sqrt(p**2 + m**2)
    if e/T > 20:
        return 0 # to avoid overflow issues in the exponent
    elif e/T < 1e-2:
        print(f"Warning: e/T={e/T} is very small. f_BE -> Infinity and will cause numerical issues.")
    return 1/(np.exp(e/T - eta)-1)

bose_einstein = np.vectorize(bose_einstein)

def fint_to_n(fint, T):
    """Input the integral over the distribution function and the temperature in MeV.
    Output the number density in MeV^3"""
    return T**3*fint/(2*np.pi**2)

# Matter potential

def matter_potential_below_EW(p, T, flavor, antineutrino=False):
        prefactor = 1 if antineutrino else -1
        baryon_potential = prefactor*np.sqrt(2)*c.Gf*n_gamma(T)*(c.eta_B/4)
        if flavor == 'electron':
            baryon_potential = -1*baryon_potential
        Z_potential = -2*n_nu(T, 1)*E_avg_nu(T)*8*np.sqrt(2)*p*c.Gf/(3*c.m_Z**2)
        W_potential = -2*n_lepton(T, flavor, 2)*E_avg_lepton(T, flavor)*8*np.sqrt(2)*p*c.Gf/(3*c.m_W**2)
        return baryon_potential + Z_potential + W_potential

def matter_potential_above_EW(p, T, flavor):
    return (3*c.g_SU2L**2 + c. g_U1Y**2)*T**2/(32*p)

# Interpolate the results from precomputed data
try:
    with open('data/matter_potential_data.pkl', 'rb') as pickle_file:
        data_dict = pickle.load(pickle_file)
    T_domain = data_dict['T_domain'] 
    poverT_domain = data_dict['poverT_domain']
    interp_functions = {}
    for flavor in ['electron', 'muon', 'tau']:
        interp_func = interpolate.RegularGridInterpolator((poverT_domain, T_domain), data_dict[flavor], bounds_error=True, method='linear')
        interp_functions[flavor] = interp_func

except Exception as e:
    print("WARNING: Error loading precomputed matter potential data. Make sure you have run precompute.py, Error message: {str(e)}}")

def matter_potential(p, T, flavor, antineutrino=False, simplify=False):
    """The full neutrino matter potential"""
    if simplify=='DW':
        return -(7*np.pi/24)*c.Gf**2*T**4*p*4*np.sin(2*c.thetaW)**2/(15*c.fine_structure)
    elif simplify=='below EW':
        return matter_potential_below_EW(p, T, flavor, antineutrino)
    else:
        if T > c.T_EW:
            return matter_potential_above_EW(p, T, flavor)
        elif T < 10**3:
            return matter_potential_below_EW(p, T, flavor, antineutrino)
        else:
            input_points = np.column_stack((p/T, T))
            return interp_functions[flavor](input_points)
    
matter_potential = np.vectorize(matter_potential, otypes=[float])
    
def matter_potential_integral(p, T, flavor, antineutrino=False):
    # The integral crashes in these limits
    if T > c.T_EW:
        return matter_potential_above_EW(p, T, flavor)
    if T < 10**3:
        return matter_potential_below_EW(p, T, flavor, antineutrino)
    
    if flavor == 'electron':
        ml = c.m_e
    elif flavor == 'muon':
        ml = c.m_mu
    elif flavor == 'tau':
        ml = c.m_tau
    else:
        print("Flavor must be electron, muon, or tau")

    vz = np.pi*c.fine_structure_W/np.cos(c.thetaW)**2
    vw = np.pi*c.fine_structure_W*(2+(ml/c.m_W)**2)
    mW = 0 if T>c.T_EW else c.m_W*np.sqrt(1-T**2/c.T_EW**2)
    mZ = 0 if T>c.T_EW else c.m_Z*np.sqrt(1-T**2/c.T_EW**2)
    result = -1*(vz*B(p, T, 0, mZ) + vw*B(p, T, ml, mW))

    return result

matter_potential_integral = np.vectorize(matter_potential_integral, otypes=[float])

def B_integrand(x, p, T, mf, mA, partial=False):
    k = T*x
    delta = mA**2 - mf**2
    eA = np.sqrt(mA**2 + k**2)
    ef = np.sqrt(mf**2 + k**2)
    boson = (0.5*delta*k*L_func(delta, p, k, eA)/eA - 4*p*k**2/eA)*bose_einstein(k, T, m=mA)
    fermion = (0.5*delta*k*L_func(delta, p, k, ef)/ef - 4*p*k**2/ef)*fermi_dirac(k, T, m=mf)
    # Add an extra factor of T to convert the dk in the integrand to dx
    if partial=='boson':
        return T*boson/(8*np.pi**2*p**2)
    elif partial=='fermion':
        return T*fermion/(8*np.pi**2*p**2)
    else:
        return T*(boson + fermion)/(8*np.pi**2*p**2)

def L_func(delta, p, k, e, m=0):
    num = (delta+2*p*(k+e))*(delta+2*p*(k-e))
    denom = (m**2+delta-2*p*(k-e))*(m**2+delta-2*p*(k+e))
    return np.log(np.abs(num/denom))

def B(p, T, mf, mA):
    delta = mA**2 - mf**2
    if delta == 0:
        return integrate.quad(B_integrand, 0.01, 10, args=(p, T, mf, mA))[0]
    else:
        fermion_singularity = delta/(4*p) - mf**2*p/delta
        boson_singularity = delta/(4*p) - mA**2*p/delta
        fermion = integrate.quad(B_integrand, 0.01, 15, args=(p, T, mf, mA, 'fermion'), points=[fermion_singularity])[0]
        boson = integrate.quad(B_integrand, 0.01, 15, args=(p, T, mf, mA, 'boson'), points=[boson_singularity])[0]
        return fermion + boson
    
def find_crossing_point(poverT, guess, flavor, T_min=10**3):
    func = lambda T: 2*poverT*T*matter_potential_integral(T*poverT, T, flavor)
    return optimize.root_scalar(func, x0=guess, bracket=[T_min, 0.99*c.T_EW], rtol=10**-3)

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

def boltzmann_integrand_T(log_Tprime, y, Tf, theta, ms, flavor, antineutrino=False, simplify=False):
    """Integrand for the boltzmann equation with temperature 'Tprime' as the independent variable. 
    The RHS of the equation (df_s/dlog(T))(y, T) = RHS 
    Uses log to improve the numerical calculation, and d/dlog(T) = T*d/dT
    y is proportional the co-moving momentum of a free-falling particle, ie a constant of motion:
    it's normalized so that y=p/Tf at Tf.
    The corresponding proper momentum at each temperature is computed using co-moving entropy conservation."""""
    Tprime = np.exp(log_Tprime)
    p = y*Tf*(SM_entropy_density(Tprime)/SM_entropy_density(Tf))**(1./3)
    dTdtime = sm.dTemperature_dtime(Tprime, hubble_rate, SM_energy_density, SM_pressure, SM_drho_dT)
    return Tprime*SID_rate(p, Tprime, theta, ms, flavor, antineutrino, simplify)*fermi_dirac(p, Tprime, eta=0)/dTdtime


def boltzmann_solve(y, Ti, Tf_integral, Tf, theta, ms, flavor, antineutrino=False, simplify=False, num_points=500):
    """Solve for the sterile neutrino distribution function by integrating the boltzmann equation"""
    T_domain = np.logspace(np.log10(Ti), np.log10(Tf_integral), num_points)
    rate = boltzmann_integrand_T(np.log(T_domain), y, Tf, theta, ms, flavor, antineutrino, simplify)
    result = integrate.simpson(rate, np.log(T_domain))
    return result

def SID_rate_DW(p, T, theta, ms):
    """Compute the SID sterile production rate according to Dodelson-Widrow's
    original paper."""
    vaccum_rate = (7*np.pi/24)*(c.Gf**2)*p*(T**4)*np.sin(2*theta)**2
#    Veff = -4*np.sqrt(2)*17*Gf*T**4*p/(2*np.pi**2*m_W**2)
    c_DW = 4*np.sin(2*c.thetaW)**2/(15*c.fine_structure)
    sinsq2th_matter = np.sin(2*theta)**2*ms**2/(np.sin(2*theta)**2*ms**2 + (c_DW*vaccum_rate*p/ms + ms/2)**2)
    return 0.5*vaccum_rate*sinsq2th_matter

def compute_omegahsq_sid(ms, theta, flavor, antineutrino, simplify=False,
                          ymin=0.25, ymax=5, num_y=50, num_T=500):
    """Compute the final omega h^2 from a scattering-induced decoherence calculation"""
    Ti = min(10*Tmax(ms), 149*10**3)
    Tfinal_integral = max(1, 0.02*Tmax(ms))
    poverT_domain = np.linspace(ymin, ymax, num_y)
    Tfinal = 1 #So that fs(p/T) is referenced to the CnuB temperature
    fs = [boltzmann_solve(y, Ti, Tfinal_integral, Tfinal, theta, ms, flavor, antineutrino, simplify, num_T) for y in poverT_domain]
    ndens = 2*(c.Tcnub**3/np.pi**2)*integrate.simpson(fs*poverT_domain**2, poverT_domain) #2 helicity states
    rho = ndens*ms
    return ds.rho_to_omegahsq(rho)
    
# From 2017 Kev's sterile neutrino review
def Tmax(ms):
    """The approximate maximum temperature for sterile neutrino production in MeV"""
    return 130*(ms*10**3)**(1./3)

# Equation 9 in DW
def fs_DW(poverT, gstar, theta, M):
    """M in MeV. Small theta approximation, theta = mu/M"""
    return (6/np.sqrt(gstar))*10**9*M*theta**2*fermi_dirac(poverT*1, 1)