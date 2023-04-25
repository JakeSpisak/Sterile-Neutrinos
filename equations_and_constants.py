import numpy as np
from scipy.special import zeta
from scipy import integrate, interpolate
import pandas as pd
import pickle
from scipy.integrate import quad


#################################
#CONSTANTS
##################################


# Unit conversions
hbar = 6.582*10**-16 #eV s
MeVtoHz = 10**6/hbar
HztoMeV = hbar/10**6
kb =  1.380649*10**-23 #J/K 
charge = 1.602176634*10**-19 #C
MeVtoJ = 10**6*charge
JtoMeV = 10**-6/charge
KtoMeV = kb*JtoMeV
eVtoMeV = 10**-6
c = 2.9979*10**8 #m/s
MeVtoinversecm = 10**6/(hbar*c*100)

# Masses, all in MeV
# Leptons
m_e, m_mu, m_tau = 0.510,105.7,1776.9
# Quarks
m_u, m_d, m_s, m_c, m_b, m_t = 2.3,4.8,95,1290,4180,173500
# Gauge bosons
m_W, m_Z, m_H = 80400,91200,125090
#Baryons
m_p, m_n = 938,939
# Mesons
m_pi0, m_pipm, m_K0, m_Kpm, m_eta, m_etaprime, m_rhopm, m_rho0, m_omega = 134.9766,139.570,497.614,493.678,647.862,957.78,775.11,775.26,782.65
# all masses
masses = np.sort([m_e, m_mu, m_tau, m_u, m_d, m_s, m_c, m_b, m_t, m_W, m_Z, m_H, m_pi0, m_pipm, m_K0, m_Kpm, m_eta, m_etaprime, m_rhopm, m_rho0, m_omega])
discontinuity_masses = np.sort([m_mu, m_tau, m_c, m_b, m_t, m_W, m_Z, m_H, m_pi0, m_pipm])

# Other fundamental constants, all in MeV unless otherwise noted
Gf = 1.166 * 10**-5 *10**-6 #MeV**-2
grav = (1.22*10**22)**-2 #MeV**-2 
thetaW = np.arcsin(np.sqrt(0.2229))
fine_structure = 7.297352*10**-3
eta_B = 6*10**-10 #baryon asymmetry
f_pi = 131 # for pion channel decay
# QCD Transition temperature and width
T_qcd = 170 
w_qcd = 10
h=0.7 # hubble factor
omega_dm_hsq = 0.120 #omega_m*h^2, planck 2018, error is 0.001
H100 =  100*3.24*10**-20*HztoMeV
Tcmb = 2.725*KtoMeV
TcmbtoTcnub = (4./11)**(1./3)


#################################
#Numerical Active Scattering Rate
#################################

#1 Mev to 10 GeV
scattering_coeffs_dict = {}
for i, flavor in enumerate(['electron', 'muon', 'tau']):
    df = pd.read_table("/home/jakespisak/Fuller/sterile_sterile_interactions/thermal_neutrino_scattering_coefficients/hatIQ_M001_alpha{}_Fermi.dat".format(i+1),
                         skiprows=6, header=None, delim_whitespace=True, names=['T/MeV', 'q/T', 'hat{I_Q}'])
    qoverTdomain = np.array(df.loc[df['T/MeV'] == 10000.00, 'q/T'])
    Tdomain = np.flip(np.array(df.loc[df['q/T'] == 1.00, 'T/MeV']))
    coefficients = np.flip(np.array(df['hat{I_Q}']).reshape((len(Tdomain), len(qoverTdomain))), axis=0)
    # Append a T=0 data point with the same coefficients as T=1 MeV
    coefficients = np.insert(coefficients, 0, coefficients[0], axis=0)
    Tdomain = np.insert(Tdomain, 0, 0)
    scattering_coeffs_dict[flavor] = interpolate.RegularGridInterpolator((Tdomain, qoverTdomain), coefficients, bounds_error=True, fill_value=None)
    
def scattering_coeffs_1Mev_to_5_GeV(T, p, flavor):
    """Active neutrino thermal scattering coefficients"""
    if np.isscalar(T) and np.isscalar(p):
        return scattering_coeffs_dict[flavor]([T, p/T])[0]
    else:
        results = []
        assert len(T) == len(p)
        for Tval, pval in zip(T, p):
            results.append(scattering_coeffs_dict[flavor]([Tval, pval/Tval])[0])
        return np.array(results)

# 5 to 150 GeV
df = pd.read_table("/home/jakespisak/Fuller/fortepiano_public/thermal_neutrino_scattering_coefficients/Gamma_5to150GeV.dat",
                     skiprows=1, header=None, delim_whitespace=True, names=['T/Gev', 'k/T', '\Gamma/T'])
koverTdomain = np.array(df.loc[df['T/Gev'] == 150, 'k/T'])
Tdomain = np.flip(np.array(df.loc[df['k/T'] == 1.0, 'T/Gev']))*10**3 #convert to MeV
coefficients = np.flip(np.array(df['\Gamma/T']).reshape((len(Tdomain), len(koverTdomain))), axis=0)
interp5to150Gev = interpolate.RegularGridInterpolator((Tdomain, koverTdomain), coefficients, bounds_error=True, fill_value=None)

def active_scattering_rate(p, T, flavor, merle_simplification=False):
    """The active neutrino scattering rate. Valid from 1 Mev to 150 Gev"""
    if merle_simplification:
        return scattering_coeffs_dict[flavor]([T, 3])[0]*Gf**2*p*T**4
    
    changeover = 5*10**3 
    if np.isscalar(T) and np.isscalar(p):
        if T< changeover:
            return scattering_coeffs_dict[flavor]([T, p/T])[0]*Gf**2*p*T**4
        else:
            return interp5to150Gev([T, p/T])[0]*T 
        
    else:
        # Convert either value to an array if it isn't already
        if np.isscalar(T):
            T = T*np.ones(len(p))
        if np.isscalar(p):
            p = p*np.ones(len(T))
        results = []
        assert len(T) == len(p)
        for Tval, pval in zip(T, p):
            if Tval < changeover:
                results.append(scattering_coeffs_dict[flavor]([Tval, pval/Tval])[0]*Gf**2*pval*Tval**4)
            else:
                results.append(interp5to150Gev([Tval, pval/Tval])[0]*Tval)
        return np.array(results)
    
#################################
#Standard Model Thermodynamics
##################################

def compute_SM_relativistic_dof_approx(T):
    """Calculate the relativistic degrees of freedom for the purposes of 
    computing the energy density. Valid up to arbitrarily high temperatures,
    and down to weak decoupling. Makes the approximation of g=0 once m>T."""
    #.....Enter particle characteristics, in the following order of particles: 
    #.....Leptons: e,mu,tau; Quarks: u,d,s,c,b,t; Gauge bosons (and Higgs): W+-,Z,H;
    #.....Baryons: p,n; Mesons: pi0,pi+-,K0,K+-,eta,eta',rho+-,rho0,omega
    #.....Particle masses in MeV
    masses = [m_e, m_mu, m_tau, m_u, m_d, m_s, m_c, m_b, m_t,m_W, m_Z,m_H,m_p,m_n,
                      m_pi0, m_pipm, m_K0, m_Kpm, m_eta, m_etaprime, m_rhopm, m_rho0, m_omega]

    stats = ['f','f','f','f','f','f','f','f','f','b','b','b',
            'f','f','b','b','b','b','b','b','b','b','b']
    #.....Particle class: 'l' for lepton, 'q' for quark, 'g' for gauge bosons (and Higgs), 'h' for hadron
    particle_classes = ['l','l','l','q','q','q','q','q','q','g','g','g',
             'h','h','h','h','h','h','h','h','h','h','h']
    gdofs = [4.,4.,4.,12.,12.,12.,12.,12.,12.,6.,3.,1., 
            4.,4.,1.,2.,2.,2.,1.,1.,6.,3.,3.]

    #.....Statistical weights for massless or nearly massless particles (photons, neutrinos, gluons)
    gphot = 2. # Photon dofs
    gnuact = 3*2*7./8 # Active neutrino dofs: prior to weak decoupling 
    ggl = 16./(np.exp((T_qcd-T)/w_qcd)+1.) # Gluon dofs
    g_tot = gphot+gnuact+ggl
    
    # Massive particles approximation: simply set g_rel=0 when m<T. 
    # Also account for QCD phase transition
    for mass, stat, particle_class, gdof in zip(masses, stats, particle_classes, gdofs):
        if mass>T:
            gdof = 0
        if(particle_class == 'q'): 
            if T>0.1*T_qcd:
                gdof=gdof/(np.exp((T_qcd-T)/w_qcd)+1.)
            # avoid overflow warnings 
            else:
                gdof=0
        elif(particle_class == 'h'):
            if T<10*T_qcd:
                gdof=gdof/(np.exp((T-T_qcd)/w_qcd)+1.)
            # avoid overflow warnings
            else:
                gdof=0
        if stat=='f':
            gdof *= 7./8
        g_tot += gdof
        
    return g_tot

def compute_energy_density(T, m, sign):
    """Return the energy density for a particle with a single degree of freedom. 
    T and m in MeV. sign = 1: bose-einstein. sign=-1: fermi-dirac"""
    assert sign in [1, -1]
    y = float(m)/T
    # limits
    if y < 0.01:
        if sign == 1:
            return T**4*(7/8.)*np.pi**2/30
        elif sign == -1:
            return T**4*np.pi**2/30
    if y > 20:
        return 0
    # intermediate calculation
    def integrand(x):
        return x**2 * np.sqrt(x**2 + y**2)/(np.exp(np.sqrt(x**2 + y**2)) + sign)
    result, err = quad(integrand, 0.1, 20)
    return T**4*result/(2*np.pi**2)

def compute_pressure(T, m, sign):
    """Return the pressure for a particle with a single degree of freedom. 
    T and m in MeV. sign = 1: bose-einstein. sign=-1: fermi-dirac"""
    assert sign in [1, -1]
    y = float(m)/T
    # limits
    if y < 0.01:
        return compute_energy_density(T, m, sign)/3.
    if y > 20:
        return 0
    # intermediate calculation
    def integrand(x):
        return x**4/(3*np.sqrt(x**2 + y**2)*(np.exp(np.sqrt(x**2 + y**2)) + sign))
    result, err = quad(integrand, 0.1, 20)
    return T**4*result/(2*np.pi**2)

def compute_drho_dT(T, m, sign):
    """Return the drho/dT for a particle with a single degree of freedom. 
    T and m in MeV. sign = 1: bose-einstein. sign=-1: fermi-dirac"""
    assert sign in [1, -1]
    y = float(m)/T
    
    if y < 0.01:
        return 4*compute_energy_density(T, m, sign)/T
    if y > 20:
        return 0
    
    # intermediate calculation
    def integrand(x):
        return x**2*(x**2+y**2)*np.exp(np.sqrt(x**2 + y**2))/(np.exp(np.sqrt(x**2 + y**2)) + sign)**2
    result, err = quad(integrand, 0.1, 20)
    return T**3*result/(2*np.pi**2)

def compute_entropy_density(T, m, sign):
    """Return the entropy for a particle with a single degree of freedom. 
    T and m in MeV. sign = 1: bose-einstein. sign=-1: fermi-dirac"""
    return (compute_energy_density(T, m, sign) + compute_pressure(T, m, sign))/T
    
def compute_total_thermodynamic_quantities(T):
    """Calculate the total energy density, pressure, entropy density, and drho/dT of all standard model particles. 
    Valid up to arbitrarily high temperatures, and down to weak decoupling. It does not treat electron-positron
    annihilation correctly."""
    #.....Enter particle characteristics, in the following order of particles: 
    #.....Leptons: e,mu,tau; Quarks: u,d,s,c,b,t; Gauge bosons (and Higgs): W+-,Z,H;
    #.....Baryons: p,n; Mesons: pi0,pi+-,K0,K+-,eta,eta',rho+-,rho0,omega
    #.....Massless particles: photon, all neutrinos, all gluons
    #.....Particle masses in MeV
    masses = [m_e, m_mu, m_tau, m_u, m_d, m_s, m_c, m_b, m_t,m_W, m_Z,m_H,m_p,m_n,
              m_pi0, m_pipm, m_K0, m_Kpm, m_eta, m_etaprime, m_rhopm, m_rho0,m_omega,
              0,0,0]

    stats = ['f','f','f','f','f','f','f','f','f','b','b','b',
            'f','f','b','b','b','b','b','b','b','b','b','b','f','b']
    #.....Particle class: 'l' for lepton, 'q' for quark, 'g' for gauge bosons (and Higgs), 'h' for hadron 
    particle_classes = ['l','l','l','q','q','q','q','q','q','g','g','g',
             'h','h','h','h','h','h','h','h','h','h','h','g','l','g']
    gdofs = [4.,4.,4.,12.,12.,12.,12.,12.,12.,6.,3.,1., 
            4.,4.,1.,2.,2.,2.,1.,1.,6.,3.,3.,2.,6.,16./(np.exp((T_qcd-T)/w_qcd)+1.)]

    
    # Calculate the contribution from each species
    rho, pressure, drho_dT = 0, 0, 0
    for mass, stat, particle_class, gdof in zip(masses, stats, particle_classes, gdofs):
        if(particle_class == 'q'): 
            if T>0.1*T_qcd:
                gdof=gdof/(np.exp((T_qcd-T)/w_qcd)+1.)
            # avoid overflow warnings 
            else:
                gdof=0
        elif(particle_class == 'h'):
            if T<10*T_qcd:
                gdof=gdof/(np.exp((T-T_qcd)/w_qcd)+1.)
            # avoid overflow warnings
            else:
                gdof=0
        if stat=='b':
            sign=-1
        elif stat=='f':
            sign=1
        rho += gdof*compute_energy_density(T, mass, sign)
        pressure += gdof*compute_pressure(T, mass, sign)
        drho_dT += gdof*compute_drho_dT(T, mass, sign)
    entropy_density = (rho + pressure)/T
        
    return rho, pressure, drho_dT, entropy_density

def save_thermodynamic_quantities(Tdomain, path):
    energy_density, pressure, drho_dT, entropy_density = [], [], [], []
    for T in Tdomain:
        r, p, dr, e = compute_total_thermodynamic_quantities(T)
        energy_density.append(r)
        pressure.append(p)
        drho_dT.append(dr)
        entropy_density.append(e)
        
    results = {
        'Tdomain':Tdomain,
        'energy_density':energy_density,
        'pressure':pressure,
        'drho_dT':drho_dT,
        'entropy_density':entropy_density
    }
    with open(path, 'wb') as file:
        pickle.dump(results, file)
        
# Save the thermodynamic quantities to interpolated functions
with open('/home/jakespisak/Fuller/sterile_sterile_interactions/for_boltzmann_calc/SM_thermodynamic_quantities.pkl', 'rb') as file:
    thermodynamic_results = pickle.load(file)
SM_energy_density = interpolate.interp1d(thermodynamic_results['Tdomain'], thermodynamic_results['energy_density'])
SM_pressure = interpolate.interp1d(thermodynamic_results['Tdomain'], thermodynamic_results['pressure'])
SM_drho_dT = interpolate.interp1d(thermodynamic_results['Tdomain'], thermodynamic_results['drho_dT'])
SM_entropy_density = interpolate.interp1d(thermodynamic_results['Tdomain'], thermodynamic_results['entropy_density'])
    
    
#################################
#Other Equations
##################################


# All temperatures and masses are in MeV. Lifetimes are in seconds

def boltmann_supressed(T, ms):
    """Return true if sterile is boltzmann supressed (T>m)"""
    if T > ms:
        return False
    else:
        return True
    
def SID_rate_DW(p, T, theta, ms):
    """Compute the SID sterile production rate according to Dodelson-Widrow's
    original paper."""
    vaccum_rate = (7*np.pi/24)*(Gf**2)*p*(T**4)*np.sin(2*theta)**2
#    Veff = -4*np.sqrt(2)*17*Gf*T**4*p/(2*np.pi**2*m_W**2)
    c_DW = 4*np.sin(2*thetaW)**2/(15*fine_structure)
    sinsq2th_matter = np.sin(2*theta)**2*ms**2/(np.sin(2*theta)**2*ms**2 + (c_DW*vaccum_rate*p/ms + ms/2)**2)
    return 0.5*vaccum_rate*sinsq2th_matter

def SID_rate(p, T, theta, ms, flavor, antineutrino=False, merle_simplification=False):
    """Compute the SID sterile production rate according to a more 
    detailed perscription."""
    delta = ms**2/(2*p)
    numerator = 0.5*(delta*np.sin(2*theta))**2
    denominator = ((delta*np.sin(2*theta))**2 + (active_scattering_rate(p, T, flavor, merle_simplification)/2)**2 + 
                   (delta*np.cos(2*theta) - matter_potential(p, T, flavor, antineutrino))**2)
    conversion_probability = numerator/denominator
    
    return 0.5*active_scattering_rate(p, T, flavor, merle_simplification)*conversion_probability

def SID_rate_integrated(T, theta, m5, flavor, antineutrino=False):
    """Compute the SID sterile production rate, integrated over momentum."""
    if np.isscalar(T):
        integrand = lambda p: SID_rate(p, T, theta, m5, flavor, antineutrino)*p**2/(np.exp(p/T)+1)
        result, err = quad(integrand, 0.01*T, 10*T)
        return result/(1.5*T**3*zeta(3))
    else:
        rates = []
        for Ti in T:
            integrand = lambda p: SID_rate(p, Ti, theta, m5, flavor)*p**2/(np.exp(p/Ti)+1)
            result, err = quad(integrand, 0.01*Ti, 10*Ti)
            rates.append(result/(1.5*Ti**3*zeta(3)))
        return np.array(rates)

def h(p, T, theta, ms, flavor, antineutrino=False, merle_simplification=False):
    """The variable 'h' in Merle 2016"""
    return 0.5*SID_rate(p, T, theta, ms, flavor, antineutrino, merle_simplification)

def matter_potential(p, T, flavor, antineutrino=False):
    """The neutrino matter potential."""
    if antineutrino:
        prefactor = -1
    else:
        prefactor = 1
    n_nu = 2*zeta(3)*T**3/(4*np.pi**2)    
    baryon_potential = prefactor*np.sqrt(2)*Gf*2*zeta(3)*T**3*eta_B/(4*np.pi**2)
    Z_potential = 8*np.sqrt(2)*Gf*2*n_nu*p*avg_p(T)/(3*m_Z**2)
    W_potential = 8*np.sqrt(2)*Gf*2*n_lepton(T, flavor)*p*E_avg_lepton(T, flavor)/(3*m_W**2)
    
    return baryon_potential - Z_potential - W_potential
    
def n_lepton(T, flavor):
    """The number density of a given lepton flavor."""
    if flavor == 'electron':
        m = m_e
    elif flavor == 'muon':
        m = m_mu
    elif flavor == 'tau':
        m = m_tau
    else:
        print("ERROR: Flavor must be electron, muon, or tau")
    return T**3*lepton_integral_interp(2,m/T)/(2*np.pi**2)
        
def E_avg_lepton(T, flavor):
    """ Average lepton energy of a given flavor."""
    if flavor == 'electron':
        m = m_e
    elif flavor == 'muon':
        m = m_mu
    elif flavor == 'tau':
        m = m_tau
    else:
        print("ERROR: Flavor must be electron, muon, or tau")
    if np.isscalar(T):
        if m/T < 10:
            return T*lepton_integral_interp(3,m/T)/lepton_integral_interp(2,m/T)
        # When the particle is extremely nonrelativistic, energy=mass
        else:
            return m
    else:
        results = []
        for temp in T:
            if m/temp < 10:
                results.append(temp*lepton_integral_interp(3,m/temp)/lepton_integral_interp(2,m/temp))
            # When the particle is extremely nonrelativistic, energy=mass
            else:
                results.append(m)
        return results
    
def avg_p(T):
    """average p for a relavistic thermal particle."""
    return 7*np.pi**4*T/(180*zeta(3))

def Hubble_rate(T):
    """Hubble rate. Only includes SM particles"""
    return np.sqrt((8*np.pi*grav/3)*SM_energy_density(T))

# From 2017 Kev's sterile neutrino review
def Tmax(ms):
    """The approximate maximum temperature for sterile neutrino production in MeV"""
    return 133*(ms*10**3)**(1./3)

def thermalization_history(T_high, T_low, theta, ms, flavor, antineutrino=False, num_T=100):
    """Find if the sterile thermalizes, and if so, when the production rate rises above 
    and then dips below the hubble rate"""
    T_domain = np.logspace(np.log10(T_high), np.log10(T_low), num_T)
    thermalized=False
    T_thermal='None'
    T_freezeout='None'
    for T in T_domain:
        gamma_greater_than_hubble = np.log(SID_rate(avg_p(T), T, theta, ms, flavor, antineutrino))-np.log(Hubble_rate(T)) > 0
        if not thermalized and gamma_greater_than_hubble:
            T_thermal=T
            thermalized=True
        if thermalized and not gamma_greater_than_hubble:
            T_freezeout=T
            return thermalized, T_thermal, T_freezeout
            
    return thermalized, T_thermal, T_freezeout

def lifetime(ms, theta):
    """The approximate lifetime (in MeV^-1) of a sterile decaying into three neutrinos
    or a pion and electon"""
    if ms < m_pipm:
        return (192*np.pi**3/Gf**2)*ms**-5*np.sin(theta)**-2/MeVtoHz
    else:
        mass_terms = ms**-1*((ms**2-(m_pipm + m_e)**2)*(ms**2-(m_pipm - m_e)**2))**-0.5
        return 16*np.pi*(Gf*f_pi*np.sin(theta))**-2*mass_terms/MeVtoHz
    
def m4_dark_matter(m_5, tau_5, g_Tf4, g_Tf5):
    """Return the mass of the lightest sterile (m_4) required
    for it to constitute all of the dark matter. Masses in MeV,
    lifetime in seconds. 
    """
    m_4 = 10**-3*(m_5/1850)*np.sqrt(tau_5/1)*g_Tf4/g_Tf5
    return m_4

def dark_matter_frac(m_4, m_5, tau_5, g_Tf4, g_Tf5):
    """Return the dark matter fraction, relative to the known abundance.
    Masses in MeV, lifetime in seconds. 
    """
    return (m_4/0.001)*(1850/m_5)*np.sqrt(1/tau_5)*g_Tf5/g_Tf4

def fermi_dirac(p, T, eta=0):
    """Fermi dirac distribution. All variables in MeV"""
    return 1/(1+np.exp(p/T - eta))

def fint_to_n(fint, T):
    """Input the integral over the distribution function and the temperature in MeV.
    Output the number density in MeV^3"""
    return T**3*fint/(2*np.pi**2)

def rho_to_omegahsq(rho):
    """Input energy density in MeV^4. Return \Omega h^2"""
    return 8*np.pi*grav*rho/(3*H100**2)

#lepton_integral. Assumes saved calculation results that can be interpolated.

with open("/home/jakespisak/Fuller/sterile_sterile_interactions/for_boltzmann_calc/lepton_integrals.pkl", 'rb') as f:
    # load the dictionary from the file
    lepton_integral_dict = pickle.load(f)
l2 = interpolate.interp1d(lepton_integral_dict[2]['x'], lepton_integral_dict[2]['value'])
l3 = interpolate.interp1d(lepton_integral_dict[3]['x'], lepton_integral_dict[3]['value'])
def lepton_integral_interp(n,x):
    if n==2:
        return l2(x)
    elif n==3:
        return l3(x)
    else:
        print("n value isn't 2 or 3: doing the numerical integration")
        return lepton_integral(n,x)
    
def boltzmann_integrand(Tprime, p, Tf, theta, ms, flavor, antineutrino=False, merle_simplification=False):
    """Integrand for the boltzmann equation"""
    momentum = p*(SM_entropy_density(Tprime)/SM_entropy_density(Tf))**(1./3)
    return fermi_dirac(momentum, Tprime, eta=0)*h(momentum, Tprime, theta, ms, flavor, antineutrino, merle_simplification)/(dTemperature_dtime(Tprime))

def boltzmann_solve(p, Ti, Tf, theta, ms, flavor, discontinuity_masses, antineutrino=False, merle_simplification=False):
    """Solve for the sterile neutrino distribution funciton by integrating the boltzmann equation"""
    limits = np.sort([m for m in discontinuity_masses if m < Ti and m > Tf] + [Ti, Tf])[::-1]
    total_result, total_err = 0, 0
    for i in range(len(limits)-1):
        result, err = quad(boltzmann_integrand, limits[i], limits[i+1], args=(p, Tf, theta, ms, flavor, antineutrino, merle_simplification))
        total_result += result
        total_err += err
    return total_result, total_err

def compute_f_and_omegahsq(ms, theta, merle_simplification, flavor, Ti, Tf, poverT_min=0.01, poverT_max=15, antineutrino=False):
    poverTs = np.logspace(np.log10(poverT_min), np.log10(poverT_max), 100)
    f = []
    for p in poverTs*Tf:
        result, err = solve(p, Ti, Tf, theta, ms, flavor, discontinuity_masses, antineutrino, merle_simplification)
        f.append(-1*result)
    f_xsq_integrand = interpolate.interp1d(poverTs, f*poverTs**2, kind='linear')
    fint, err = quad(f_xsq_integrand, poverT_min, poverT_max)
    
    # Compute number density today: multiplying by two to include antineutrinos
    ndens = fint_to_n(2*fint, Tcmb)
    omegahsq = rho_to_omegahsq(ms*ndens)
    
    return f, poverTs, f_xsq_integrand, omegahsq 

def dTemperature_dtime(T):
    """Compute the time vs temperature relation in the SM plasma"""
    return -1*3*Hubble_rate(T)*(SM_energy_density(T)+SM_pressure(T))*SM_drho_dT(T)**-1