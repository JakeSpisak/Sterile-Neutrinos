import numpy as np
from scipy import interpolate, integrate, optimize, special
import constants as c
import pickle
import os

def compute_total_thermodynamic_quantities(T):
    """Calculate the total energy density, pressure, entropy density, and drho/dT of all standard model particles. 
    Valid up to arbitrarily high temperatures, and down to weak decoupling. It does not treat electron-positron
    annihilation correctly."""
    
    # Calculate the contribution from each species
    rho, pressure, drho_dT = 0, 0, 0
    for mass, stat, particle_class, gdof in zip(c.SM_masses, c.SM_stats, c.SM_particle_classes, c.SM_gdofs):
        if stat=='b':
            sign=-1
        elif stat=='f':
            sign=1
        if particle_class in ['q', 'gl']:
            if T>0.1*c.T_qcd:
                # Change from the variation in relativistic DOFs across the QCD epoch
                drho_dT+= compute_energy_density(T, mass, sign)*compute_QCD_dg_dT(gdof, T, c.T_qcd, c.w_qcd, 1)
                gdof=gdof/(np.exp((c.T_qcd-T)/c.w_qcd)+1.)
            # avoid overflow warnings 
            else:
                gdof=0
        elif(particle_class == 'h'):
            if T<10*c.T_qcd:
                # Change from the variation in relativistic DOFs across the QCD epoch
                drho_dT+= compute_energy_density(T, mass, sign)*compute_QCD_dg_dT(gdof, T, c.T_qcd, c.w_qcd, -1)
                gdof=gdof/(np.exp((T-c.T_qcd)/c.w_qcd)+1.)
            # avoid overflow warnings
            else:
                gdof=0
        rho += gdof*compute_energy_density(T, mass, sign)
        pressure += gdof*compute_pressure(T, mass, sign)
        drho_dT += gdof*compute_drho_single_dT(T, mass, sign)
    entropy_density = (rho + pressure)/T
        
    return rho, pressure, drho_dT, entropy_density

def compute_energy_density(T, m, sign):
    """Return the energy density for a particle with a single degree of freedom. 
    T and m in MeV. sign = -1: bose-einstein. sign=1: fermi-dirac"""
    assert sign in [1, -1]
    y = float(m)/T
    # limits
    if y < 0.1:
        if sign == 1:
            return T**4*(7/8.)*np.pi**2/30
        elif sign == -1:
            return T**4*np.pi**2/30
    if y > 20:
        return 0
    # intermediate calculation
    def integrand(x):
        return x**2 * np.sqrt(x**2 + y**2)/(np.exp(np.sqrt(x**2 + y**2)) + sign)
    result, err = integrate.quad(integrand, 0.1, 20)
    return T**4*result/(2*np.pi**2)
# # vectorize compute_energy_density
# compute_energy_density = np.vectorize(compute_energy_density)

def compute_number_density(T, m, sign):
    """Return the number density for a particle with a single degree of freedom. 
    T and m in MeV. sign = -1: bose-einstein. sign=1: fermi-dirac"""
    assert sign in [1, -1]
    y = float(m)/T
    # limits
    if y < 0.1:
        if sign == 1:
            return T**3*3*special.zeta(3)/(4*np.pi**2)
        elif sign == -1:
            return T**3*special.zeta(3)/(np.pi**2)
    if y > 20:
        return 0
    # intermediate calculation
    def integrand(x):
        return x**2/(np.exp(np.sqrt(x**2 + y**2)) + sign)
    result, err = integrate.quad(integrand, 0.1, 20)
    return T**3*result/(2*np.pi**2)
# # vectorize compute_energy_density
# compute_number_density = np.vectorize(compute_number_density)

def compute_pressure(T, m, sign):
    """Return the pressure for a particle with a single degree of freedom. 
    T and m in MeV. sign = 1: bose-einstein. sign=-1: fermi-dirac"""
    assert sign in [1, -1]
    y = float(m)/T
    # limits
    if y < 0.1:
        return compute_energy_density(T, m, sign)/3.
    if y > 20:
        return 0
    # intermediate calculation
    def integrand(x):
        return x**4/(3*np.sqrt(x**2 + y**2)*(np.exp(np.sqrt(x**2 + y**2)) + sign))
    result, err = integrate.quad(integrand, 0.1, 20)
    return T**4*result/(2*np.pi**2)

def compute_drho_single_dT(T, m, sign):
    """Return the drho/dT for a particle with a single degree of freedom. 
    T and m in MeV. sign = 1: bose-einstein. sign=-1: fermi-dirac"""
    assert sign in [1, -1]
    y = float(m)/T
    # limits
    if y < 0.1:
        return 4*compute_energy_density(T, m, sign)/T
    if y > 20:
        return 0
    # intermediate calculation
    def integrand(x):
        return x**2*(x**2+y**2)*np.exp(np.sqrt(x**2 + y**2))/(np.exp(np.sqrt(x**2 + y**2)) + sign)**2
    result, err = integrate.quad(integrand, 0.1, 20)
    return T**3*result/(2*np.pi**2)

def compute_entropy_density(T, m, sign):
    """Return the entropy for a particle with a single degree of freedom. 
    T and m in MeV. sign = 1: bose-einstein. sign=-1: fermi-dirac"""
    return (compute_energy_density(T, m, sign) + compute_pressure(T, m, sign))/T

def compute_QCD_dg_dT(gdof, T, T_qcd, w_qcd, sign):
    """Return dg/dT during the QCD phase transition for a particle type. 
    sign: 1=quark/gluon, -1=hadron"""
    return gdof*sign*np.exp(sign*(T_qcd-T)/w_qcd)*(np.exp(sign*(T_qcd-T)/w_qcd)+1)**-2/w_qcd

def compute_hubble_rate(energy_density):
    return np.sqrt((8*np.pi*c.grav/3)*energy_density)

def compute_SM_relativistic_dof_approx(T):
    """Calculate the relativistic degrees of freedom for the purposes of 
    computing the energy density. Valid up to arbitrarily high temperatures,
    and down to weak decoupling. Makes the approximation of g=0 once m>T."""

    #.....Statistical weights for massless or nearly massless particles (photons, neutrinos, gluons)
    gphot = 2. # Photon dofs
    gnuact = 3*2*7./8 # Active neutrino dofs: prior to weak decoupling 
    ggl = 16./(np.exp((c.T_qcd-T)/c.w_qcd)+1.) # Gluon dofs
    g_tot = gphot+gnuact+ggl
    
    # Massive particles approximation: simply set g_rel=0 when m<T. 
    # Also account for QCD phase transition
    for mass, stat, particle_class, gdof in zip(c.SM_masses, c.SM_stats, c.SM_particle_classes, c.SM_gdofs):
        if mass == 0:
            continue # already accounted for
        if mass>T:
            gdof = 0
        if(particle_class == 'q'): 
            if T>0.1*c.T_qcd:
                gdof=gdof/(np.exp((c.T_qcd-T)/c.w_qcd)+1.)
            # avoid overflow warnings 
            else:
                gdof=0
        elif(particle_class == 'h'):
            if T<10*c.T_qcd:
                gdof=gdof/(np.exp((T-c.T_qcd)/c.w_qcd)+1.)
            # avoid overflow warnings
            else:
                gdof=0
        if stat=='f':
            gdof *= 7./8
        g_tot += gdof
        
    return g_tot

# Compute the standard model temperature vs scale factor

def compute_T_SM_vs_a_conserving_entropy(a_domain, T_SM_initial, SM_entropy_density_func,
                                         log_Tmin, log_Tmax):
    """Compute SM temperature vs scale factor using
    conservation of co-moving entropy"""
    assert a_domain[0] == 1
    comoving_entropy_constant = SM_entropy_density_func(T_SM_initial)*1**3
    
    T_domain = []
    for a in a_domain:
        entropy_diff = lambda logT: np.abs(SM_entropy_density_func(10**logT)*a**3-comoving_entropy_constant)
        res = optimize.minimize(entropy_diff, [np.log10(T_SM_initial/a)], bounds=[(log_Tmin, log_Tmax)])
        T_domain.append(10**res.x[0])

    return T_domain

def compute_T_SM_vs_a_using_energy(a_domain, T_SM_initial, hubble_rate_func, SM_energy_density_func, 
                                   SM_pressure_func, SM_drho_dT_func):
    """Compute  SM temperature vs scale factor using
    the covariant energy conservation equation"""
    assert a_domain[0] == 1
    
    T_domain = integrate.odeint(
        compute_dT_SM_da, T_SM_initial, a_domain, 
        args=(hubble_rate_func, SM_energy_density_func, SM_pressure_func, SM_drho_dT_func)
        ).flatten()
         
    return T_domain

def compute_dT_SM_da(T_SM, a, hubble_rate_func, SM_energy_density_func, SM_pressure_func, SM_drho_dT_func):
    """The rate of change of the SM temeperature with scale factor"""
    return dTemperature_dtime(T_SM, hubble_rate_func, SM_energy_density_func, SM_pressure_func, SM_drho_dT_func)/(a*hubble_rate_func(T_SM))

def dTemperature_dtime(T_SM, hubble_rate_func, SM_energy_density_func, SM_pressure_func, SM_drho_dT_func):
    """Compute the time vs temperature relation in the SM plasma"""
    return -1*3*hubble_rate_func(T_SM)*(SM_energy_density_func(T_SM)+SM_pressure_func(T_SM))*SM_drho_dT_func(T_SM)**-1


# Compute the thermodynamic quantities and T vs scale factor relation and save both
if __name__ == '__main__':
    # Thermodynamic quantities
    log_T_min, log_T_max, num = 0, 6, 10**4
    print("Computing SM thermodynamic quantities for T in [{}, {}] MeV with {} data points".format(10**log_T_min, 10**log_T_max, num))
    T_domain = np.logspace(log_T_min, log_T_max, num)
    energy_density, pressure, drho_dT, entropy_density = [], [], [], []
    for T in T_domain:
        r, p, dr, e = compute_total_thermodynamic_quantities(T)
        energy_density.append(r)
        pressure.append(p)
        drho_dT.append(dr)
        entropy_density.append(e)
    hubble_rate = compute_hubble_rate(np.array(energy_density))
        
    results = {
        'T_domain':T_domain,
        'energy_density':energy_density,
        'pressure':pressure,
        'drho_dT':drho_dT,
        'entropy_density':entropy_density,
        'hubble_rate':hubble_rate
    }
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dir_path, 'data/SM_thermodynamic_quantities.pkl')
    with open(path, 'wb') as file:
        pickle.dump(results, file)

    # T vs scale factor relation
    log_a_min, log_a_max, num = 0, 6.1, 10**4
    print("Computing T vs a relation for a in [{}, {}] with {} data points".format(10**log_a_min, 10**log_a_max, num))
    print("Starting temperature for computation: {} MeV".format(10**log_T_max))
    a_domain = np.logspace(log_a_min, log_a_max, num)
    # Interpolate all of the SM thermodyanmic quantities
    SM_entropy_density_func = interpolate.interp1d(T_domain, entropy_density)
    SM_energy_density_func = interpolate.interp1d(T_domain, energy_density)
    SM_pressure_func = interpolate.interp1d(T_domain, pressure)
    SM_drho_dT_func = interpolate.interp1d(T_domain, drho_dT)
    hubble_rate_func = interpolate.interp1d(T_domain, hubble_rate)

    # Compute the T vs a relation using entropy conservation
    T_SM_entropy = compute_T_SM_vs_a_conserving_entropy(a_domain, 10**log_T_max, SM_entropy_density_func, log_T_min, log_T_max)
    # Compute the T vs a relation using energy conservation
    T_SM_energy = compute_T_SM_vs_a_using_energy(a_domain, 10**log_T_max, hubble_rate_func, SM_energy_density_func, SM_pressure_func, SM_drho_dT_func)

    results = {
            'a_domain':a_domain,
            'T_domain_entropy':T_SM_entropy,
            'T_domain_energy':T_SM_energy
        }
    path = os.path.join(dir_path, 'data/T_SM_vs_a.pkl')
    with open(path, 'wb') as file:
        pickle.dump(results, file)