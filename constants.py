import numpy as np
from scipy import special

# Unit conversions
c = 2.9979*10**8 #m/s
hbar = 6.582*10**-16 #eV s
MeVtoHz = 10**6/hbar
HztoMeV = hbar/10**6
charge = 1.602176634*10**-19 #C
MeVtoJ = 10**6*charge
JtoMeV = 10**-6/charge
gtoMeV = 10**-3*c**2*JtoMeV
kb =  1.380649*10**-23 #J/K 
KtoMeV = kb*JtoMeV
MeVtoK = 1/KtoMeV
eVtoMeV = 10**-6
MeVtoinversecm = 10**6/(hbar*c*100)
inversecmtoMeV = 1/MeVtoinversecm

# SM Masses, all in MeV

# Leptons
m_e, m_mu, m_tau = 0.510,105.7,1776.9
# Quarks
m_u, m_d, m_s, m_c, m_b, m_t = 2.3,4.8,95,1290,4180,173500
# Gauge bosons
m_W, m_Z, m_H = 80377,91187.6,125025
#Baryons
m_p, m_n = 938,939
# Mesons
m_pi0, m_pipm, m_K0, m_Kpm, m_eta, m_etaprime, m_rhopm, m_rho0, m_omega = 134.9766,139.570,497.614,493.678,647.862,957.78,775.11,775.26,782.65

# Fundamental constants or measured quantities, all in MeV unless otherwise noted
grav = (1.22*10**22)**-2 #MeV**-2 
eta_B = 6*10**-10 #baryon asymmetry
f_pi = 131 # for pion channel decay
hubble=0.7 # hubble factor
omega_dm_hsq = 0.120 #omega_m*h^2, planck 2018, error is 0.001
H100 =  100*3.24*10**-20*HztoMeV
Tcmb = 2.725*KtoMeV
f_pion_decay = 131 # for pion channel decay
T_EW = 160*10**3 #EW phase transition temperature
fine_structure = 1/137.036

# EW derived quantities: following Schwartz pg. 588. 
e = np.sqrt(4*np.pi*fine_structure)
thetaW = np.arccos(m_W/m_Z)
sinsqthetaW = np.sin(thetaW)**2
g_SU2L = e/np.sin(thetaW)
g_U1Y = e/np.cos(thetaW)
fine_structure_W = g_SU2L**2/(4*np.pi)
higgs_vev = 2*m_W/g_SU2L
Gf = np.sqrt(2)*g_SU2L**2/(8*m_W**2) #=1.124*10^-5 GeV^-2. Differs from 1.166 low energy coeff due to choice of renormalization scale
# Done in this way to be consistent: ignoring renormalization corrections at few percent level. 

# Other useful quantities
mplanck = grav**-(1./2)
TcmbtoTcnub = (4./11)**(1./3)
Tcnub = Tcmb*TcmbtoTcnub
rho_crit_over_hsq = 3*H100**2/(8*np.pi*grav)
CnuB_ndens = 6*(3/2.)*special.zeta(3)*Tcnub**3/(2*np.pi**2) # Total CnuB number density today
# QCD Transition temperature and width
T_qcd = 170 
w_qcd = 10

# Particle properties for SM thermodynamic calculations

#.....Enter particle characteristics, in the following order of particles: 
#.....Leptons: e,mu,tau; Quarks: u,d,s,c,b,t; Gauge bosons (and Higgs): W+-,Z,H;
#.....Baryons: p,n; Mesons: pi0,pi+-,K0,K+-,eta,eta',rho+-,rho0,omega
#.....Massless particles: photon, all neutrinos, all gluons
#.....Particle masses in MeV
SM_masses = [m_e, m_mu, m_tau, m_u, m_d, m_s, m_c, m_b, m_t,m_W, m_Z,m_H,m_p,m_n,
            m_pi0, m_pipm, m_K0, m_Kpm, m_eta, m_etaprime, m_rhopm, m_rho0,m_omega,
            0,0,0]

SM_stats = ['f','f','f','f','f','f','f','f','f','b','b','b','f',
            'f','b','b','b','b','b','b','b','b','b','b','f','b']
#.....Particle class: 'l' for lepton, 'q' for quark, 'g' for gauge bosons (and Higgs), 'gl' for gluon, 'h' for hadron 
SM_particle_classes = ['l','l','l','q','q','q','q','q','q','g','g','g','h',
                       'h','h','h','h','h','h','h','h','h','h','g','l','gl']
SM_gdofs = [4.,4.,4.,12.,12.,12.,12.,12.,12.,6.,3.,1., 4.,4.,1.,2.,2.,2.,1.,1.,6.,3.,3.,2.,6.,16.]