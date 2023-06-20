import numpy as np

# Unit conversions
hbar = 6.582*10**-16 #eV s
MeVtoHz = 10**6/hbar
HztoMeV = hbar/10**6
charge = 1.602176634*10**-19 #C
MeVtoJ = 10**6*charge
JtoMeV = 10**-6/charge
kb =  1.380649*10**-23 #J/K 
KtoMeV = kb*JtoMeV
MeVtoK = 1/KtoMeV
eVtoMeV = 10**-6
c = 2.9979*10**8 #m/s
MeVtoinversecm = 10**6/(hbar*c*100)

# Fundamental constants, all in MeV unless otherwise noted
Gf = 1.166 * 10**-5 *10**-6 #MeV**-2
grav = (1.22*10**22)**-2 #MeV**-2 
thetaW = np.arcsin(np.sqrt(0.2229))
fine_structure = 7.297352*10**-3
eta_B = 6*10**-10 #baryon asymmetry
f_pi = 131 # for pion channel decay
# QCD Transition temperature and width
T_qcd = 170 
w_qcd = 10
hubble=0.7 # hubble factor
omega_dm_hsq = 0.120 #omega_m*h^2, planck 2018, error is 0.001
H100 =  100*3.24*10**-20*HztoMeV
Tcmb = 2.725*KtoMeV
TcmbtoTcnub = (4./11)**(1./3)
rho_crit_over_hsq = 3*H100**2/(8*np.pi*grav)
fine_structure = 7.297352*10**-3
f_pion_decay = 131 # for pion channel decay

# SM Masses, all in MeV

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