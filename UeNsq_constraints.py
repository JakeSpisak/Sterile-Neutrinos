"""
UeNsq_constraints.py - 31/03/2023

Summary: 
Code for plotting constraints on the mixing squared between
the electron neutrino and sterile neutrino |U_{eN}|^2 as 
a function of the sterile neutrino mass m_N

References for each individual constraint are compiled
on the 'Plots and Data' page of the website.

Here data with consistent log units are loaded and plotted.

Requires numpy, matplotlib, scipy and pandas.
"""

import numpy as np
from numpy import cos as Cos
from numpy import sin as Sin
from numpy import sqrt as Sqrt
from numpy import ma
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib import ticker, cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.ndimage
import pandas as pd
from colour import Color

### Load data frame for each data set ###

df_Daya_Bay = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Daya_Bay_data.csv",header=None, sep=",", names = ["X", "Y"])
df_DANSS = pd.read_csv("data/heavy_sterile_constraints/current_Ue/DANSS_data.csv",header=None, sep=",", names = ["X", "Y"])
df_KamLAND = pd.read_csv("data/heavy_sterile_constraints/current_Ue/KamLAND_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Solar_KamLAND = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Solar_KamLAND_data.csv",header=None, sep=",", names = ["X", "Y"])
df_NEOS = pd.read_csv("data/heavy_sterile_constraints/current_Ue/NEOS_data.csv",header=None, sep=",", names = ["X", "Y"])
df_PROSPECT = pd.read_csv("data/heavy_sterile_constraints/current_Ue/PROSPECT_data.csv",header=None, sep=",", names = ["X", "Y"])
df_SKICDC = pd.read_csv("data/heavy_sterile_constraints/current_Ue/SKICDC_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Troitsk_2013 = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Troitsk_2013_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Troitsk_2017 = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Troitsk_2017_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Hiddemann = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Hiddemann_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Mainz = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Mainz_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Re187 = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Re187_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Ni63 = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Ni63_data.csv",header=None, sep=",", names = ["X", "Y"])
df_S35 = pd.read_csv("data/heavy_sterile_constraints/current_Ue/S35_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Cu64 = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Cu64_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Ca45 = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Ca45_data.csv",header=None, sep=",", names = ["X", "Y"])
df_CePr144 = pd.read_csv("data/heavy_sterile_constraints/current_Ue/CePr144_data.csv",header=None, sep=",", names = ["X", "Y"])
df_F20 = pd.read_csv("data/heavy_sterile_constraints/current_Ue/F20_data.csv",header=None, sep=",", names = ["X", "Y"])
df_BeEST = pd.read_csv("data/heavy_sterile_constraints/current_Ue/BeEST_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Rovno = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Rovno_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Bugey = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Bugey_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Borexino = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Borexino_data.csv",header=None, sep=",", names = ["X", "Y"])
df_PIENU = pd.read_csv("data/heavy_sterile_constraints/current_Ue/PIENU_data.csv",header=None, sep=",", names = ["X", "Y"])
df_PionDecay = pd.read_csv("data/heavy_sterile_constraints/current_Ue/PionDecay_data.csv",header=None, sep=",", names = ["X", "Y"])
df_PS191 = pd.read_csv("data/heavy_sterile_constraints/current_Ue/PS191_data.csv",header=None, sep=",", names = ["X", "Y"])
df_PS191_Re = pd.read_csv("data/heavy_sterile_constraints/current_Ue/PS191_Re_data.csv",header=None, sep=",", names = ["X", "Y"])
df_JINR = pd.read_csv("data/heavy_sterile_constraints/current_Ue/JINR_data.csv",header=None, sep=",", names = ["X", "Y"])
df_NA62 = pd.read_csv("data/heavy_sterile_constraints/current_Ue/NA62_data.csv",header=None, sep=",", names = ["X", "Y"])
df_T2K = pd.read_csv("data/heavy_sterile_constraints/current_Ue/T2K_data.csv",header=None, sep=",", names = ["X", "Y"])
df_SuperK = pd.read_csv("data/heavy_sterile_constraints/current_Ue/SuperK_data.csv",header=None, sep=",", names = ["X", "Y"])
df_MesonDecays_LNV = pd.read_csv("data/heavy_sterile_constraints/current_Ue/MesonDecays_LNV_data.csv",header=None, sep=",", names = ["X", "Y"])
df_CHARM = pd.read_csv("data/heavy_sterile_constraints/current_Ue/CHARM_data.csv",header=None, sep=",", names = ["X", "Y"])
df_BEBC = pd.read_csv("data/heavy_sterile_constraints/current_Ue/BEBC_data.csv",header=None, sep=",", names = ["X", "Y"])
df_BESIII_1 = pd.read_csv("data/heavy_sterile_constraints/current_Ue/BESIII_1_data.csv",header=None, sep=",", names = ["X", "Y"])
df_BESIII_2 = pd.read_csv("data/heavy_sterile_constraints/current_Ue/BESIII_2_data.csv",header=None, sep=",", names = ["X", "Y"])
df_NA3 = pd.read_csv("data/heavy_sterile_constraints/current_Ue/NA3_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Belle = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Belle_data.csv",header=None, sep=",", names = ["X", "Y"])
df_DELPHI = pd.read_csv("data/heavy_sterile_constraints/current_Ue/DELPHI_data.csv",header=None, sep=",", names = ["X", "Y"])
df_L3_1 = pd.read_csv("data/heavy_sterile_constraints/current_Ue/L3_1_data.csv",header=None, sep=",", names = ["X", "Y"])
df_L3_2 = pd.read_csv("data/heavy_sterile_constraints/current_Ue/L3_2_data.csv",header=None, sep=",", names = ["X", "Y"])
df_ATLAS = pd.read_csv("data/heavy_sterile_constraints/current_Ue/ATLAS_data.csv",header=None, sep=",", names = ["X", "Y"])
df_ATLAS_LNV = pd.read_csv("data/heavy_sterile_constraints/current_Ue/ATLAS_LNV_data.csv",header=None, sep=",", names = ["X", "Y"])
df_ATLAS_LNC = pd.read_csv("data/heavy_sterile_constraints/current_Ue/ATLAS_LNC_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Higgs = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Higgs_data.csv",header=None, sep=",", names = ["X", "Y"])
df_CMS_SameSign = pd.read_csv("data/heavy_sterile_constraints/current_Ue/CMS_SameSign_data.csv",header=None, sep=",", names = ["X", "Y"])
df_CMS_TriLepton = pd.read_csv("data/heavy_sterile_constraints/current_Ue/CMS_TriLepton_data.csv",header=None, sep=",", names = ["X", "Y"])
df_EWPD = pd.read_csv("data/heavy_sterile_constraints/current_Ue/EWPD_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Planck = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Planck_data.csv",header=None, sep=",", names = ["X", "Y"])
df_CMB = pd.read_csv("data/heavy_sterile_constraints/current_Ue/CMB_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Xray = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Xray_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Xray_2 = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Xray_2_data.csv",header=None, sep=",", names = ["X", "Y"])
df_CMB_Linear = pd.read_csv("data/heavy_sterile_constraints/current_Ue/CMB_Linear_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Zhou_SN = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Zhou_SN_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Zhou_SN_2 = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Zhou_SN_2_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Raffelt_SN = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Raffelt_SN_data.csv",header=None, sep=",", names = ["X", "Y"])
df_ShiSigl_SN = pd.read_csv("data/heavy_sterile_constraints/current_Ue/ShiSigl_SN_data.csv",header=None, sep=",", names = ["X", "Y"])
df_Valle_SN = pd.read_csv("data/heavy_sterile_constraints/current_Ue/Valle_SN_data.csv",header=None, sep=",", names = ["X", "Y"])
df_CMB_BAO_H = pd.read_csv("data/heavy_sterile_constraints/current_Ue/CMB_BAO_H_data.csv",header=None, sep=",", names = ["X", "Y"])
df_CMB_H_only = pd.read_csv("data/heavy_sterile_constraints/current_Ue/CMB_H_only_data.csv",header=None, sep=",", names = ["X", "Y"])
df_BBN = pd.read_csv("data/heavy_sterile_constraints/current_Ue/BBN_data.csv",header=None, sep=",", names = ["X", "Y"])

### Read to (x,y) = (m_N,|V_{eN}|^2)  ####

x_Daya_Bay, y_Daya_Bay = [], []
for i in range(len(df_Daya_Bay.index)):
    x_Daya_Bay.append(df_Daya_Bay.iloc[i]['X'])
    y_Daya_Bay.append(df_Daya_Bay.iloc[i]['Y'])

x_DANSS, y_DANSS = [], []
for i in range(len(df_DANSS.index)):
    x_DANSS.append(df_DANSS.iloc[i]['X'])
    y_DANSS.append(df_DANSS.iloc[i]['Y'])

x_KamLAND, y_KamLAND = [], []
for i in range(len(df_KamLAND.index)):
    x_KamLAND.append(df_KamLAND.iloc[i]['X'])
    y_KamLAND.append(df_KamLAND.iloc[i]['Y'])

x_Solar_KamLAND, y_Solar_KamLAND = [], []
for i in range(len(df_Solar_KamLAND.index)):
    x_Solar_KamLAND.append(df_Solar_KamLAND.iloc[i]['X'])
    y_Solar_KamLAND.append(df_Solar_KamLAND.iloc[i]['Y'])

x_NEOS, y_NEOS = [], []
for i in range(len(df_NEOS.index)):
    x_NEOS.append(df_NEOS.iloc[i]['X'])
    y_NEOS.append(df_NEOS.iloc[i]['Y'])

x_PROSPECT, y_PROSPECT = [], []
for i in range(len(df_PROSPECT.index)):
    x_PROSPECT.append(df_PROSPECT.iloc[i]['X'])
    y_PROSPECT.append(df_PROSPECT.iloc[i]['Y'])

x_SKICDC, y_SKICDC = [], []
for i in range(len(df_SKICDC.index)):
    x_SKICDC.append(df_SKICDC.iloc[i]['X'])
    y_SKICDC.append(df_SKICDC.iloc[i]['Y'])

x_Troitsk_2013, y_Troitsk_2013 = [], []
for i in range(len(df_Troitsk_2013.index)):
    x_Troitsk_2013.append(df_Troitsk_2013.iloc[i]['X'])
    y_Troitsk_2013.append(df_Troitsk_2013.iloc[i]['Y'])

x_Troitsk_2017, y_Troitsk_2017 = [], []
for i in range(len(df_Troitsk_2017.index)):
    x_Troitsk_2017.append(df_Troitsk_2017.iloc[i]['X'])
    y_Troitsk_2017.append(df_Troitsk_2017.iloc[i]['Y'])

x_Tritium, y_Tritium = [], []
for i in range(len(df_Troitsk_2013.index)):
    x_Tritium.append(df_Troitsk_2013.iloc[i]['X'])
    y_Tritium.append(df_Troitsk_2013.iloc[i]['Y'])
for i in range(len(df_Troitsk_2017.index)):
    x_Tritium.append(df_Troitsk_2017.iloc[i]['X'])
    y_Tritium.append(df_Troitsk_2017.iloc[i]['Y'])

x_Hiddemann, y_Hiddemann = [], []
for i in range(len(df_Hiddemann.index)):
    x_Hiddemann.append(df_Hiddemann.iloc[i]['X'])
    y_Hiddemann.append(df_Hiddemann.iloc[i]['Y'])

x_Mainz, y_Mainz = [], []
for i in range(len(df_Mainz.index)):
    x_Mainz.append(df_Mainz.iloc[i]['X'])
    y_Mainz.append(df_Mainz.iloc[i]['Y'])

x_Re187, y_Re187 = [], []
for i in range(len(df_Re187.index)):
    x_Re187.append(df_Re187.iloc[i]['X'])
    y_Re187.append(df_Re187.iloc[i]['Y'])

x_Ni63, y_Ni63 = [], []
for i in range(len(df_Ni63.index)):
    x_Ni63.append(df_Ni63.iloc[i]['X'])
    y_Ni63.append(df_Ni63.iloc[i]['Y'])

x_S35, y_S35 = [], []
for i in range(len(df_S35.index)):
    x_S35.append(df_S35.iloc[i]['X'])
    y_S35.append(df_S35.iloc[i]['Y'])

x_Cu64, y_Cu64 = [], []
for i in range(len(df_Cu64.index)):
    x_Cu64.append(df_Cu64.iloc[i]['X'])
    y_Cu64.append(df_Cu64.iloc[i]['Y'])

x_Ca45, y_Ca45 = [], []
for i in range(len(df_Ca45.index)):
    x_Ca45.append(df_Ca45.iloc[i]['X'])
    y_Ca45.append(df_Ca45.iloc[i]['Y'])

x_CePr144, y_CePr144 = [], []
for i in range(len(df_CePr144.index)):
    x_CePr144.append(df_CePr144.iloc[i]['X'])
    y_CePr144.append(df_CePr144.iloc[i]['Y'])

x_F20, y_F20 = [], []
for i in range(len(df_F20.index)):
    x_F20.append(df_F20.iloc[i]['X'])
    y_F20.append(df_F20.iloc[i]['Y'])

x_BeEST, y_BeEST    = [], []
for i in range(len(df_BeEST.index)):
    x_BeEST.append(df_BeEST.iloc[i]['X'])
    y_BeEST.append(df_BeEST.iloc[i]['Y'])

x_Rovno, y_Rovno = [], []
for i in range(len(df_Rovno.index)):
    x_Rovno.append(df_Rovno.iloc[i]['X'])
    y_Rovno.append(df_Rovno.iloc[i]['Y'])

x_Bugey, y_Bugey = [], []
for i in range(len(df_Bugey.index)):
    x_Bugey.append(df_Bugey.iloc[i]['X'])
    y_Bugey.append(df_Bugey.iloc[i]['Y'])

x_Borexino, y_Borexino = [], []
for i in range(len(df_Borexino.index)):
    x_Borexino.append(df_Borexino.iloc[i]['X'])
    y_Borexino.append(df_Borexino.iloc[i]['Y'])

x_PionDecay, y_PionDecay = [], []
for i in range(len(df_PionDecay.index)):
    x_PionDecay.append(df_PionDecay.iloc[i]['X'])
    y_PionDecay.append(df_PionDecay.iloc[i]['Y'])

x_PIENU, y_PIENU = [], []
for i in range(len(df_PIENU.index)):
    x_PIENU.append(df_PIENU.iloc[i]['X'])
    y_PIENU.append(df_PIENU.iloc[i]['Y'])

x_PionDecay_tot, y_PionDecay_tot = [], []
for i in range(len(df_PionDecay.index)):
    x_PionDecay_tot.append(df_PionDecay.iloc[i]['X'])
    y_PionDecay_tot.append(df_PionDecay.iloc[i]['Y'])
for i in range(len(df_PIENU.index)):
    x_PionDecay_tot.append(df_PIENU.iloc[i]['X'])
    y_PionDecay_tot.append(df_PIENU.iloc[i]['Y'])

x_PS191, y_PS191 = [], []
for i in range(len(df_PS191.index)):
    x_PS191.append(df_PS191.iloc[i]['X'])
    y_PS191.append(df_PS191.iloc[i]['Y'])

x_PS191_Re, y_PS191_Re = [], []
for i in range(len(df_PS191_Re.index)):
    x_PS191_Re.append(df_PS191_Re.iloc[i]['X'])
    y_PS191_Re.append(df_PS191_Re.iloc[i]['Y'])

x_JINR, y_JINR = [], []
for i in range(len(df_JINR.index)):
    x_JINR.append(df_JINR.iloc[i]['X'])
    y_JINR.append(df_JINR.iloc[i]['Y'])

x_NA62, y_NA62 = [], []
for i in range(len(df_NA62.index)):
    x_NA62.append(df_NA62.iloc[i]['X'])
    y_NA62.append(df_NA62.iloc[i]['Y'])

x_T2K, y_T2K = [], []
for i in range(len(df_T2K.index)):
    x_T2K.append(df_T2K.iloc[i]['X'])
    y_T2K.append(df_T2K.iloc[i]['Y'])

x_SuperK, y_SuperK = [], []
for i in range(len(df_SuperK.index)):
    x_SuperK.append(df_SuperK.iloc[i]['X'])
    y_SuperK.append(df_SuperK.iloc[i]['Y'])

x_MesonDecays_LNV, y_MesonDecays_LNV = [], []
for i in range(len(df_MesonDecays_LNV.index)):
    x_MesonDecays_LNV.append(df_MesonDecays_LNV.iloc[i]['X'])
    y_MesonDecays_LNV.append(df_MesonDecays_LNV.iloc[i]['Y'])

x_CHARM, y_CHARM = [], []
for i in range(len(df_CHARM.index)):
    x_CHARM.append(df_CHARM.iloc[i]['X'])
    y_CHARM.append(df_CHARM.iloc[i]['Y'])

x_BEBC, y_BEBC = [], []
for i in range(len(df_BEBC.index)):
    x_BEBC.append(df_BEBC.iloc[i]['X'])
    y_BEBC.append(df_BEBC.iloc[i]['Y'])

x_BESIII_1, y_BESIII_1 = [], []
for i in range(len(df_BESIII_1.index)):
    x_BESIII_1.append(df_BESIII_1.iloc[i]['X'])
    y_BESIII_1.append(df_BESIII_1.iloc[i]['Y'])

x_BESIII_2, y_BESIII_2 = [], []
for i in range(len(df_BESIII_2.index)):
    x_BESIII_2.append(df_BESIII_2.iloc[i]['X'])
    y_BESIII_2.append(df_BESIII_2.iloc[i]['Y'])

x_NA3, y_NA3 = [], []
for i in range(len(df_NA3.index)):
    x_NA3.append(df_NA3.iloc[i]['X'])
    y_NA3.append(df_NA3.iloc[i]['Y'])

x_Belle, y_Belle = [], []
for i in range(len(df_Belle.index)):
    x_Belle.append(df_Belle.iloc[i]['X'])
    y_Belle.append(df_Belle.iloc[i]['Y'])

x_DELPHI, y_DELPHI = [], []
for i in range(len(df_DELPHI.index)):
    x_DELPHI.append(df_DELPHI.iloc[i]['X'])
    y_DELPHI.append(df_DELPHI.iloc[i]['Y'])

x_L3_1, y_L3_1 = [], []
for i in range(len(df_L3_1.index)):
    x_L3_1.append(df_L3_1.iloc[i]['X'])
    y_L3_1.append(df_L3_1.iloc[i]['Y'])

x_L3_2, y_L3_2 = [], []
for i in range(len(df_L3_2.index)):
    x_L3_2.append(df_L3_2.iloc[i]['X'])
    y_L3_2.append(df_L3_2.iloc[i]['Y'])

x_ATLAS, y_ATLAS = [], []
for i in range(len(df_ATLAS.index)):
    x_ATLAS.append(df_ATLAS.iloc[i]['X'])
    y_ATLAS.append(df_ATLAS.iloc[i]['Y'])

x_ATLAS_LNV, y_ATLAS_LNV = [], []
for i in range(len(df_ATLAS_LNV.index)):
    x_ATLAS_LNV.append(df_ATLAS_LNV.iloc[i]['X'])
    y_ATLAS_LNV.append(df_ATLAS_LNV.iloc[i]['Y'])

x_ATLAS_LNC, y_ATLAS_LNC = [], []
for i in range(len(df_ATLAS_LNC.index)):
    x_ATLAS_LNC.append(df_ATLAS_LNC.iloc[i]['X'])
    y_ATLAS_LNC.append(df_ATLAS_LNC.iloc[i]['Y'])

x_Higgs, y_Higgs = [], []
for i in range(len(df_Higgs.index)):
    x_Higgs.append(df_Higgs.iloc[i]['X'])
    y_Higgs.append(df_Higgs.iloc[i]['Y'])

x_CMS_SameSign, y_CMS_SameSign = [], []
for i in range(len(df_CMS_SameSign.index)):
    x_CMS_SameSign.append(df_CMS_SameSign.iloc[i]['X'])
    y_CMS_SameSign.append(df_CMS_SameSign.iloc[i]['Y'])

x_CMS_TriLepton, y_CMS_TriLepton = [], []
for i in range(len(df_CMS_TriLepton.index)):
    x_CMS_TriLepton.append(df_CMS_TriLepton.iloc[i]['X'])
    y_CMS_TriLepton.append(df_CMS_TriLepton.iloc[i]['Y'])

x_EWPD, y_EWPD = [], []
for i in range(len(df_EWPD.index)):
    x_EWPD.append(df_EWPD.iloc[i]['X'])
    y_EWPD.append(df_EWPD.iloc[i]['Y'])

x_Planck, y_Planck = [], []
for i in range(len(df_Planck.index)):
    x_Planck.append(df_Planck.iloc[i]['X'])
    y_Planck.append(df_Planck.iloc[i]['Y'])

x_CMB, y_CMB, z_CMB = [], [], []
for i in range(len(df_CMB.index)):
    x_CMB.append(df_CMB.iloc[i]['X'])
    y_CMB.append(df_CMB.iloc[i]['Y'])
    z_CMB.append(df_CMB.iloc[i]['Y']+0.7)

x_CMB_Linear, y_CMB_Linear, z_CMB_Linear = [], [], []
for i in range(len(df_CMB_Linear.index)):
    x_CMB_Linear.append(df_CMB_Linear.iloc[i]['X'])
    y_CMB_Linear.append(df_CMB_Linear.iloc[i]['Y'])
    z_CMB_Linear.append(df_CMB_Linear.iloc[i]['Y']+0.7)

x_CMB_BAO_H, y_CMB_BAO_H  = [], []
for i in range(len(df_CMB_BAO_H.index)):
    x_CMB_BAO_H.append(df_CMB_BAO_H.iloc[i]['X'])
    y_CMB_BAO_H.append(df_CMB_BAO_H.iloc[i]['Y'])

x_CMB_BAO_H_2, y_CMB_BAO_H_2  = [], []
for i in range(len(df_CMB_BAO_H.index)):
    x_CMB_BAO_H_2.append(df_CMB_BAO_H.iloc[i]['X'])
    y_CMB_BAO_H_2.append(df_CMB_BAO_H.iloc[i]['Y'])
for i in range(1,len(df_CMB_BAO_H.index)+1):
    x_CMB_BAO_H_2.append(df_CMB_BAO_H.iloc[-i]['X']+0.5)
    y_CMB_BAO_H_2.append(df_CMB_BAO_H.iloc[-i]['Y']+0.5)

x_CMB_H_only, y_CMB_H_only  = [], []
for i in range(len(df_CMB_H_only.index)):
    x_CMB_H_only.append(df_CMB_H_only.iloc[i]['X'])
    y_CMB_H_only.append(df_CMB_H_only.iloc[i]['Y'])

x_BBN, y_BBN  = [], []
for i in range(len(df_BBN.index)):
    x_BBN.append(df_BBN.iloc[i]['X'])
    y_BBN.append(df_BBN.iloc[i]['Y'])
x_BBN.append(-2.16)
y_BBN.append(0.1)

x_BBN_2, y_BBN_2  = [], []
x_BBN_2.append(-2.16 - 0.5)
y_BBN_2.append(0.1)
for i in range(1,len(df_BBN.index)+1):
    x_BBN_2.append(df_BBN.iloc[-i]['X'] - 0.5)
    y_BBN_2.append(df_BBN.iloc[-i]['Y'] - 0.5)
for i in range(len(df_BBN.index)):
    x_BBN_2.append(df_BBN.iloc[i]['X'])
    y_BBN_2.append(df_BBN.iloc[i]['Y'])
x_BBN_2.append(-2.16)
y_BBN_2.append(0.1)

x_Xray, y_Xray, z_Xray = [], [], []
for i in range(len(df_Xray.index)):
    x_Xray.append(df_Xray.iloc[i]['X'])
    y_Xray.append(df_Xray.iloc[i]['Y'])
    z_Xray.append(df_Xray.iloc[i]['Y']+0.9)

x_Xray_2, y_Xray_2 = [], []
for i in range(len(df_Xray_2.index)):
    x_Xray_2.append(df_Xray_2.iloc[i]['X'])
    y_Xray_2.append(df_Xray_2.iloc[i]['Y'])

x_Xray_2_shift, y_Xray_2_shift = [], []
for i in range(1,len(df_Xray_2.index)+1):
    x_Xray_2_shift.append(df_Xray_2.iloc[-i]['X'])
    y_Xray_2_shift.append(df_Xray_2.iloc[-i]['Y']+0.7)
for i in range(len(df_Xray_2.index)):
    x_Xray_2_shift.append(df_Xray_2.iloc[i]['X'])
    y_Xray_2_shift.append(df_Xray_2.iloc[i]['Y'])

x_Zhou_SN, y_Zhou_SN = [], []
for i in range(len(df_Zhou_SN.index)):
    x_Zhou_SN.append(df_Zhou_SN.iloc[i]['X'])
    y_Zhou_SN.append(df_Zhou_SN.iloc[i]['Y'])

x_Zhou_SN_shift, y_Zhou_SN_shift, z_Zhou_SN_shift = [], [], []
for i in range(43+1):
    x_Zhou_SN_shift.append(df_Zhou_SN.iloc[i]['Y']-9)
    y_Zhou_SN_shift.append(df_Zhou_SN.iloc[i]['X'])
    z_Zhou_SN_shift.append(df_Zhou_SN.iloc[i]['Y']+0.25)

x_Zhou_SN_shift2, y_Zhou_SN_shift2, z_Zhou_SN_shift2 = [], [], []
for i in range(len(df_Zhou_SN.index)-30,len(df_Zhou_SN.index)):
    x_Zhou_SN_shift2.append(df_Zhou_SN.iloc[i]['X'])
    y_Zhou_SN_shift2.append(df_Zhou_SN.iloc[i]['Y'])
    z_Zhou_SN_shift2.append(df_Zhou_SN.iloc[i]['Y']+0.5)

x_Zhou_SN_2, y_Zhou_SN_2, z_Zhou_SN_2 = [], [], []
for i in range(len(df_Zhou_SN_2.index)-3):
    x_Zhou_SN_2.append(df_Zhou_SN_2.iloc[i]['X'])
    y_Zhou_SN_2.append(df_Zhou_SN_2.iloc[i]['Y'])
    z_Zhou_SN_2.append(df_Zhou_SN_2.iloc[i]['Y']-0.3)

x_Raffelt_SN, y_Raffelt_SN = [], []
for i in range(len(df_Raffelt_SN.index)):
    x_Raffelt_SN.append(df_Raffelt_SN.iloc[i]['X'])
    y_Raffelt_SN.append(df_Raffelt_SN.iloc[i]['Y'])

x_ShiSigl_SN, y_ShiSigl_SN = [], []
for i in range(len(df_ShiSigl_SN.index)):
    x_ShiSigl_SN.append(df_ShiSigl_SN.iloc[i]['X'])
    y_ShiSigl_SN.append(df_ShiSigl_SN.iloc[i]['Y'])
x_ShiSigl_SN.append(x_ShiSigl_SN[0])
y_ShiSigl_SN.append(y_ShiSigl_SN[0])

x_Valle_SN, y_Valle_SN = [], []
for i in range(len(df_Valle_SN.index)):
    x_Valle_SN.append(df_Valle_SN.iloc[i]['X'])
    y_Valle_SN.append(df_Valle_SN.iloc[i]['Y'])

fig, axes = plt.subplots(nrows=1, ncols=1)

spacing=0.2
m = np.arange(-12,6+spacing, spacing)
age_bound = np.log10(1.1 * 10**(-7) * 10**(-45) * ((50 * 10**(3))/10**(m))**5)
bbn_bound = np.log10(1.00*10**(34) * 10**(-45) * (1/10**(m))**5)
seesaw_bound = np.log10(0.05*10**(-9)/10**(m))

### Plot Current Constraints ###

axes.plot(x_Daya_Bay,y_Daya_Bay,linewidth=1.5,linestyle='-',color='mediumorchid') # Daya bay
# axes.plot(x_DANSS,y_DANSS,linewidth=1.5,linestyle='-',color='darkred') # DANSS
# axes.plot(x_KamLAND,y_KamLAND,linewidth=1.5,linestyle='-',color='orangered') # Solar and KamLAND
# axes.plot(x_Solar_KamLAND,y_Solar_KamLAND,linewidth=1.5,linestyle='-',color='green') # Solar and KamLAND
axes.plot(x_NEOS,y_NEOS,linewidth=1.5,linestyle='-',color='forestgreen') # NEOS
axes.plot(x_PROSPECT,y_PROSPECT,linewidth=1.5,linestyle='-',color='darkturquoise') # PROSPECT
axes.plot(x_SKICDC,y_SKICDC,linewidth=1.5,linestyle='--',color='darkorange') # SuperK + IceCube + DeepCore

axes.plot(x_Tritium,y_Tritium,linewidth=1.5,linestyle='-',color='lime') # Troitsk 2013 Tritium (and Nu--Mass)
# axes.plot(x_Mainz,y_Mainz,linewidth=1.5,linestyle='-',color='lime') # Mainz Tritium
# axes.plot(x_Hiddemann,y_Hiddemann,linewidth=1.5,linestyle='-',color='b') # Hiddemannn Tritium
axes.plot(x_Re187,y_Re187,linewidth=1.5,linestyle='-',color='violet') # Re--187
axes.plot(x_Ni63,y_Ni63,linewidth=1.5,linestyle='-',color='darkmagenta') # Ni--63
axes.plot(x_S35,y_S35,linewidth=1.5,linestyle='-',color='y') # S--35
axes.plot(x_Ca45,y_Ca45,linewidth=1.5,linestyle='-',color='plum') # Ca--45
axes.plot(x_Cu64,y_Cu64,linewidth=1.5,linestyle='-',color='darkcyan') # Cu--64
axes.plot(x_CePr144,y_CePr144,linewidth=1.5,linestyle='--',color='crimson') # Ce Pr--144
axes.plot(x_F20,y_F20,linewidth=1.5,linestyle='--',color='green') # F--20
axes.plot(x_BeEST,y_BeEST,linewidth=1.5,linestyle='-',color='rosybrown') # Be--7

axes.plot(x_Rovno,y_Rovno,linewidth=1.5,linestyle='--',color='tomato') # Rovno reactor decay
axes.plot(x_Bugey,y_Bugey,linewidth=1.5,linestyle='--',color='turquoise') # Buguy reactor decay
axes.plot(x_Borexino,y_Borexino,linewidth=1.5,linestyle='-.',color='blue') # BOREXINO
axes.plot(x_PionDecay_tot,y_PionDecay_tot,linewidth=1.5,linestyle='-',color='gold') # Pion decay low and PIENU
axes.plot(x_PS191,y_PS191,linewidth=1.5,linestyle='--',color='magenta') # PS191
# axes.plot(x_PS191_Re,y_PS191_Re,linewidth=1.5,linestyle='-',color='purple') # PS191 Reanalysis
axes.plot(x_JINR,y_JINR,linewidth=1.5,linestyle='--',color='sienna') # JINR
axes.plot(x_NA62,y_NA62,linewidth=1.5,linestyle='-',color='teal') # NA62 Constraints New
axes.plot(x_T2K,y_T2K,linewidth=1.5,linestyle='-',color='purple') # T2K ND
axes.plot(x_SuperK,y_SuperK,linewidth=1.5,linestyle='-',color='mediumseagreen') # Super-Kamiokande
axes.plot(x_MesonDecays_LNV,y_MesonDecays_LNV,linewidth=1.5,linestyle='-',color='mediumaquamarine') # LNV meson decays
axes.plot(x_CHARM,y_CHARM,linewidth=1.5,linestyle='-.',color='darkslategray') # CHARM
axes.plot(x_BEBC,y_BEBC,linewidth=1.5,linestyle='-',color='yellowgreen') # BEBC
# axes.plot(x_BESIII_1,x_BESIII_1,linewidth=1.5,linestyle='-',color='r') # BESIII D0 decays
axes.plot(x_BESIII_2,y_BESIII_2,linewidth=1.5,linestyle='-',color='c') # BESIII D+ decays

axes.plot(x_NA3,y_NA3,linewidth=1.5,linestyle='-',color='cornsilk') # NA3
axes.plot(x_Belle,y_Belle,linewidth=1.5,linestyle='-',color='darkgreen') # BELLE
axes.plot(x_DELPHI,y_DELPHI,linewidth=1.5,linestyle='--',color='teal') # DEPLHI
axes.plot(x_L3_1,y_L3_1,linewidth=1.5,linestyle='--',color='salmon') # L3
axes.plot(x_L3_2,y_L3_2,linewidth=1.5,linestyle='--',color='salmon') # L3
axes.plot(x_ATLAS,y_ATLAS,linewidth=1.5,linestyle='--',color='mediumblue') # ALTLAS dilepton + jets
# axes.plot(x_ATLAS_LNC,y_ATLAS_LNC,linewidth=1.5,linestyle='--',color='mediumblue') # ALTLAS prompt LNC
axes.plot(x_ATLAS_LNV,y_ATLAS_LNV,linewidth=1.5,linestyle='--',color='mediumblue') # ALTLAS trilepton prompt LNV
axes.plot(x_Higgs,y_Higgs,linewidth=1.5,linestyle='--',color='sandybrown') # Higgs Constraints
# axes.plot(x_CMS_SameSign,y_CMS_SameSign,linewidth=1.5,linestyle='--',color='r') # CMS Same Sign LNV
axes.plot(x_CMS_TriLepton,y_CMS_TriLepton,linewidth=1.5,linestyle='--',color='red') # CMS trilepton
axes.plot(x_EWPD,y_EWPD,linewidth=1.5,linestyle='--',color='darkgoldenrod') # Electroweak precision data

#axes.plot(x_CMB,y_CMB,linewidth=1,linestyle='-.',color='dimgrey') # Evans data
#axes.plot(x_CMB_Linear,y_CMB_Linear,linewidth=1,linestyle='-.',color='dimgrey') # Linear CMB
axes.plot(x_CMB_BAO_H,y_CMB_BAO_H,linewidth=0.5,linestyle='-',color='grey') # # Decay after BBN constraints
#axes.plot(x_CMB_H_only,y_CMB_H_only,linewidth=1.5,linestyle='--',color='red') # Decay after BBN, Hubble only
axes.plot(x_BBN,y_BBN,linewidth=0.5,linestyle='-',color='grey') # Decay before BBN constraints
#axes.plot(x_Xray,y_Xray,linewidth=1.5,linestyle='-',color='orangered') # Combined X-ray observations OLD
axes.plot(x_Xray_2,y_Xray_2,linewidth=1.5,linestyle='-',color='orangered') # Combined X-ray data
# axes.plot([2.986328125-9,2.986328125-9],[1,-13],linewidth=0.5,linestyle='--',color='black') # Tremaine-Gunn / Lyman-alpha
#axes.plot(x_Kopp_SN,y_Kopp_SN,linewidth=1.5,linestyle=':',color='darkslategrey',alpha=0.15) # Kopp Supernova constraints
#axes.plot(x_Kopp_SN_2,y_Kopp_SN_2,linewidth=1.5,linestyle=':',color='darkslateblue',alpha=0.15) # Kopp Supernova constraints 2
#axes.plot(x_Zhou_SN,y_Zhou_SN,linewidth=1.5,linestyle=':',color='darkslateblue',alpha=0.15) # Zhou Supernova constraints 1
#axes.plot(x_Zhou_SN_2,y_Zhou_SN_2,linewidth=1.5,linestyle=':',color='darkslateblue',alpha=0.15) # Zhou Supernova constraints 2
#axes.plot(x_Raffelt_SN,y_Raffelt_SN,linewidth=1.5,linestyle=':',color='darkslateblue',alpha=0.15) # Raffelt and Zhou Supernova constraints
axes.plot(x_ShiSigl_SN,y_ShiSigl_SN,linewidth=1.5,linestyle=':',color='darkslateblue',alpha=0.15) # Shi and Sigl Supernova constraints
#axes.plot(x_Valle_SN,y_Valle_SN,linewidth=1.5,linestyle=':',color='darkslateblue',alpha=0.15) # Valle Supernova r nucleosynthesis

# axes.plot(x_Planck,y_Planck,linewidth=1.5,linestyle='--',color='m') # Planck
# axes.plot([np.log10(0.5*10**-6),np.log10(0.5*10**-6)],[-13,1],linewidth=1.5,linestyle='-.',color='black') # Tremaine-Gunn bound (assumes HNL is DM)

# axes.plot(m,age_bound,linewidth=1.5,linestyle='--',color='r') # Age of the universe decay
# axes.plot(m,bbn_bound,linewidth=1.5,linestyle='--',color='r') # BBN decay
axes.plot(m,seesaw_bound,linewidth=1,linestyle='dotted',color='black') # Seesaw line

### Shading ###

plt.fill_between(x_Daya_Bay,0.1,y_Daya_Bay, facecolor='mediumorchid', alpha=0.075,lw=0)
plt.fill_between(x_PROSPECT,0.1,y_PROSPECT, facecolor='darkturquoise', alpha=0.075,lw=0)
plt.fill_between(x_NEOS,0.1,y_NEOS, facecolor='forestgreen', alpha=0.075,lw=0)
# plt.fill_between(x_SKICDC,y_SKICDC, facecolor='crimson', alpha=0.075,lw=0)
plt.fill_between(x_Tritium,0.1,y_Tritium, facecolor='lime', alpha=0.075)
# plt.fill_between(x_Mainz,y_Mainz, facecolor='navy', alpha=0.075)
plt.fill_between(x_Re187,0.1,y_Re187, facecolor='violet', alpha=0.075)
plt.fill_between(x_Ni63,0.1,y_Ni63, facecolor='darkmagenta', alpha=0.075)
plt.fill_between(x_S35,0.1,y_S35, facecolor='y', alpha=0.075)
# plt.fill_between(x_Ca45,0.1,y_Ca45, facecolor='plum', alpha=0.075)
# plt.fill_between(x_Cu64,0.1,y_Cu64, facecolor='mediumvioletred', alpha=0.075)
plt.fill_between(x_CePr144,0.1,y_CePr144, facecolor='crimson', alpha=0.075)
plt.fill_between(x_F20,0.1,y_F20, facecolor='green', alpha=0.075)
plt.fill_between(x_BeEST,0.1,y_BeEST, facecolor='rosybrown', alpha=0.075)
# plt.fill_between(x_Rovno,0.1,y_Rovno, facecolor='tomato', alpha=0.075)
# plt.fill_between(x_Bugey,0.1,y_Bugey, facecolor='black', alpha=0.075)
plt.fill_between(x_Borexino,0.1,y_Borexino, facecolor='blue', alpha=0.075)
plt.fill_between(x_PionDecay_tot,0.1,y_PionDecay_tot, facecolor='gold', alpha=0.075)
# plt.fill_between(x_PS191,0.1,y_PS191, facecolor='magenta', alpha=0.075)
# plt.fill_between(x23,0.1,y23, facecolor='gold', alpha=0.075)
plt.fill_between(x_T2K,0.1,y_T2K, facecolor='purple', alpha=0.075)
# plt.fill_between(x_NA62,0.1,y_NA62, facecolor='teal', alpha=0.075)
# plt.fill_between(x_JINR,0.1,y_JINR, facecolor='sienna', alpha=0.075)
# plt.fill_between(x_CHARM,0.1,y_CHARM, facecolor='darkslategray', alpha=0.075)
plt.fill_between(x_BEBC,0.1,y_BEBC, facecolor='yellowgreen', alpha=0.075)
plt.fill_between(x_DELPHI,0.1,y_DELPHI, facecolor='teal', alpha=0.075)
plt.fill_between(x_Belle,0.1,y_Belle, facecolor='darkgreen', alpha=0.075)
# plt.fill_between(x_ATLAS_LNC,0.1,y_ATLAS_LNC, facecolor='mediumblue', alpha=0.075)
# plt.fill_between(x_CMS_TriLepton,0.1,y_CMS_TriLepton, facecolor='red', alpha=0.075)
plt.fill_between(x_EWPD,0.1,y_EWPD, facecolor='darkgoldenrod', alpha=0.075)

#plt.fill_between(x_CMB,y_CMB,z_CMB, facecolor='black', alpha=0.02,lw=0)
plt.fill_between(x_CMB_BAO_H_2,0.1,y_CMB_BAO_H_2, facecolor='black', alpha=0.02,lw=0)
#plt.fill_between(x_BBN_2,0.1,y_BBN_2, facecolor='black', alpha=0.02,lw=0)
#plt.fill_between(x_CMB_Linear,y_CMB_Linear,z_CMB_Linear,facecolor='black', alpha=0.02,lw=0)
#plt.fill_between(x_Xray_2_shift,-20,y_Xray_2_shift,color='orangered', alpha=0.075,lw=0)
# plt.fill_between(x_Kopp_SN,y_Kopp_SN,facecolor='darkslateblue', alpha=0.005,lw=0)
# plt.fill_between(x_Kopp_SN_2,y_Kopp_SN_2,facecolor='darkslategrey', alpha=0.005,lw=0)
# plt.fill_between(x_Zhou_SN,y_Zhou_SN_2,y_Zhou_SN,facecolor='darkslategrey', alpha=0.005,lw=0)
# plt.fill_between(x_Zhou_SN_shift,y_Zhou_SN_shift,z_Zhou_SN_2,facecolor='darkslateblue', alpha=0.005,lw=0)
# plt.fill_between(x_Zhou_SN_shift2,y_Zhou_SN_shift2,z_Zhou_SN_3,facecolor='darkslateblue', alpha=0.005,lw=0)
# plt.fill_between(x_Zhou_SN_2,y_Zhou_SN_2,z_Zhou_SN_2_1,facecolor='darkslateblue', alpha=0.005,lw=0)
# plt.fill_between(x_Raffelt_SN,y_Raffelt_SN,facecolor='darkslateblue', alpha=0.005,lw=0)
plt.fill_between(x_ShiSigl_SN,-3.6,y_ShiSigl_SN,facecolor='darkslateblue', alpha=0.005,lw=0)
# plt.fill_between(x_Valle_SN,0.1,y_Valle_SN,facecolor='darkslateblue', alpha=0.005,lw=0)

### Labels ###

# plt.text(-1.3-9, -3.4, r'$\mathrm{Daya\,Bay}$',fontsize=15,rotation=0,color='mediumorchid')
#plt.text(-0.06-9, -0.4, r'$\mathrm{PROSPECT}$',fontsize=13,rotation=0,color='darkturquoise')
#plt.text(0.08-9, -2.9, r'$\mathrm{NEOS}$',fontsize=15,rotation=0,color='forestgreen')
#plt.text(0.71-9, -0.80, r'$\mathrm{SK+IC+DC}$',fontsize=12,rotation=0,color='darkorange')
#plt.text(1.5-9, -2.55, r'$^{3}\mathrm{H}$',fontsize=16,rotation=0,color='lime')
#plt.text(2.25-9, -1.65, r'$^{187}\mathrm{Re}$',fontsize=16,rotation=0,color='violet')
plt.text(3.75-9, -2.10, r'$^{45}\mathrm{Ca}$',fontsize=16,rotation=0,color='plum')
plt.text(4.8-9, -3.8, r'$^{35}\mathrm{S}$',fontsize=16,rotation=0,color='y')
plt.text(4.15-9, -3.95, r'$^{63}\mathrm{Ni}$',fontsize=16,rotation=0,color='darkmagenta')
plt.text(4.55-9, -1.35, r'$^{144}\mathrm{Ce}-^{144}\mathrm{Pr}$',fontsize=16,rotation=0,color='crimson')
plt.text(4-9, -1.55, r'$^{64}\mathrm{Cu}$',fontsize=16,rotation=0,color='darkcyan')
plt.text(5.5-9, -3.1, r'$^{20}\mathrm{F}$',fontsize=16,rotation=0,color='green')
plt.text(5.5-9, -4.3, r'$^{7}\mathrm{Be}$',fontsize=16,rotation=0,color='rosybrown')
plt.text(6.2-9, -2, r'$\mathrm{Rovno}$',fontsize=14,rotation=0,color='tomato')
plt.text(6.4-9, -3.9, r'$\mathrm{Bugey}$',fontsize=14,rotation=0,color='turquoise')
plt.text(5.55-9, -5, r'$\mathrm{Borexino}$',fontsize=16,rotation=0,color='blue')
plt.text(6.85-9, -7, r'$\mathrm{PIENU}$',fontsize=16,rotation=0,color='gold')
plt.text(8.75-9, -8.2, r'$\mathrm{NA62}$',fontsize=16,rotation=0,color='teal')
plt.text(7.5-9, -5.1, r'$\mathrm{Super-K}$',fontsize=16,rotation=290,color='mediumseagreen')
plt.text(7.16-9, -1.5, r'$\mathrm{IHEP-JINR}$',fontsize=16,rotation=287.5,color='sienna')
plt.text(9.725-9, -2.55, r'$\mathrm{Belle}$',fontsize=16,rotation=0,color='darkgreen')
plt.text(9.16-9, -1, r'$\mathrm{NA3}$',fontsize=14,rotation=0,color='cornsilk')
plt.text(9.4-9, -6.5, r'$\mathrm{CHARM}$',fontsize=16,rotation=0,color='darkslategray')
plt.text(9.4-9, -7.5, r'$\mathrm{BEBC}$',fontsize=16,rotation=0,color='yellowgreen')
plt.text(7.2-9, -4, r'$\mathrm{PS191}$',fontsize=16,rotation=290,color='magenta')
plt.text(8.8-9, -9, r'$\mathrm{T2K} $',fontsize=16,rotation=0,color='purple')
plt.text(8.52-9, -5, r'$\mathrm{BESIII}$',fontsize=16,rotation=0,color='c')
plt.text(7.9-9, -2.4, r'$\mathrm{LNV\,Decays } $',fontsize=14,rotation=90,color='mediumaquamarine')
plt.text(10-9, -3.6, r'$\mathrm{L3}$',fontsize=16,rotation=0,color='salmon')
plt.text(9.91-9, -4.2, r'$\mathrm{DELPHI}$',fontsize=14,rotation=0,color='teal')
#plt.text(11.7-9, -1.5, r'$\mathrm{CMS}$',fontsize=16,rotation=0,color='red')
#plt.text(10.7-9, -4.9, r'$\mathrm{ATLAS}$',fontsize=16,rotation=0,color='mediumblue')
#plt.text(11.1-9, -3.5, r'$\mathrm{Higgs}$',fontsize=16,rotation=0,color='sandybrown')
#plt.text(12.1-9, -3.1, r'$\mathrm{EWPD}$',fontsize=16,rotation=0,color='darkgoldenrod')
# plt.text(-1.2-9, -4.8, r'$m^{\mathrm{sterile}}_{\mathrm{eff}},\,\Delta N_{\mathrm{eff}}$',fontsize=16,rotation=0,color='grey')
#plt.text(0.2-9, -5.4, r'$\mathrm{CMB}$',fontsize=16,rotation=0,color='dimgrey')
plt.text(4.25-9, -15.5, r'$\mathrm{X-ray} (N/A)$',fontsize=16,rotation=0,color='orangered')
# plt.text(-0.6-9, -5.8, r'$\mathrm{DM \, abundance}$',fontsize=16,rotation=0,color='grey')
plt.text(5.5-9, -11.5, r'$\mathrm{CMB}+\mathrm{BAO}+H_0$ (N/A)',fontsize=16,rotation=0,color='grey')
plt.text(8.75-9, -9.8, r'$\mathrm{BBN}$',fontsize=16,rotation=0,color='grey')
plt.text(10.4-9, -11.5, r'$\mathrm{Seesaw}$',fontsize=16,rotation=0,color='black')
plt.text(1.4-9, -9.2, r'$\mathrm{Supernovae}$',fontsize=16,alpha=0.7,rotation=0,color='darkslateblue')

axes.set_xticks([-7,-6,-5,-4,-3,-2,-1,0,1])
axes.xaxis.set_ticklabels([r'',r'$10^{-6}$',r'',r'',r'$10^{-3}$',r'',r'',r'$1$',r''],fontsize =26)
axes.set_yticks([-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0])
axes.yaxis.set_ticklabels([r'$10^{-20}$',r'',r'$10^{-18}$',r'',r'$10^{-16}$',r'',r'$10^{-14}$',r'',r'$10^{-12}$',r'',r'$10^{-10}$',r'',r'$10^{-8}$',r'',r'$10^{-6}$',r'',r'$10^{-4}$',r'',r'$10^{-2}$',r'',r'$1$'],fontsize =26)
axes.tick_params(axis='x', which='major', pad=7.5)

axes.set_ylabel(r'$|U_{eN}|^2$',fontsize=30,rotation=90)
axes.set_xlabel(r'$m_N \, [\mathrm{GeV}]$',fontsize=30,rotation=0)

axes.xaxis.set_label_coords(0.52,-0.08)
axes.yaxis.set_label_coords(-0.09,0.5)
axes.set_xlim(-6.1,1.1)
axes.set_ylim(-20.1,0.1)

### Set aspect ratio (golden ratio) ###

x0,x1 = axes.get_xlim()
y0,y1 = axes.get_ylim()
axes.set_aspect(2*(x1-x0)/(1+Sqrt(5))/(y1-y0))

fig.set_size_inches(15,15)

plt.legend(loc='lower right',fontsize=18,frameon=False)

# Add our results

# Load the saved data
ms1_dense = np.loadtxt('data/results_plot_data/ms1_dense.txt')
log_sinsq2theta_equal = np.loadtxt('data/results_plot_data/log_sinsq2theta_equal.txt')
X = np.loadtxt('data/results_plot_data/X.txt')
Y = np.loadtxt('data/results_plot_data/Y.txt')
ms2_required = np.loadtxt('data/results_plot_data/ms2_required.txt')

# ms1=ms2 constraint
# Need to convert from sin^2(2 theta) to |UeN|^2 and from GeV to MeV
plt.plot(np.log10(ms1_dense)-3, log_sinsq2theta_equal-np.log10(4), c='b')
plt.annotate(r'$m_{s1}=m_{s2}$', (-2, -18.2), fontsize=15, c='b', rotation=-30)

# fs constraints
fs_limit = 9.7  # keV: from Nadler et. al. 2021
contours = plt.contour(X-3, Y-np.log10(4), np.log10(ms2_required.T), colors='k', linestyles='solid', levels=[np.log10(fs_limit)])
plt.annotate('free streaming limit', (-2, -14), fontsize=15, c='k', rotation=-15)

#plt.show()
plt.savefig("./plots/UeNsq_constraints.pdf",bbox_inches='tight')
plt.savefig("./plots/UeNsq_constraints.png",bbox_inches='tight')