"""
UeNsq_future.py - 31/03/2023

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

df_current_LNC = pd.read_csv("data/heavy_sterile_constraints/future_Ue/current_LNC_data.csv",header=None, sep=",", names = ["X", "Y"])
df_current_LNV = pd.read_csv("data/heavy_sterile_constraints/future_Ue/current_LNV_data.csv",header=None, sep=",", names = ["X","Y"])

df_TRISTAN = pd.read_csv("data/heavy_sterile_constraints/future_Ue/TRISTAN_data.csv",header=None, sep=",", names = ["X", "Y"])
df_KATRIN = pd.read_csv("data/heavy_sterile_constraints/future_Ue/KATRIN_data.csv",header=None, sep=",", names = ["X", "Y"])
df_HUNTER_1 = pd.read_csv("data/heavy_sterile_constraints/future_Ue/HUNTER_1_data.csv",header=None, sep=",", names = ["X", "Y"])
df_HUNTER_2 = pd.read_csv("data/heavy_sterile_constraints/future_Ue/HUNTER_2_data.csv",header=None, sep=",", names = ["X", "Y"])
df_HUNTER_3 = pd.read_csv("data/heavy_sterile_constraints/future_Ue/HUNTER_3_data.csv",header=None, sep=",", names = ["X", "Y"])
df_BeEST_1 = pd.read_csv("data/heavy_sterile_constraints/future_Ue/BeEST_1_data.csv",header=None, sep=",", names = ["X", "Y"])
df_BeEST_2 = pd.read_csv("data/heavy_sterile_constraints/future_Ue/BeEST_2_data.csv",header=None, sep=",", names = ["X", "Y"])
df_BeEST_3 = pd.read_csv("data/heavy_sterile_constraints/future_Ue/BeEST_3_data.csv",header=None, sep=",", names = ["X", "Y"])
df_PIONEER = pd.read_csv("data/heavy_sterile_constraints/future_Ue/PIONEER_data.csv",header=None, sep=",", names = ["X", "Y"])
df_SHiP = pd.read_csv("data/heavy_sterile_constraints/future_Ue/SHiP_data.csv",header=None, sep=",", names = ["X", "Y"])
df_DUNE_1 = pd.read_csv("data/heavy_sterile_constraints/future_Ue/DUNE_1_data.csv",header=None, sep=",", names = ["X", "Y"])
df_DUNE_2 = pd.read_csv("data/heavy_sterile_constraints/future_Ue/DUNE_2_data.csv",header=None, sep=",", names = ["X", "Y"])
df_DUNE_Indirect = pd.read_csv("data/heavy_sterile_constraints/future_Ue/DUNE_Indirect_data.csv",header=None, sep=",", names = ["X", "Y"])
df_FCC_ee = pd.read_csv("data/heavy_sterile_constraints/future_Ue/FCC_ee_data.csv",header=None, sep=",", names = ["X", "Y"])
df_LHCb_disp = pd.read_csv("data/heavy_sterile_constraints/future_Ue/LHCb_disp_data.csv",header=None, sep=",", names = ["X", "Y"])
df_ATLAS_disp = pd.read_csv("data/heavy_sterile_constraints/future_Ue/ATLAS_disp_data.csv",header=None, sep=",", names = ["X", "Y"])
df_CMS_disp = pd.read_csv("data/heavy_sterile_constraints/future_Ue/CMS_disp_data.csv",header=None, sep=",", names = ["X", "Y"])
df_lept_disp = pd.read_csv("data/heavy_sterile_constraints/future_Ue/lept_disp_data.csv",header=None, sep=",", names = ["X", "Y"])
df_MATHUSLA_disp = pd.read_csv("data/heavy_sterile_constraints/future_Ue/MATHUSLA_disp_data.csv",header=None, sep=",", names = ["X", "Y"])
df_FASER_disp = pd.read_csv("data/heavy_sterile_constraints/future_Ue/FASER_disp_data.csv",header=None, sep=",", names = ["X", "Y"])
df_AL3X_disp = pd.read_csv("data/heavy_sterile_constraints/future_Ue/AL3X_disp_data.csv",header=None, sep=",", names = ["X", "Y"])
df_CODEX_b_disp = pd.read_csv("data/heavy_sterile_constraints/future_Ue/CODEX_b_e_data.csv",header=None, sep=",", names = ["X", "Y"])
df_ANUBIS_disp = pd.read_csv("data/heavy_sterile_constraints/future_Ue/ANUBIS_data.csv",header=None, sep=",", names = ["X", "Y"])
df_NA62 = pd.read_csv("data/heavy_sterile_constraints/future_Ue/NA62_data.csv",header=None, sep=",", names = ["X", "Y"])
df_ILC = pd.read_csv("data/heavy_sterile_constraints/future_Ue/ILC_data.csv",header=None, sep=",", names = ["X", "Y"])
df_DUNE_osc = pd.read_csv("data/heavy_sterile_constraints/future_Ue/DUNE_osc_data.csv",header=None, sep=",", names = ["X", "Y"])
df_JUNO = pd.read_csv("data/heavy_sterile_constraints/future_Ue/JUNO_data.csv",header=None, sep=",", names = ["X", "Y"])
df_ATHENA = pd.read_csv("data/heavy_sterile_constraints/future_Ue/ATHENA_data.csv",header=None, sep=",", names = ["X", "Y"])
df_eROSITA = pd.read_csv("data/heavy_sterile_constraints/future_Ue/eROSITA_data.csv",header=None, sep=",", names = ["X", "Y"])
df_collider_14TeV = pd.read_csv("data/heavy_sterile_constraints/future_Ue/collider_14TeV_data.csv",header=None, sep=",", names = ["X", "Y"])
df_collider_100TeV = pd.read_csv("data/heavy_sterile_constraints/future_Ue/collider_100TeV_data.csv",header=None, sep=",", names = ["X", "Y"])
df_FCC_he = pd.read_csv("data/heavy_sterile_constraints/future_Ue/FCC_he_data.csv",header=None, sep=",", names = ["X", "Y"])
df_LHeC_1 = pd.read_csv("data/heavy_sterile_constraints/future_Ue/LHeC_1_data.csv",header=None, sep=",", names = ["X", "Y"])
df_LHeC_2 = pd.read_csv("data/heavy_sterile_constraints/future_Ue/LHeC_2_data.csv",header=None, sep=",", names = ["X", "Y"])
df_CLIC_1 = pd.read_csv("data/heavy_sterile_constraints/future_Ue/CLIC_1_data.csv",header=None, sep=",", names = ["X", "Y"])
df_CLIC_2 = pd.read_csv("data/heavy_sterile_constraints/future_Ue/CLIC_2_data.csv",header=None, sep=",", names = ["X", "Y"])
df_MesonDecays_LNV = pd.read_csv("data/heavy_sterile_constraints/future_Ue/MesonDecays_LNV_data.csv",header=None, sep=",", names = ["X", "Y"])
df_CMB = pd.read_csv("data/heavy_sterile_constraints/future_Ue/CMB_data.csv",header=None, sep=",", names = ["X", "Y"])
df_CMB_Linear = pd.read_csv("data/heavy_sterile_constraints/future_Ue/CMB_Linear_data.csv",header=None, sep=",", names = ["X", "Y"])
df_ShiSigl_SN = pd.read_csv("data/heavy_sterile_constraints/future_Ue/ShiSigl_SN_data.csv",header=None, sep=",", names = ["X", "Y"])
df_CMB_BAO_H = pd.read_csv("data/heavy_sterile_constraints/future_Ue/CMB_BAO_H_data.csv",header=None, sep=",", names = ["X", "Y"])
df_CMB_H_only = pd.read_csv("data/heavy_sterile_constraints/future_Ue/CMB_H_only_data.csv",header=None, sep=",", names = ["X", "Y"])
df_BBN = pd.read_csv("data/heavy_sterile_constraints/future_Ue/BBN_data.csv",header=None, sep=",", names = ["X", "Y"])

### Read to (x,y) = (m_N,|V_{eN}|^2)  ###

x_current_LNC, y_current_LNC = [], []
for i in range(len(df_current_LNC.index)):
    x_current_LNC.append(df_current_LNC.iloc[i]['X'])
    y_current_LNC.append(df_current_LNC.iloc[i]['Y'])

x_current_LNV, y_current_LNV = [], []
for i in range(len(df_current_LNV.index)):
    x_current_LNV.append(df_current_LNV.iloc[i]['X'])
    y_current_LNV.append(df_current_LNV.iloc[i]['Y'])

x_TRISTAN, y_TRISTAN = [], []
for i in range(len(df_TRISTAN.index)):
    x_TRISTAN.append(df_TRISTAN.iloc[i]['X'])
    y_TRISTAN.append(df_TRISTAN.iloc[i]['Y'])

x_KATRIN, y_KATRIN = [], []
for i in range(len(df_KATRIN.index)):
    x_KATRIN.append(df_KATRIN.iloc[i]['X'])
    y_KATRIN.append(df_KATRIN.iloc[i]['Y'])

x_HUNTER_1, y_HUNTER_1 = [], []
for i in range(len(df_HUNTER_1.index)):
    x_HUNTER_1.append(df_HUNTER_1.iloc[i]['X'])
    y_HUNTER_1.append(df_HUNTER_1.iloc[i]['Y'])

x_HUNTER_2, y_HUNTER_2 = [], []
for i in range(len(df_HUNTER_2.index)):
    x_HUNTER_2.append(df_HUNTER_2.iloc[i]['X'])
    y_HUNTER_2.append(df_HUNTER_2.iloc[i]['Y'])

x_HUNTER_3, y_HUNTER_3 = [], []
for i in range(len(df_HUNTER_3.index)):
    x_HUNTER_3.append(df_HUNTER_3.iloc[i]['X'])
    y_HUNTER_3.append(df_HUNTER_3.iloc[i]['Y'])

x_BeEST_1, y_BeEST_1 = [], []
for i in range(len(df_BeEST_1.index)):
    x_BeEST_1.append(df_BeEST_1.iloc[i]['X'])
    y_BeEST_1.append(df_BeEST_1.iloc[i]['Y'])

x_BeEST_2, y_BeEST_2 = [], []
for i in range(len(df_BeEST_2.index)):
    x_BeEST_2.append(df_BeEST_2.iloc[i]['X'])
    y_BeEST_2.append(df_BeEST_2.iloc[i]['Y'])

x_BeEST_3, y_BeEST_3 = [], []
for i in range(len(df_BeEST_3.index)):
    x_BeEST_3.append(df_BeEST_3.iloc[i]['X'])
    y_BeEST_3.append(df_BeEST_3.iloc[i]['Y'])

x_PIONEER, y_PIONEER = [], []
for i in range(len(df_PIONEER.index)):
    x_PIONEER.append(df_PIONEER.iloc[i]['X'])
    y_PIONEER.append(df_PIONEER.iloc[i]['Y'])

x_SHiP, y_SHiP = [], []
for i in range(len(df_SHiP.index)):
    x_SHiP.append(df_SHiP.iloc[i]['X'])
    y_SHiP.append(df_SHiP.iloc[i]['Y'])

x_DUNE_1, y_DUNE_1 = [], []
for i in range(len(df_DUNE_1.index)):
    x_DUNE_1.append(df_DUNE_1.iloc[i]['X'])
    y_DUNE_1.append(df_DUNE_1.iloc[i]['Y'])

x_DUNE_2, y_DUNE_2 = [], []
for i in range(len(df_DUNE_2.index)):
    x_DUNE_2.append(df_DUNE_2.iloc[i]['X'])
    y_DUNE_2.append(df_DUNE_2.iloc[i]['Y'])

x_DUNE_Indirect, y_DUNE_Indirect = [], []
for i in range(len(df_DUNE_Indirect.index)):
    x_DUNE_Indirect.append(df_DUNE_Indirect.iloc[i]['X'])
    y_DUNE_Indirect.append(df_DUNE_Indirect.iloc[i]['Y'])

x_FCC_ee, y_FCC_ee = [], []
for i in range(len(df_FCC_ee.index)):
    x_FCC_ee.append(df_FCC_ee.iloc[i]['X'])
    y_FCC_ee.append(df_FCC_ee.iloc[i]['Y'])

x_LHCb_disp, y_LHCb_disp = [], []
for i in range(len(df_LHCb_disp.index)):
    x_LHCb_disp.append(df_LHCb_disp.iloc[i]['X'])
    y_LHCb_disp.append(df_LHCb_disp.iloc[i]['Y'])

x_ATLAS_disp, y_ATLAS_disp = [], []
for i in range(len(df_ATLAS_disp.index)):
    x_ATLAS_disp.append(df_ATLAS_disp.iloc[i]['X'])
    y_ATLAS_disp.append(df_ATLAS_disp.iloc[i]['Y'])

x_CMS_disp, y_CMS_disp = [], []
for i in range(len(df_CMS_disp.index)):
    x_CMS_disp.append(df_CMS_disp.iloc[i]['X'])
    y_CMS_disp.append(df_CMS_disp.iloc[i]['Y'])

x_lept_disp, y_lept_disp = [], []
for i in range(len(df_lept_disp.index)):
    x_lept_disp.append(df_lept_disp.iloc[i]['X'])
    y_lept_disp.append(df_lept_disp.iloc[i]['Y'])

x_MATHUSLA_disp, y_MATHUSLA_disp = [], []
for i in range(len(df_MATHUSLA_disp.index)):
    x_MATHUSLA_disp.append(df_MATHUSLA_disp.iloc[i]['X'])
    y_MATHUSLA_disp.append(df_MATHUSLA_disp.iloc[i]['Y'])

x_FASER_disp, y_FASER_disp = [], []
for i in range(len(df_FASER_disp.index)):
    x_FASER_disp.append(df_FASER_disp.iloc[i]['X'])
    y_FASER_disp.append(df_FASER_disp.iloc[i]['Y'])

x_AL3X_disp, y_AL3X_disp = [], []
for i in range(len(df_AL3X_disp.index)):
    x_AL3X_disp.append(df_AL3X_disp.iloc[i]['X'])
    y_AL3X_disp.append(df_AL3X_disp.iloc[i]['Y'])

x_CODEX_b_disp, y_CODEX_b_disp = [], []
for i in range(len(df_CODEX_b_disp.index)):
    x_CODEX_b_disp.append(df_CODEX_b_disp.iloc[i]['X'])
    y_CODEX_b_disp.append(df_CODEX_b_disp.iloc[i]['Y'])

x_ANUBIS_disp, y_ANUBIS_disp = [], []
for i in range(len(df_ANUBIS_disp.index)):
    x_ANUBIS_disp.append(df_ANUBIS_disp.iloc[i]['X'])
    y_ANUBIS_disp.append(df_ANUBIS_disp.iloc[i]['Y'])

x_NA62, y_NA62 = [], []
for i in range(len(df_NA62.index)):
    x_NA62.append(df_NA62.iloc[i]['X'])
    y_NA62.append(df_NA62.iloc[i]['Y'])

x_ILC, y_ILC = [], []
for i in range(len(df_ILC.index)):
    x_ILC.append(df_ILC.iloc[i]['X'])
    y_ILC.append(df_ILC.iloc[i]['Y'])

x_DUNE_osc, y_DUNE_osc = [], []
for i in range(len(df_DUNE_osc.index)):
    x_DUNE_osc.append(df_DUNE_osc.iloc[i]['X'])
    y_DUNE_osc.append(df_DUNE_osc.iloc[i]['Y'])

x_JUNO, y_JUNO = [], []
for i in range(len(df_JUNO.index)):
    x_JUNO.append(df_JUNO.iloc[i]['X'])
    y_JUNO.append(df_JUNO.iloc[i]['Y'])

x_ATHENA, y_ATHENA = [], []
for i in range(len(df_ATHENA.index)):
    x_ATHENA.append(df_ATHENA.iloc[i]['X'])
    y_ATHENA.append(df_ATHENA.iloc[i]['Y'])

x_eROSITA, y_eROSITA = [], []
for i in range(len(df_eROSITA.index)):
    x_eROSITA.append(df_eROSITA.iloc[i]['X'])
    y_eROSITA.append(df_eROSITA.iloc[i]['Y'])

x_collider_14TeV, y_collider_14TeV = [], []
for i in range(len(df_collider_14TeV.index)):
    x_collider_14TeV.append(df_collider_14TeV.iloc[i]['X'])
    y_collider_14TeV.append(df_collider_14TeV.iloc[i]['Y'])

x_collider_100TeV, y_collider_100TeV = [], []
for i in range(len(df_collider_100TeV.index)):
    x_collider_100TeV.append(df_collider_100TeV.iloc[i]['X'])
    y_collider_100TeV.append(df_collider_100TeV.iloc[i]['Y'])

x_FCC_he, y_FCC_he = [], []
for i in range(len(df_FCC_he.index)):
    x_FCC_he.append(df_FCC_he.iloc[i]['X'])
    y_FCC_he.append(df_FCC_he.iloc[i]['Y'])

x_LHeC_1, y_LHeC_1 = [], []
for i in range(len(df_LHeC_1.index)):
    x_LHeC_1.append(df_LHeC_1.iloc[i]['X'])
    y_LHeC_1.append(df_LHeC_1.iloc[i]['Y'])

x_CLIC_1, y_CLIC_1 = [], []
for i in range(len(df_CLIC_1.index)):
    x_CLIC_1.append(df_CLIC_1.iloc[i]['X'])
    y_CLIC_1.append(df_CLIC_1.iloc[i]['Y'])

x_CLIC_2, y_CLIC_2 = [], []
for i in range(len(df_CLIC_2.index)):
    x_CLIC_2.append(df_CLIC_2.iloc[i]['X'])
    y_CLIC_2.append(df_CLIC_2.iloc[i]['Y'])

x_LHeC_2, y_LHeC_2 = [], []
for i in range(len(df_LHeC_2.index)):
    x_LHeC_2.append(df_LHeC_2.iloc[i]['X'])
    y_LHeC_2.append(df_LHeC_2.iloc[i]['Y'])

x_MesonDecays_LNV, y_MesonDecays_LNV = [], []
for i in range(len(df_MesonDecays_LNV.index)):
    x_MesonDecays_LNV.append(df_MesonDecays_LNV.iloc[i]['X'])
    y_MesonDecays_LNV.append(df_MesonDecays_LNV.iloc[i]['Y'])

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
    x_CMB_BAO_H_2.append(df_CMB_BAO_H.iloc[-i]['X'] + 0.5)
    y_CMB_BAO_H_2.append(df_CMB_BAO_H.iloc[-i]['Y'] + 0.5)

x_CMB_H_only, y_CMB_H_only  = [], []
for i in range(len(df_CMB_H_only.index)):
    x_CMB_H_only.append(df_CMB_H_only.iloc[i]['X'])
    y_CMB_H_only.append(df_CMB_H_only.iloc[i]['Y'])

x_BBN, y_BBN  = [], []
for i in range(len(df_BBN.index)):
    x_BBN.append(df_BBN.iloc[i]['X'])
    y_BBN.append(df_BBN.iloc[i]['Y'])
x_BBN.append(6.84-9)
y_BBN.append(0.1)

x_BBN_2, y_BBN_2  = [], []
x_BBN_2.append(6.84 - 0.5-9)
y_BBN_2.append(0.1)
for i in range(1,len(df_BBN.index)+1):
    x_BBN_2.append(df_BBN.iloc[-i]['X'] - 0.5)
    y_BBN_2.append(df_BBN.iloc[-i]['Y'] - 0.5)
for i in range(len(df_BBN.index)):
    x_BBN_2.append(df_BBN.iloc[i]['X'])
    y_BBN_2.append(df_BBN.iloc[i]['Y'])
x_BBN_2.append(6.84-9)
y_BBN_2.append(0.1)

x_ShiSigl_SN, y_ShiSigl_SN = [], []
for i in range(len(df_ShiSigl_SN.index)):
    x_ShiSigl_SN.append(df_ShiSigl_SN.iloc[i]['X'])
    y_ShiSigl_SN.append(df_ShiSigl_SN.iloc[i]['Y'])

fig, axes = plt.subplots(nrows=1, ncols=1)

spacing=0.2
m = np.arange(-12,6+spacing, spacing)
age_bound = np.log10(1.1 * 10**(-7) * ((50 * 10**(3))/10**(m))**5)
bbn_bound = np.log10(5.55007 * 10**(35) * (1/10**(m))**5)
seesaw_bound = np.log10(0.05*10**(-9)/10**(m))

### Current Constraints ###

# axes.plot(m,seesaw_bound,linewidth=1,linestyle='dotted',color='black') # Seesaw line
axes.plot(x_current_LNC,y_current_LNC,linewidth=0.5,linestyle='-.',color='black')
axes.plot(x_current_LNV,y_current_LNV,linewidth=0.5,linestyle='-.',color='black') 

### Future Constraints ###

axes.plot(x_TRISTAN,y_TRISTAN,linewidth=1.5,linestyle='-.',color='mediumslateblue') # TRISTAN
axes.plot(x_KATRIN,y_KATRIN,linewidth=1.5,linestyle='-',color='tomato') # KATRIN
axes.plot(x_HUNTER_1,y_HUNTER_1,linewidth=1.5,linestyle='-',color='mediumvioletred') # HUNTER 1
axes.plot(x_HUNTER_2,y_HUNTER_2,linewidth=1.5,linestyle='--',color='mediumvioletred') # HUNTER 2
axes.plot(x_HUNTER_3,y_HUNTER_3,linewidth=1.5,linestyle='-.',color='mediumvioletred') # HUNTER 3
axes.plot(x_BeEST_1,y_BeEST_1,linewidth=1.5,linestyle='-',color='rosybrown') # BeEST Future 1
axes.plot(x_BeEST_2,y_BeEST_2,linewidth=1.5,linestyle='--',color='rosybrown') # BeEST Future 2
axes.plot(x_BeEST_3,y_BeEST_3,linewidth=1.5,linestyle='-.',color='rosybrown') # BeEST Future 3
axes.plot(x_LHCb_disp,y_LHCb_disp,linewidth=1.5,linestyle='--',color='darkgreen') # LHCb displaced
axes.plot(x_ATLAS_disp,y_ATLAS_disp,linewidth=1.5,linestyle='--',color='blue') # ATLAS displaced
axes.plot(x_CMS_disp,y_CMS_disp,linewidth=1.5,linestyle='--',color='red') # CMS displaced
# axes.plot(x_lept_disp,y_lept_disp,linewidth=1.5,linestyle='-',color='m') # Lepton displaced
axes.plot(x_MATHUSLA_disp,y_MATHUSLA_disp,linewidth=1.5,linestyle='-',color='gold') # MATHUSLA
axes.plot(x_FASER_disp,y_FASER_disp,linewidth=1.5,linestyle='--',color='c') # FASER
axes.plot(x_AL3X_disp,y_AL3X_disp,linewidth=1.5,linestyle='--',color='sienna') # AL3X
axes.plot(x_MesonDecays_LNV,y_MesonDecays_LNV,linewidth=1.5,linestyle='-',color='mediumseagreen') # Future LNV meson decays
axes.plot(x_NA62,y_NA62,linewidth=1.5,linestyle='--',color='teal') # NA62 forcast (beam dump)
axes.plot(x_ILC,y_ILC,linewidth=1.5,linestyle='-',color='fuchsia') # ILC
axes.plot(x_SHiP,y_SHiP,linewidth=1.5,linestyle='-',color='purple') # SHiP
# axes.plot(x_DUNE_1,y_DUNE_1,linewidth=1.5,linestyle='-',color='black') # DUNE ND
axes.plot(x_DUNE_2,y_DUNE_2,linewidth=1.5,linestyle='-',color='navy') # DUNE ND
axes.plot(x_DUNE_Indirect,y_DUNE_Indirect,linewidth=1.5,linestyle=':',color='navy') # DUNE Indirect
axes.plot(x_FCC_ee,y_FCC_ee,linewidth=1.5,linestyle='-',color='limegreen') # FCC-ee
# axes.plot(x_DUNE_osc,y_DUNE_osc,linewidth=1.5,linestyle='-',color='darkred') # DUNE
# axes.plot(x_FCC_he,y_FCC_he,linewidth=1.5,linestyle='-',color='lime') # FCC-he LFV
# axes.plot(x_LHeC_1,y_LHeC_1,linewidth=1.5,linestyle='-',color='indianred') # LHeC LFV
axes.plot(x_LHeC_2,y_LHeC_2,linewidth=1.5,linestyle='--',color='indianred') # LHeC Das
# axes.plot(x_CLIC_1,y_CLIC_1,linewidth=1.5,linestyle='-',color='darkslategrey') # CLIC Mitra
axes.plot(x_CLIC_2,y_CLIC_2,linewidth=1.5,linestyle='-',color='darkslategrey') # CLIC Das
axes.plot(x_PIONEER,y_PIONEER,linewidth=1.5,linestyle='-',color='olive') # PIONEER
# axes.plot(x_JUNO,y_JUNO,linewidth=1.5,linestyle='-',color='seagreen') # JUNO
# axes.plot(x_ATHENA,y_ATHENA,linewidth=1.5,linestyle='-',color='orange') # ATHENA
# axes.plot(x_eROSITA, y_eROSITA,linewidth=1.5,linestyle='-',color='olivedrab') # eROSITA
axes.plot(x_collider_14TeV,y_collider_14TeV,linewidth=1.5,linestyle='--',color='cadetblue') # Future collider 14 TeV 3 inverse ab
axes.plot(x_collider_100TeV,y_collider_100TeV,linewidth=1.5,linestyle='-',color='y') # Future collider 100 TeV 30 inverse ab

axes.plot(x_CMB,y_CMB,linewidth=1,linestyle='-.',color='dimgrey') # Evans data
# axes.plot(x_CMB_Linear,y_CMB_Linear,linewidth=1,linestyle='-.',color='dimgrey') # Linear CMB
# axes.plot(x_CMB_BAO_H,y_CMB_BAO_H,linewidth=0.5,linestyle='-',color='grey') # # Decay after BBN constraints
# axes.plot(x_CMB_H_only,y_CMB_H_only,linewidth=1.5,linestyle='--',color='red') # Decay after BBN, Hubble only
# axes.plot(x_BBN,y_BBN,linewidth=0.5,linestyle='-',color='grey') # Decay before BBN constraints
axes.plot(x_ShiSigl_SN,y_ShiSigl_SN,linewidth=1.5,linestyle=':',color='darkslateblue',alpha=0.4) # Shi and Sigl Supernova constraints

### Shading ###

plt.fill_between(x_current_LNC,0.2,y_current_LNC, facecolor='k', alpha=0.075)
plt.fill_between(x_current_LNV,0.2,y_current_LNV, facecolor='grey', alpha=0.075)
# plt.fill_between(x_CMB,y_CMB,z_CMB, facecolor='black', alpha=0.02,lw=0)
# plt.fill_between(x_CMB_BAO_H_2,0.1,y_CMB_BAO_H_2, facecolor='black', alpha=0.02,lw=0)
# plt.fill_between(x_BBN_2,0.1,y_BBN_2, facecolor='black', alpha=0.02,lw=0)
# plt.fill_between(x_CMB_Linear,y_CMB_Linear,z_CMB_Linear,facecolor='black', alpha=0.02,lw=0)
plt.fill_between(x_ShiSigl_SN,-3.6,y_ShiSigl_SN,facecolor='darkslateblue', alpha=0.01,lw=0)

### Labels ###

plt.text(10.7-9, -11.2, r'$\mathrm{FCC-ee} $',fontsize=16,rotation=0,color='limegreen')
# plt.text(11.6-9, -6.6, r'$\mathrm{FCC-he} $',fontsize=16,rotation=0,color='lime')
plt.text(9.1-9, -10.35, r'$\mathrm{SHiP} $',fontsize=16,rotation=0,color='purple')
plt.text(8.25-9, -10.4, r'$\mathrm{DUNE} (N/A)$',fontsize=16,rotation=0,color='navy')
plt.text(10.65-9, -8.5, r'$\mathrm{ATLAS} $',fontsize=15,rotation=0,color='blue')
plt.text(10.18-9, -9.8, r'$\mathrm{CMS} $',fontsize=14,rotation=0,color='red')
plt.text(9.8-9, -7.3, r'$\mathrm{LHCb} $',fontsize=15,rotation=300,color='darkgreen')
plt.text(9.35-9, -8.9, r'$\mathrm{MATHUSLA} $',fontsize=15,rotation=300,color='gold')
plt.text(8.25-9, -4.9, r'$\mathrm{FASER2} $',fontsize=15,rotation=0,color='c')
plt.text(7.4-9, -6.0, r'$\mathrm{AL3X} $',fontsize=15,rotation=0,color='sienna')
plt.text(6.7-9, -3.2, r'$\mathrm{NA62} $',fontsize=15,rotation=0,color='teal')
plt.text(4.25-9, -3.8, r'$\mathrm{KATRIN} $',fontsize=15,rotation=0,color='tomato')
plt.text(4.25-9, -7.4, r'$\mathrm{TRISTAN} $',fontsize=15,rotation=0,color='mediumslateblue')
#plt.text(11.2-9, -4.3, r'$\mathrm{ILC} $',fontsize=16,rotation=0,color='fuchsia')
#plt.text(5.5-9, -11.5, r'$\mathrm{CMB}+\mathrm{BAO}+H_0 (N/A)$',fontsize=16,rotation=0,color='grey')
#plt.text(9.4-9, -11.7, r'$\mathrm{BBN} (N/A)$',fontsize=16,rotation=0,color='grey')
#plt.text(-0.9-9, -1.7, r'$\mathrm{JUNO}$',fontsize=16,rotation=0,color='seagreen')
#plt.text(3.15-9, -14.3, r'$\mathrm{ATHENA (N/A)}$',fontsize=16,rotation=0,color='orange')
#plt.text(3.25-9, -13.4, r'$\mathrm{eROSITA} (N/A)$',fontsize=16,rotation=0,color='olivedrab')
#plt.text(10.7-9, -11.8, r'$\mathrm{Seesaw}$',fontsize=16,rotation=0,color='black')
#plt.text(0.2-9, -5.4, r'$\mathrm{CMB}$',fontsize=16,rotation=0,color='dimgrey')
#plt.text(12.15-9, -2.35, r'$\mathrm{FCC-hh}$',fontsize=14,rotation=0,color='y')
#plt.text(12.1-9, -0.3, r'$\mathrm{HL-LHC}$',fontsize=14,rotation=0,color='cadetblue')
plt.text(7.88-9, -2.35, r'$\mathrm{Future\,LNV}$',fontsize=14,rotation=90,color='mediumseagreen')
#plt.text(12.15-9, -3.7, r'$\mathrm{LHeC} $',fontsize=16,rotation=0,color='indianred')
#plt.text(12.3-9, -5, r'$\mathrm{CLIC} $',fontsize=16,rotation=0,color='darkslategrey')
plt.text(7-9, -10.65, r'$\mathrm{PIONEER} (OK)$',fontsize=16,rotation=0,color='olive')
plt.text(6.9-9, -0.5, r'$\mathrm{DUNE}\,\mathrm{Indirect}$',fontsize=16,rotation=0,color='navy')
plt.text(4.5-9, -9.8, r'$\mathrm{HUNTER} (OK)$',fontsize=16,rotation=0,color='mediumvioletred')
plt.text(6-9, -6.2, r'$\mathrm{BeEST} $',fontsize=15,rotation=0,color='rosybrown')

axes.set_xticks([-7,-6,-5,-4,-3,-2,-1,0,1, 2])
axes.xaxis.set_ticklabels([r'',r'$10^{-6}$',r'',r'',r'$10^{-3}$',r'',r'',r'$1$',r'',r''],fontsize =26)
axes.set_yticks([-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0])
axes.yaxis.set_ticklabels([r'$10^{-20}$',r'',r'$10^{-18}$',r'',r'$10^{-16}$',r'',r'$10^{-14}$',r'',r'$10^{-12}$',r'',r'$10^{-10}$',r'',r'$10^{-8}$',r'',r'$10^{-6}$',r'',r'$10^{-4}$',r'',r'$10^{-2}$',r'',r'$1$'],fontsize =26)
axes.tick_params(axis='x', which='major', pad=7.5)

axes.set_ylabel(r'$|U_{eN}|^2$',fontsize=30,rotation=90)
axes.set_xlabel(r'$m_N \, [\mathrm{GeV}]$',fontsize=30,rotation=0)

axes.xaxis.set_label_coords(0.52,-0.08)
axes.yaxis.set_label_coords(-0.09,0.5)
axes.set_xlim(-6.1,2.1)
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
plt.savefig("plots/UeNsq_future.pdf",bbox_inches='tight')
plt.savefig("plots/UeNsq_future.png",bbox_inches='tight')