import os
import glob
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from utils.background import fit_subbg_pp, lowTpieces_poly
from utils.magnetic_peak import fit_magnetic
from utils.generalize_fit import fit_genfnc

filedir = './extracted_data/'
resudir = './results/'
fields = ['Field (Oersted)', 'Sample Temp (Kelvin)', 'Samp HC (µJ/K)', 'Samp HC Err (µJ/K)']

colors = mpl.colormaps['gnuplot'](np.linspace(0.2, 0.8, 10))

# fig, ax = plt.subplots(1, 1, figsize = (8,7))

# for filepath in glob.glob(os.path.join(filedir, 'NiS2_upto*')):
#     data = pd.read_pickle(filepath)
#     B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
#     T = data[fields[1]].to_numpy()
#     C = data[fields[2]].to_numpy()
#     idx = np.argsort(T)
#     T = T[idx]
#     C = C[idx]
#     ax.plot(T, C, '.', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')

# ax.set_xlabel(r'$T$ [K]', fontsize = 30)
# ax.set_ylabel(r'$C$ [$\mu$J/K]', fontsize = 30)
# ax.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# plt.legend(fontsize = 20)
# fig.tight_layout()
# plt.savefig(os.path.join(resudir, 'TvsC_300K_lin_lin.png'))
# plt.close()

# fig, ax = plt.subplots(1, 1, figsize = (8,7))

# for filepath in glob.glob(os.path.join(filedir, 'NiS2_upto*')):
#     data = pd.read_pickle(filepath)
#     B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
#     T = data[fields[1]].to_numpy()
#     C = data[fields[2]].to_numpy()
#     idx = np.argsort(T)
#     T = T[idx]
#     C = C[idx]
#     ax.plot(np.log(T), np.log(C), '.', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')

# ax.set_xlabel(r'$\log(T)$ []', fontsize = 30)
# ax.set_ylabel(r'$\log(C)$ []', fontsize = 30)
# ax.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# plt.legend(fontsize = 20)
# fig.tight_layout()
# plt.savefig(os.path.join(resudir, 'TvsC_300K_log_log.png'))
# plt.close()

# fig, ax = plt.subplots(1, 1, figsize = (8,7))

# for filepath in glob.glob(os.path.join(filedir, 'NiS2_@*'))[::-1]:
#     data = pd.read_pickle(filepath)
#     B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
#     T = data[fields[1]].to_numpy()
#     C = data[fields[2]].to_numpy()
#     idx = np.argsort(T)
#     T = T[idx]
#     C = C[idx]
#     ax.plot(T, C, '.', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')

# ax.set_xlabel(r'$T$ [K]', fontsize = 30)
# ax.set_ylabel(r'$C$ [$\mu$J/K]', fontsize = 30)
# ax.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# plt.legend(fontsize = 20)
# fig.tight_layout()
# plt.savefig(os.path.join(resudir, 'TvsC_050K_lin_lin.png'))
# plt.close()

# fig, ax = plt.subplots(1, 1, figsize = (8,7))

# for filepath in glob.glob(os.path.join(filedir, 'NiS2_@*'))[::-1]:
#     data = pd.read_pickle(filepath)
#     B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
#     T = data[fields[1]].to_numpy()
#     C = data[fields[2]].to_numpy()
#     idx = np.argsort(T)
#     T = T[idx]
#     C = C[idx]
#     ax.plot(np.log(T), np.log(C), '.', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')

# ax.set_xlabel(r'$\log(T)$ []', fontsize = 30)
# ax.set_ylabel(r'$\log(C)$ []', fontsize = 30)
# ax.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# plt.legend(fontsize = 20)
# fig.tight_layout()
# plt.savefig(os.path.join(resudir, 'TvsC_050K_log_log.png'))
# plt.close()

ignore_T = [20, 45]
max_T = 200

# dataf = pd.read_pickle('./extracted_data/NiS2_upto_300K_@0T.pkl')
# datal = pd.read_pickle('./extracted_data/NiS2_@0T_20_12_2022.pkl')
# B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
# T = np.concatenate([dataf[fields[1]].to_numpy(), datal[fields[1]].to_numpy()])
# C = np.concatenate([dataf[fields[2]].to_numpy(), datal[fields[2]].to_numpy()])
# idx = np.argsort(T)
# T = T[idx]
# C = C[idx]
# C = C[T<max_T]
# T = T[T<max_T]


# pp_power = 3
# fit_subbg_pp(T, C, B, pp_power, ignore_T, resudir)

data = pd.read_pickle('./extracted_data/NiS2_upto_300K_@9T.pkl')
B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
T = data[fields[1]].to_numpy()
C = data[fields[2]].to_numpy()
idx = np.argsort(T)
T = T[idx]
C = C[idx]
C = C[T<max_T]
T = T[T<max_T]

pp_power = 3
xpspss = fit_subbg_pp(T, C, B, pp_power, ignore_T, resudir)

# fig, ax = plt.subplots(1, 1, figsize = (8,7))

# for filepath in glob.glob(os.path.join(filedir, 'NiS2_@*'))[::-1]:
#     data = pd.read_pickle(filepath)
#     B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
#     T = data[fields[1]].to_numpy()
#     C = data[fields[2]].to_numpy()
#     idx = np.argsort(T)
#     T = T[idx]
#     C = C[idx]
#     ax.plot(T, C - lowTpieces_poly(T, *xpspss), '.', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')

# ax.set_xlabel(r'$T$ [K]', fontsize = 30)
# ax.set_ylabel(r'$C$ [$\mu$J/K]', fontsize = 30)
# ax.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# plt.legend(fontsize = 20)
# fig.tight_layout()
# plt.savefig(os.path.join(resudir, 'TvsdC_050K_lin_lin.png'))
# plt.close()

# fig, ax = plt.subplots(1, 1, figsize = (8,7))

# for filepath in glob.glob(os.path.join(filedir, 'NiS2_@*'))[::-1]:
#     data = pd.read_pickle(filepath)
#     B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
#     T = data[fields[1]].to_numpy()
#     C = data[fields[2]].to_numpy()
#     idx = np.argsort(T)
#     T = T[idx]
#     C = C[idx]
#     C_sub = C - lowTpieces_poly(T, *xpspss)
#     T_sub = T[C_sub > 0]
#     C_sub = C_sub[C_sub > 0]
#     ax.plot(np.log(T_sub), np.log(C_sub), '.', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')

# ax.set_xlabel(r'$\log(T)$ []', fontsize = 30)
# ax.set_ylabel(r'$\log(C)$ []', fontsize = 30)
# ax.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# plt.legend(fontsize = 20)
# fig.tight_layout()
# plt.savefig(os.path.join(resudir, 'TvsdC_050K_log_log.png'))
# plt.close()

# fig, axs = plt.subplots(1, 2, figsize = (15,7), sharey=True)

# for filepath in glob.glob(os.path.join(filedir, 'NiS2_@*'))[::-1]:
#     data = pd.read_pickle(filepath)
#     B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
#     T = data[fields[1]].to_numpy()
#     C = data[fields[2]].to_numpy()
#     idx = np.argsort(T)
#     T = T[idx]
#     C = C[idx]
#     C_sub = C - lowTpieces_poly(T, *xpspss)
#     T_sub = T[C_sub > 0]
#     C_sub = C_sub[C_sub > 0]
#     idm = np.argmax(C_sub)
#     T_sub = np.delete(T_sub, idm, axis = 0)
#     C_sub = np.delete(C_sub, idm, axis = 0)
#     axs[0].plot(np.log(xc2[B] - T_sub[T_sub<xc2[B]]), C_sub[T_sub<xc2[B]], 'o-', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')
#     axs[1].plot(np.log(T_sub[T_sub>xc2[B]] - xc2[B]), C_sub[T_sub>xc2[B]], 'o-', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')
    

# axs[0].set_xlabel(r'$\log(\|T-T_{c2}\|)$ []', fontsize = 30)
# axs[1].set_xlabel(r'$\log(\|T-T_{c2}\|)$ []', fontsize = 30)
# axs[0].invert_xaxis()
# axs[0].set_ylabel(r'$C$ [$\mu$J/K]', fontsize = 30)
# axs[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# axs[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# plt.legend(fontsize = 20)
# fig.tight_layout()
# plt.savefig(os.path.join(resudir, 'TvsdC_050K_log_lin_M2.png'))
# plt.close()

# fig, axs = plt.subplots(1, 2, figsize = (15,7), sharey=True)

# for filepath in glob.glob(os.path.join(filedir, 'NiS2_@*'))[::-1]:
#     data = pd.read_pickle(filepath)
#     B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
#     T = data[fields[1]].to_numpy()
#     C = data[fields[2]].to_numpy()
#     idx = np.argsort(T)
#     T = T[idx]
#     C = C[idx]
#     C_sub = C - lowTpieces_poly(T, *xpspss)
#     T_sub = T[C_sub > 0]
#     C_sub = C_sub[C_sub > 0]
#     idm = np.argmax(C_sub)
#     T_sub = np.delete(T_sub, idm, axis = 0)
#     C_sub = np.delete(C_sub, idm, axis = 0)
#     axs[0].plot(np.log(xc2[B] - T_sub[T_sub<xc2[B]]), np.log(C_sub[T_sub<xc2[B]]), 'o-', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')
#     axs[1].plot(np.log(T_sub[T_sub>xc2[B]] - xc2[B]), np.log(C_sub[T_sub>xc2[B]]), 'o-', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')
    

# axs[0].set_xlabel(r'$\log(\|T-T_{c2}\|)$ []', fontsize = 30)
# axs[1].set_xlabel(r'$\log(\|T-T_{c2}\|)$ []', fontsize = 30)
# axs[0].invert_xaxis()
# axs[0].set_ylabel(r'$\log(C)$ []', fontsize = 30)
# axs[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# axs[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# axs[0].set_ylim(4, 7)
# plt.legend(fontsize = 20)
# fig.tight_layout()
# plt.savefig(os.path.join(resudir, 'TvsdC_050K_log_log_M2.png'))
# plt.close()

# fig, axs = plt.subplots(1, 2, figsize = (15,7), sharey=True)

# for filepath in glob.glob(os.path.join(filedir, 'NiS2_@*'))[::-1]:
#     data = pd.read_pickle(filepath)
#     B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
#     T = data[fields[1]].to_numpy()
#     C = data[fields[2]].to_numpy()
#     idx = np.argsort(T)
#     T = T[idx]
#     C = C[idx]
#     C_sub = C - lowTpieces_poly(T, *xpspss)
#     T_sub = T[C_sub > 0]
#     C_sub = C_sub[C_sub > 0]
#     axs[0].plot(np.log(xc1[B] - T_sub[T_sub<xc1[B]]), C_sub[T_sub<xc1[B]], 'o-', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')
#     axs[1].plot(np.log(T_sub[T_sub>xc1[B]] - xc1[B]), C_sub[T_sub>xc1[B]], 'o-', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')
    

# axs[0].set_xlabel(r'$\log(\|T-T_{c1}\|)$ []', fontsize = 30)
# axs[1].set_xlabel(r'$\log(\|T-T_{c1}\|)$ []', fontsize = 30)
# axs[0].invert_xaxis()
# axs[0].set_ylabel(r'$C$ [$\mu$J/K]', fontsize = 30)
# axs[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# axs[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# axs[0].set_ylim(0, 250)
# plt.legend(fontsize = 20)
# fig.tight_layout()
# plt.savefig(os.path.join(resudir, 'TvsdC_050K_log_lin_M1.png'))
# plt.close()

# fig, axs = plt.subplots(1, 2, figsize = (15,7), sharey=True)

# for filepath in glob.glob(os.path.join(filedir, 'NiS2_@*'))[::-1]:
#     data = pd.read_pickle(filepath)
#     B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
#     T = data[fields[1]].to_numpy()
#     C = data[fields[2]].to_numpy()
#     idx = np.argsort(T)
#     T = T[idx]
#     C = C[idx]
#     C_sub = C - lowTpieces_poly(T, *xpspss)
#     T_sub = T[C_sub > 0]
#     C_sub = C_sub[C_sub > 0]
#     axs[0].plot(np.log(xc1[B] - T_sub[T_sub<xc1[B]]), np.log(C_sub[T_sub<xc1[B]]), 'o-', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')
#     axs[1].plot(np.log(T_sub[T_sub>xc1[B]] - xc1[B]), np.log(C_sub[T_sub>xc1[B]]), 'o-', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')
    

# axs[0].set_xlabel(r'$\log(\|T-T_{c1}\|)$ []', fontsize = 30)
# axs[1].set_xlabel(r'$\log(\|T-T_{c1}\|)$ []', fontsize = 30)
# axs[0].invert_xaxis()
# axs[0].set_ylabel(r'$\log(C)$ []', fontsize = 30)
# axs[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# axs[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# axs[0].set_ylim(3.5, 5.5)
# plt.legend(fontsize = 20)
# fig.tight_layout()
# plt.savefig(os.path.join(resudir, 'TvsdC_050K_log_log_M1.png'))
# plt.close()

# for filepath in glob.glob(os.path.join(filedir, 'NiS2_@*'))[::-1]:
#     data = pd.read_pickle(filepath)
#     B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
#     if B != 0.0:
#         continue
#     T = data[fields[1]].to_numpy()
#     C = data[fields[2]].to_numpy()
#     idx = np.argsort(T)
#     T = T[idx]
#     C = C[idx]
#     C_sub = C - lowTpieces_poly(T, *xpspss)
#     fit_magnetic(T, C_sub, B, resudir)

for filepath in glob.glob(os.path.join(filedir, 'NiS2_@*'))[::-1]:
    data = pd.read_pickle(filepath)
    B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
    if B != 0.0:
        continue
    T = data[fields[1]].to_numpy()
    C = data[fields[2]].to_numpy()
    idx = np.argsort(T)
    T = T[idx]
    C = C[idx]
    C_sub = C - lowTpieces_poly(T, *xpspss)
    fit_genfnc(T, C_sub, B, resudir)

# xc2 = {0.:29.64, 1.:29.55, 2.:29.81, 3.:29.58, 5.:29.68, 6.:29.77, 7.:29.70, 8.:29.96}
# xc1 = {0.:38.19, 1.:37.83, 2.:37.91, 3.:37.81, 5.:37.86, 6.:37.31, 7.:37.81, 8.:37.93}

# fig, axs = plt.subplots(1, 2, figsize = (15,7), sharey=True)

# for filepath in glob.glob(os.path.join(filedir, 'NiS2_@*'))[::-1]:
#     data = pd.read_pickle(filepath)
#     B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
#     T = data[fields[1]].to_numpy()
#     C = data[fields[2]].to_numpy()
#     idx = np.argsort(T)
#     T = T[idx]
#     C = C[idx]
#     C_sub = C - lowTpieces_poly(T, *xpspss)
#     T_sub = T[C_sub > 0]
#     C_sub = C_sub[C_sub > 0]
#     idm = np.argmax(C_sub)
#     T_sub = np.delete(T_sub, idm, axis = 0)
#     C_sub = np.delete(C_sub, idm, axis = 0)
#     axs[0].plot(np.log(xc2[B] - T_sub[T_sub<xc2[B]]), C_sub[T_sub<xc2[B]], 'o-', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')
#     axs[1].plot(np.log(T_sub[T_sub>xc2[B]] - xc2[B]), C_sub[T_sub>xc2[B]], 'o-', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')
    

# axs[0].set_xlabel(r'$\log(\|T-T_{c2}\|)$ []', fontsize = 30)
# axs[1].set_xlabel(r'$\log(\|T-T_{c2}\|)$ []', fontsize = 30)
# axs[0].invert_xaxis()
# axs[0].set_ylabel(r'$C$ [$\mu$J/K]', fontsize = 30)
# axs[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# axs[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# plt.legend(fontsize = 20)
# fig.tight_layout()
# plt.savefig(os.path.join(resudir, 'TvsdC_050K_log_lin_M2_fit.png'))
# plt.close()

# fig, axs = plt.subplots(1, 2, figsize = (15,7), sharey=True)

# for filepath in glob.glob(os.path.join(filedir, 'NiS2_@*'))[::-1]:
#     data = pd.read_pickle(filepath)
#     B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
#     T = data[fields[1]].to_numpy()
#     C = data[fields[2]].to_numpy()
#     idx = np.argsort(T)
#     T = T[idx]
#     C = C[idx]
#     C_sub = C - lowTpieces_poly(T, *xpspss)
#     T_sub = T[C_sub > 0]
#     C_sub = C_sub[C_sub > 0]
#     idm = np.argmax(C_sub)
#     T_sub = np.delete(T_sub, idm, axis = 0)
#     C_sub = np.delete(C_sub, idm, axis = 0)
#     axs[0].plot(np.log(xc2[B] - T_sub[T_sub<xc2[B]]), np.log(C_sub[T_sub<xc2[B]]), 'o-', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')
#     axs[1].plot(np.log(T_sub[T_sub>xc2[B]] - xc2[B]), np.log(C_sub[T_sub>xc2[B]]), 'o-', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')
    

# axs[0].set_xlabel(r'$\log(\|T-T_{c2}\|)$ []', fontsize = 30)
# axs[1].set_xlabel(r'$\log(\|T-T_{c2}\|)$ []', fontsize = 30)
# axs[0].invert_xaxis()
# axs[0].set_ylabel(r'$\log(C)$ []', fontsize = 30)
# axs[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# axs[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# axs[0].set_ylim(4, 7)
# plt.legend(fontsize = 20)
# fig.tight_layout()
# plt.savefig(os.path.join(resudir, 'TvsdC_050K_log_log_M2_fit.png'))
# plt.close()

# fig, axs = plt.subplots(1, 2, figsize = (15,7), sharey=True)

# for filepath in glob.glob(os.path.join(filedir, 'NiS2_@*'))[::-1]:
#     data = pd.read_pickle(filepath)
#     B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
#     T = data[fields[1]].to_numpy()
#     C = data[fields[2]].to_numpy()
#     idx = np.argsort(T)
#     T = T[idx]
#     C = C[idx]
#     C_sub = C - lowTpieces_poly(T, *xpspss)
#     T_sub = T[C_sub > 0]
#     C_sub = C_sub[C_sub > 0]
#     axs[0].plot(np.log(xc1[B] - T_sub[T_sub<xc1[B]]), C_sub[T_sub<xc1[B]], 'o-', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')
#     axs[1].plot(np.log(T_sub[T_sub>xc1[B]] - xc1[B]), C_sub[T_sub>xc1[B]], 'o-', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')
    

# axs[0].set_xlabel(r'$\log(\|T-T_{c1}\|)$ []', fontsize = 30)
# axs[1].set_xlabel(r'$\log(\|T-T_{c1}\|)$ []', fontsize = 30)
# axs[0].invert_xaxis()
# axs[0].set_ylabel(r'$C$ [$\mu$J/K]', fontsize = 30)
# axs[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# axs[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# axs[0].set_ylim(0, 250)
# plt.legend(fontsize = 20)
# fig.tight_layout()
# plt.savefig(os.path.join(resudir, 'TvsdC_050K_log_lin_M1_fit.png'))
# plt.close()

# fig, axs = plt.subplots(1, 2, figsize = (15,7), sharey=True)

# for filepath in glob.glob(os.path.join(filedir, 'NiS2_@*'))[::-1]:
#     data = pd.read_pickle(filepath)
#     B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
#     T = data[fields[1]].to_numpy()
#     C = data[fields[2]].to_numpy()
#     idx = np.argsort(T)
#     T = T[idx]
#     C = C[idx]
#     C_sub = C - lowTpieces_poly(T, *xpspss)
#     T_sub = T[C_sub > 0]
#     C_sub = C_sub[C_sub > 0]
#     axs[0].plot(np.log(xc1[B] - T_sub[T_sub<xc1[B]]), np.log(C_sub[T_sub<xc1[B]]), 'o-', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')
#     axs[1].plot(np.log(T_sub[T_sub>xc1[B]] - xc1[B]), np.log(C_sub[T_sub>xc1[B]]), 'o-', linewidth = 3, markersize=10, color = colors[int(B)], label = f'{B} T')
    

# axs[0].set_xlabel(r'$\log(\|T-T_{c1}\|)$ []', fontsize = 30)
# axs[1].set_xlabel(r'$\log(\|T-T_{c1}\|)$ []', fontsize = 30)
# axs[0].invert_xaxis()
# axs[0].set_ylabel(r'$\log(C)$ []', fontsize = 30)
# axs[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# axs[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
# axs[0].set_ylim(3.5, 5.5)
# plt.legend(fontsize = 20)
# fig.tight_layout()
# plt.savefig(os.path.join(resudir, 'TvsdC_050K_log_log_M1_fit.png'))
# plt.close()