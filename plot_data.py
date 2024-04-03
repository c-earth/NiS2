import os
import glob
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

filedir = './extracted_data/'
resudir = './results/'
fields = ['Field (Oersted)', 'Sample Temp (Kelvin)', 'Samp HC (µJ/K)', 'Samp HC Err (µJ/K)']

colors = mpl.colormaps['gnuplot'](np.linspace(0.2, 0.8, 10))

fig, ax = plt.subplots(1, 1, figsize = (8,7))

for filepath in glob.glob(os.path.join(filedir, 'NiS2_upto*')):
    data = pd.read_pickle(filepath)
    B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
    T = data[fields[1]].to_numpy()
    C = data[fields[2]].to_numpy()
    idx = np.argsort(T)
    T = T[idx]
    C = C[idx]
    ax.plot(T, C, '-', linewidth = 3, color = colors[int(B)], label = f'{B} T')

ax.set_xlabel(r'$T$ [K]')
ax.set_ylabel(r'$C$ [$\mu$J/K]')
plt.legend()

plt.savefig(os.path.join(resudir, 'TvsC_300K.png'))
plt.show()
plt.close()

fig=plt.figure(figsize = (8, 7))
ax=fig.add_subplot(1, 1, 1, projection='3d')

for filepath in glob.glob(os.path.join(filedir, 'NiS2_@*'))[::-1]:
    data = pd.read_pickle(filepath)
    B = np.round(data[fields[0]].to_numpy()[0]/1e4, 0)
    T = data[fields[1]].to_numpy()
    C = data[fields[2]].to_numpy()
    idx = np.argsort(T)
    T = T[idx]
    C = C[idx]
    ax.plot(T, B*np.ones(T.shape), C, '-', linewidth = 3, color = colors[int(B)], label = f'{B} T')

ax.set_xlabel(r'$T$ [K]')
ax.set_ylabel(r'$B$ [T]')
ax.set_zlabel(r'$C$ [$\mu$J/K]')
ax.view_init(10, -75)

plt.savefig(os.path.join(resudir, 'TvsC_050K.png'))
plt.show()
plt.close()