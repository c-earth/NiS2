import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import leastsq

inf = np.inf

def lowTpoly(x, *ps):
    return ps[0] * x ** (3/2) + ps[1] * x ** 3

def dlowTpoly(x, *ps):
    return 1.5 * ps[0] * x ** (1/2) + 3 * ps[1] * x ** 2

def singularity(x, *ps):
    x_c = ps[2]
    out = np.zeros(x.shape)
    out[x < x_c] = ps[0] * np.log(x_c - x[x < x_c]) + ps[1]
    out[x > x_c] = ps[3] * np.log(x[x > x_c] - x_c) + ps[4]
    return out

def magnetic(x, *xpspss):
    dx_p = xpspss[0]
    cps = xpspss[1:3]
    lps = xpspss[3:]

    x_p = lps[0] + dx_p

    fnb = 0
    dfnb = 0
    out = (x <= x_p) * lowTpoly(x, *cps)

    fnb = lowTpoly(x_p, *cps)
    dfnb = dlowTpoly(x_p, *cps)
    if lps[0] > x_p:
        A = dfnb*(x_p - lps[0])
        out += (x_p < x) * singularity(x, *np.concatenate([[A, fnb - A*np.log(lps[0] - x_p)], lps]))
    elif lps[0] < x_p:
        A = dfnb*(lps[0] - x_p)
        out += (x_p < x) * singularity(x, *np.concatenate([[A, fnb - A*np.log(x_p - lps[0])], lps]))
    return out

def residuals(xpspss, x, y, weight = 1):
    res = y - magnetic(x, *xpspss)
    penalize = (res < 0) #* (x > 30)
    return res * (1 + (weight - 1) * penalize)

def fit_magnetic(T, C, B, resu_dir):

    colors = mpl.colormaps['gnuplot'](np.linspace(0.2, 0.8, 10))
    p0     =  [  -10,   1,   1,  29,  -1,   1]

    xpspss, _ = leastsq(residuals, x0 = p0, args=(T, C, 10))

    fit_curve = magnetic(T, *xpspss)

    f, ax = plt.subplots(2, 1, figsize = (8,7), sharex = True, height_ratios=[2, 1])

    ax[0].plot(T, C, '.', color = colors[int(B)], linewidth = 3, markersize = 10, label = f'{B} T')
    ax[0].plot(T, fit_curve, ':', color = 'k', linewidth = 3, markersize = 2, label = 'fit')
    ax[0].vlines(xpspss[3] + xpspss[0], 0, np.max(C), linestyle = '--', color = 'r', linewidth = 3)
    ax[0].vlines(xpspss[3], 0, np.max(C), linestyle = '--', color = 'b', linewidth = 1)
    ax[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
    ax[0].set_ylabel(r'$\Delta C$ [$\mu$J/K]', fontsize = 30)
    ax[0].legend(fontsize = 20)
    

    ax[1].plot(T, C-fit_curve, '.', color = colors[int(B)], linewidth = 3, markersize = 10, label = f'{B} T')
    ax[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
    ax[1].set_ylabel(r'$\Delta\Delta C$ [$\mu$J/K]', fontsize = 30)
    ax[1].set_xlabel(r'$T$ [K]', fontsize = 30)
    f.tight_layout()
    f.savefig(os.path.join(resu_dir, f'fit_{B}T.png'))
    plt.show()
    return xpspss