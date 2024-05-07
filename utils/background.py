import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit, leastsq

def poly(x, *ps):
    out = 0
    for i in range(len(ps)):
        out += ps[i] * x ** i
    return out

def dpoly(x, *ps):
    out = 0
    for i in range(len(ps)):
        if i == 0:
            continue
        else:
            out += i * ps[i] * x ** (i - 1)
    return out

def lowTpoly(x, *ps):
    return ps[0] * x + ps[1] * x ** 3

def dlowTpoly(x, *ps):
    return ps[0] + 3 * ps[1] * x ** 2

def lowTpieces_poly(x, *xpspss):
    x_p = xpspss[0]
    cps = xpspss[1:3]
    pps = xpspss[3:]

    fnb = 0
    dfnb = 0

    out = (x < x_p) * lowTpoly(x, *cps)
    fnb = lowTpoly(x_p, *cps)
    dfnb = dlowTpoly(x_p, *cps)

    return out + (x_p <= x) * poly(x-x_p, *np.concatenate([[fnb, dfnb], pps]))

def residuals(xpspss, x, y, ignore_x, weight = 1):
    res = y - lowTpieces_poly(x, *xpspss)
    penalize = (res < 0) * (x < ignore_x[1])
    return res * (1 + (weight - 1) * penalize)

def fit_subbg_pp(T, C, B, pp_power, ignore_T, resu_dir):
    idx = np.concatenate([np.where(T <= ignore_T[0])[0], np.where(T >= ignore_T[1])[0]])

    colors = mpl.colormaps['gnuplot'](np.linspace(0.2, 0.8, 10))

    p0 = [100, 0, 1] + [1] * (pp_power - 2)
    xpspss, _ = leastsq(residuals, x0 = p0, args=(T[idx], C[idx], ignore_T, 1000))

    bg = lowTpieces_poly(T, *xpspss)

    f, ax = plt.subplots(2, 1, figsize = (8,7), sharex = True, height_ratios=[1, 1])

    ax[0].plot(T, C, '.', color = colors[int(B)], linewidth = 3, markersize = 10, label = f'{B} T')
    ax[0].plot(T, bg, ':', color = 'k', linewidth = 3, markersize = 2, label = 'background')
    ax[0].vlines(xpspss[0], 0, np.max(C), linestyle = '--', color = 'r')
    ax[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
    ax[0].set_xlabel(r'$T$ [K]', fontsize = 30)
    ax[0].set_ylabel(r'$C$ [$\mu$J/K]', fontsize = 30)
    ax[0].set_xlim((0, 200))
    ax[0].set_ylim((0, 3000))
    ax[0].legend(fontsize = 20)
    

    ax[1].plot(T, C - bg, '.', color = colors[int(B)], linewidth = 3, markersize = 10, label = f'{B} T')
    ax[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
    ax[1].set_ylabel(r'$\Delta C$ [$\mu$J/K]', fontsize = 30)
    ax[1].set_ylim((0, 1000))
    ax[1].legend(fontsize = 20)
    f.tight_layout()
    f.savefig(os.path.join(resu_dir, f'subbg_pp_{B}T.png'))
    plt.close()

    return xpspss