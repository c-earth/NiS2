import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from lmfit import Model

def genspw(x, a, b, c):
    return a * x ** b * np.exp(-c/x)

def dgnspw(x, a, b, c):
    return genspw(x, a, b-2, c) * (b * x + c)

def gensgr(x, a, b, c, xc):
    return a * np.abs(x - xc).astype(np.float64) ** b + c

def dgnsgr(x, a, b, c, xc):
    return np.sign(x - xc) * a * b * np.abs(x - xc).astype(np.float64) ** (b - 1)

def genfnc(x,
           a3l, b3l, c3l,
           a3r, b3r, c3r,
           xc2,
           a2l, b2l, c2l,
           a2c, b2c, c2c,
           a2r, b2r, c2r,
           xc1,
           a1l, b1l, c1l,
           a1r, b1r, c1r):

    out = np.zeros(x.shape)
    out[x < xc2] = genspw(x[x < xc2], a3l, b3l, c3l) + gensgr(x[x < xc2], a3r, b3r, c3r, xc2)
    out[np.logical_and(xc2 < x, x < xc1)] = gensgr(x[np.logical_and(xc2 < x, x < xc1)], a2l, b2l, c2l, xc2) +\
                                            genspw(x[np.logical_and(xc2 < x, x < xc1)], a2c, b2c, c2c) +\
                                            gensgr(x[np.logical_and(xc2 < x, x < xc1)], a2r, b2r, c2r, xc1)
    out[xc1 < x] = genspw(x[xc1 < x], a1r, b1r, c1r) + gensgr(x[xc1 < x], a1l, b1l, c1l, xc1)

    return out

def fit_genfnc(T, C, B, resu_dir):
    colors = mpl.colormaps['gnuplot'](np.linspace(0.2, 0.8, 10))

    peaks = []
    for i, c in enumerate(C):
        if i in [0, len(C) - 1]:
            continue
        
        if c > C[i-1] and c > C[i+1]:
            peaks.append(i)

    gmodel = Model(genfnc)

    vary3l = 1
    vary3r = 1
    varyc2 = 1
    vary2l = 1
    vary2c = 1
    vary2r = 1
    varyc1 = 1
    vary1l = 1
    vary1r = 1
    
    params = gmodel.make_params(a3l = dict(value = 0,
                                           vary  = vary3l,
                                           min   = 0,
                                           max   = 0.01),
                                b3l = dict(value = 3,
                                           vary  = vary3l,
                                           min   = 0,
                                           max   = 6),
                                c3l = dict(value = 0,
                                           vary  = vary3l,
                                           min   = -100,
                                           max   = 0),
                                a3r = dict(value = 1,
                                           vary  = vary3r,
                                           min   = 0,
                                           max   = 1000),
                                b3r = dict(value = -2,
                                           vary  = vary3r,
                                           min   = -10,
                                           max   = 0),
                                c3r = dict(value = 1,
                                           vary  = vary3r,
                                           min   = -1000,
                                           max   = 100),
                                xc2 = dict(value = 29.5,
                                           vary  = varyc2,
                                           min   = T[peaks[0]-1],
                                           max   = T[peaks[0]+1]),
                                a2l = dict(value = 1,
                                           vary  = vary2l,
                                           min   = 0,
                                           max   = 1000),
                                b2l = dict(value = -2,
                                           vary  = vary2l,
                                           min   = -10,
                                           max   = 0),
                                c2l = dict(value = 1,
                                           vary  = vary2l,
                                           min   = -1000,
                                           max   = 1000),
                                a2c = dict(value = 0,
                                           vary  = vary2c,
                                           min   = 0,
                                           max   = 0.01),
                                b2c = dict(value = 3,
                                           vary  = vary2c,
                                           min   = 0,
                                           max   = 10),
                                c2c = dict(value = 1,
                                           vary  = vary2c,
                                           min   = -100,
                                           max   = 1000),
                                a2r = dict(value = 1,
                                           vary  = vary2r,
                                           min   = 0,
                                           max   = 1000),
                                b2r = dict(value = -2,
                                           vary  = vary2r,
                                           min   = -10,
                                           max   = 0),
                                c2r = dict(value = 1,
                                           vary  = vary2r,
                                           min   = -1000,
                                           max   = 1000),
                                xc1 = dict(value = 37.7,
                                           vary  = varyc1,
                                           min   = T[peaks[1]-1],
                                           max   = T[peaks[1]+1]),
                                a1l = dict(value = 1,
                                           vary  = vary1l,
                                           min   = 0,
                                           max   = 1000),
                                b1l = dict(value = -2,
                                           vary  = vary1l,
                                           min   = -10,
                                           max   = 0),
                                c1l = dict(value = 1,
                                           vary  = vary1l,
                                           min   = -1000,
                                           max   = 1000),
                                a1r = dict(value = 0,
                                           vary  = vary1r,
                                           min   = -1e7,
                                           max   = 1e7),
                                b1r = dict(value = 0,
                                           vary  = vary1r,
                                           min   = -10,
                                           max   = 10),
                                c1r = dict(value = 0,
                                           vary  = vary1r,
                                           min   = -1000,
                                           max   = 1000))
    
    result = gmodel.fit(C, params, x = T)

    f, ax = plt.subplots(2, 1, figsize = (8,7), sharex = True, height_ratios = [2, 1])
    
    ax[0].plot(T, result.best_fit, '-', color = 'k', linewidth = 3, markersize = 2, label = 'fit')
    ax[0].plot(T, C, '.', color = colors[int(B)], linewidth = 3, markersize = 10, label = f'{B} T')

    ax[0].vlines(result.best_values['xc1'], 0, np.max(C), linestyle = '--', color = 'k', linewidth = 1)
    ax[0].vlines(result.best_values['xc2'], 0, np.max(C), linestyle = '--', color = 'k', linewidth = 1)

    ax[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
    ax[0].set_ylabel(r'$\Delta C$ [$\mu$J/K]', fontsize = 30)
    ax[0].legend(fontsize = 20)
    

    ax[1].plot(T, C-result.best_fit, '.', color = colors[int(B)], linewidth = 3, markersize = 10, label = f'{B} T')
    ax[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
    ax[1].set_ylabel(r'$\Delta\Delta C$ [$\mu$J/K]', fontsize = 30)
    ax[1].set_xlabel(r'$T$ [K]', fontsize = 30)
    f.tight_layout()
    plt.show()
    # f.savefig(os.path.join(resu_dir, f'fit_sgr_{B}T.png'))
    print(result.best_values)