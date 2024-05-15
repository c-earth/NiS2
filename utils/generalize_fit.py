import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from lmfit import Model

def genspw(x, a, b, c):
    return a * x ** b * np.exp(-c/x)

def gensgr(x, a, b, c, xc):
    return a * np.abs(x - xc).astype(np.float64) ** b + c

def genfnc(x,
           a3l, b3l, c3l,
           dxp3,
           a3r, b3r,
           xc2,
           a2l, b2l,
           dxq2,
           a2c, b2c, c2c,
           dxp2,
           a2r, b2r,
           xc1,
           a1l, b1l,
           dxp1,
           a1r, b1r, c1r):
    xp1 = xc1 + dxp1
    xp2 = xc1 - dxp2
    xq2 = xc2 + dxq2
    xp3 = xc2 - dxp3
    c3r = genspw(xp3, a3l, b3l, c3l) - gensgr(xp3, a3r, b3r, 0, xc2)
    c2l = genspw(xq2, a2c, b2c, c2c) - gensgr(xq2, a2l, b2l, 0, xc2)
    c2r = genspw(xp2, a2c, b2c, c2c) - gensgr(xp2, a2r, b2r, 0, xc1)
    c1l = genspw(xp1, a1r, b1r, c1r) - gensgr(xp1, a1l, b1l, 0, xc1)

    out = np.zeros(x.shape)
    out[x <= xp3] = genspw(x[x <= xp3], a3l, b3l, c3l)
    out[np.logical_and(xp3 < x, x < xc2)] = gensgr(x[np.logical_and(xp3 < x, x < xc2)], a3r, b3r, c3r, xc2)
    out[np.logical_and(xc2 < x, x < xq2)] = gensgr(x[np.logical_and(xc2 < x, x < xq2)], a2l, b2l, c2l, xc2)
    out[np.logical_and(xq2 <= x, x <= xp2)] = genspw(x[np.logical_and(xq2 <= x, x <= xp2)], a2c, b2c, c2c)
    out[np.logical_and(xp2 < x, x < xc1)] = gensgr(x[np.logical_and(xp2 < x, x < xc1)], a2r, b2r, c2r, xc1)
    out[np.logical_and(xc1 < x, x < xp1)] = gensgr(x[np.logical_and(xc1 < x, x < xp1)], a1l, b1l, c1l, xc1)
    out[xp1 <= x] = genspw(x[xp1 <= x], a1r, b1r, c1r)

    return out

def fit_genfnc(T, C, B, resu_dir):
    colors = mpl.colormaps['gnuplot'](np.linspace(0.2, 0.8, 10))
    gmodel = Model(genfnc)

    vary3l = 1
    varyp3 = 1
    vary3r = 1
    varyc2 = 1
    vary2l = 1
    varyq2 = 1
    vary2c = 1
    varyp2 = 1
    vary2r = 1
    varyc1 = 1
    vary1l = 1
    varyp1 = 1
    vary1r = 1
    
    params = gmodel.make_params(a3l = dict(value = 0,
                                           vary  = vary3l,
                                           min   = -10,
                                           max   = 10),
                                b3l = dict(value = 3,
                                           vary  = vary3l,
                                           min   = -10,
                                           max   = 10),
                                c3l = dict(value = 0,
                                           vary  = vary3l,
                                           min   = -100,
                                           max   = 100),
                               dxp3 = dict(value = 10,
                                           vary  = varyp3,
                                           min   = 0,
                                           max   = 15),
                                a3r = dict(value = 100,
                                           vary  = vary3r,
                                           min   = 0,
                                           max   = 1e6),
                                b3r = dict(value = -1,
                                           vary  = vary3r,
                                           min   = -10,
                                           max   = 10),
                                xc2 = dict(value = 29.5,
                                           vary  = varyc2,
                                           min   = 29.2,
                                           max   = 30.4),
                                a2l = dict(value = 1,
                                           vary  = vary2l,
                                           min   = 0,
                                           max   = 20),
                                b2l = dict(value = 1,
                                           vary  = vary2l,
                                           min   = -20,
                                           max   = 20),
                               dxq2 = dict(value = 1.,
                                           vary  = varyq2,
                                           min   = 1,
                                           max   = 7),
                                a2c = dict(value = 1,
                                           vary  = vary2c,
                                           min   = -20,
                                           max   = 20),
                                b2c = dict(value = 1,
                                           vary  = vary2c,
                                           min   = -20,
                                           max   = 20),
                                c2c = dict(value = 1,
                                           vary  = vary2c,
                                           min   = -100,
                                           max   = 20),
                               dxp2 = dict(value = 1.,
                                           vary  = varyp2,
                                           min   = 0,
                                           max   = 5),
                                a2r = dict(value = 1,
                                           vary  = vary2r,
                                           min   = -20,
                                           max   = 20),
                                b2r = dict(value = 1,
                                           vary  = vary2r,
                                           min   = -100,
                                           max   = 100),
                                xc1 = dict(value = 37.5,
                                           vary  = varyc1,
                                           min   = 37.3,
                                           max   = 39),
                                a1l = dict(value = 1,
                                           vary  = vary1l,
                                           min   = -20,
                                           max   = 20),
                                b1l = dict(value = 1,
                                           vary  = vary1l,
                                           min   = -20,
                                           max   = 0),
                               dxp1 = dict(value = 5,
                                           vary  = varyp1,
                                           min   = 0,
                                           max   = 10),
                                a1r = dict(value = 0,
                                           vary  = vary1r,
                                           min   = 0,
                                           max   = 100),
                                b1r = dict(value = -2,
                                           vary  = vary1r,
                                           min   = -10,
                                           max   = 10),
                                c1r = dict(value = 1,
                                           vary  = vary1r,
                                           min   = -1000,
                                           max   = 1000))
    
    result = gmodel.fit(C, params, x = T)

    f, ax = plt.subplots(2, 1, figsize = (8,7), sharex = True, height_ratios = [2, 1])
    
    ax[0].plot(T, result.best_fit, '-', color = 'k', linewidth = 3, markersize = 2, label = 'fit')
    ax[0].plot(T, C, '.', color = colors[int(B)], linewidth = 3, markersize = 10, label = f'{B} T')

    ax[0].vlines(result.best_values['xc1'], 0, np.max(C), linestyle = '--', color = 'k', linewidth = 1)
    ax[0].vlines(result.best_values['xc2'], 0, np.max(C), linestyle = '--', color = 'k', linewidth = 1)
    ax[0].vlines(result.best_values['xc1']+result.best_values['dxp1'], 0, np.max(C), linestyle = '--', color = 'r', linewidth = 1)
    ax[0].vlines(result.best_values['xc1']-result.best_values['dxp2'], 0, np.max(C), linestyle = '--', color = 'g', linewidth = 1)
    ax[0].vlines(result.best_values['xc2']+result.best_values['dxq2'], 0, np.max(C), linestyle = '--', color = 'g', linewidth = 1)
    ax[0].vlines(result.best_values['xc2']-result.best_values['dxp3'], 0, np.max(C), linestyle = '--', color = 'b', linewidth = 1)

    ax[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
    ax[0].set_ylabel(r'$\Delta C$ [$\mu$J/K]', fontsize = 30)
    ax[0].legend(fontsize = 20)
    

    ax[1].plot(T, C-result.best_fit, '.', color = colors[int(B)], linewidth = 3, markersize = 10, label = f'{B} T')
    ax[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
    ax[1].set_ylabel(r'$\Delta\Delta C$ [$\mu$J/K]', fontsize = 30)
    ax[1].set_xlabel(r'$T$ [K]', fontsize = 30)
    f.tight_layout()
    # f.savefig(os.path.join(resu_dir, f'fit_gen_{B}T.png'))
    print(result.best_values)
    plt.show()