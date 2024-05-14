import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from lmfit import Model

def lowTpoly(x, ferro, antiferro):
    return ferro * x ** (3/2) + antiferro * x ** 3

def dlowTpoly(x, ferro, antiferro):
    return 1.5 * ferro * x ** (1/2) + 3 * antiferro * x ** 2

def singularity(x, la, lb, x_c, ra, rb):
    out = np.zeros(x.shape)
    out[x < x_c] = la * np.log(x_c - x[x < x_c]) + lb
    out[x > x_c] = ra * np.log(x[x > x_c] - x_c) + rb
    return out

def magnetic(x, dx_p, ferro, antiferro, x_c, ra, rb):
    x_p = x_c - dx_p

    out = (x <= x_p) * lowTpoly(x, ferro, antiferro)

    fnp = lowTpoly(x_p, ferro, antiferro)
    dfnp = dlowTpoly(x_p, ferro, antiferro)

    la = -dfnp*np.abs(dx_p)
    lb = fnp - la*np.log(np.abs(dx_p))

    out += (x_p < x) * singularity(x, la, lb, x_c, ra, rb)
    # out[out<0] = 0
    return out

def two_magnetic(x, dx_p1, ferro1, antiferro1, x_c1, ra1, rb1, \
                    dx_p2, ferro2, antiferro2, x_c2, ra2, rb2, c, d):
    out =  c + d*x + magnetic(x, dx_p1, ferro1, antiferro1, x_c1, ra1, rb1) \
             + magnetic(x, dx_p2, ferro2, antiferro2, x_c2, ra2, rb2) #* (x > x_c1)
    # print(x)
    # print(dx_p1, ferro1, antiferro1, x_c1, ra1, rb1, \
    #                 dx_p2, ferro2, antiferro2, x_c2, ra2, rb2, c, d)
    # print(magnetic(x, dx_p1, ferro1, antiferro1, x_c1, ra1, rb1))
    # print(magnetic(x, dx_p2, ferro2, antiferro2, x_c2, ra2, rb2))
    return out

def fit_magnetic(T, C, B, resu_dir):

    colors = mpl.colormaps['gnuplot'](np.linspace(0.2, 0.8, 10))

    mmodel = Model(two_magnetic)

    params = mmodel.make_params(dx_p1       = dict(value=5,     vary=1, min=    0, max=  20), 
                                ferro1      = dict(value=0,     vary=1, min=    0, max= 0.1), 
                                antiferro1  = dict(value=0,     vary=1, min=    0, max= 0.1), 
                                x_c1        = dict(value=29,    vary=1, min=   27, max=  30), 
                                ra1         = dict(value=0,     vary=1, min=-1000, max=   0), 
                                rb1         = dict(value=0,     vary=1, min=-1000, max=1000),
                                dx_p2       = dict(value=15,    vary=1, min=    0, max=  10),  
                                ferro2      = dict(value=0,     vary=1, min=    0, max= 0.1), 
                                antiferro2  = dict(value=0,     vary=1, min=    0, max= 0.1), 
                                x_c2        = dict(value=38,    vary=1, min=   35, max=  40), 
                                ra2         = dict(value=0,     vary=1, min=-1000, max=   0), 
                                rb2         = dict(value=0,     vary=1, min=-1000, max=1000),
                                c           = dict(value=0,     vary=1, min= -100, max= 100),
                                d           = dict(value=0,     vary=1, min= -100, max= 100))
    
    result = mmodel.fit(C, params, x=T)
    ps = result.best_values

    f, ax = plt.subplots(2, 1, figsize = (8,7), sharex = True, height_ratios=[2, 1])

    
    ax[0].plot(T, result.best_fit, '-', color = 'k', linewidth = 3, markersize = 2, label = 'fit')
    ax[0].plot(T, C, '.', color = colors[int(B)], linewidth = 3, markersize = 10, label = f'{B} T')
    ax[0].plot(T, ps['c']+ps['d']*T, ':', color = 'k', linewidth = 3, markersize = 2, label = 'fit')
    ax[0].plot(T, magnetic(T, *[ps[key] for key in ['dx_p1', 'ferro1', 'antiferro1', 'x_c1', 'ra1', 'rb1']]), ':', color = 'k', linewidth = 3, markersize = 2, label = 'fit')
    ax[0].plot(T, magnetic(T, *[ps[key] for key in ['dx_p2', 'ferro2', 'antiferro2', 'x_c2', 'ra2', 'rb2']]), ':', color = 'k', linewidth = 3, markersize = 2, label = 'fit')
    ax[0].vlines(result.best_values['x_c1'] - result.best_values['dx_p1'], 0, np.max(C), linestyle = '--', color = 'r', linewidth = 1)
    ax[0].vlines(result.best_values['x_c1'], 0, np.max(C), linestyle = '--', color = 'b', linewidth = 1)
    ax[0].vlines(result.best_values['x_c2'] - result.best_values['dx_p2'], 0, np.max(C), linestyle = '-', color = 'r', linewidth = 1)
    ax[0].vlines(result.best_values['x_c2'], 0, np.max(C), linestyle = '-', color = 'b', linewidth = 1)
    ax[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
    ax[0].set_ylabel(r'$\Delta C$ [$\mu$J/K]', fontsize = 30)
    ax[0].legend(fontsize = 20)
    

    ax[1].plot(T, C-result.best_fit, '.', color = colors[int(B)], linewidth = 3, markersize = 10, label = f'{B} T')
    ax[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
    ax[1].set_ylabel(r'$\Delta\Delta C$ [$\mu$J/K]', fontsize = 30)
    ax[1].set_xlabel(r'$T$ [K]', fontsize = 30)
    f.tight_layout()
    f.savefig(os.path.join(resu_dir, f'fit_{B}T.png'))
    print(result.best_values)
    plt.show()
    