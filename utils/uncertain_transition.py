import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def transition(x, al, ar, bl, br, xc):
    return (x-xc < 0)*(np.abs(x-xc) ** al + bl) + (x-xc > 0)*(np.abs(x-xc) ** ar + br)

def unctransition(y, al, ar, bl, br, xc, d):
    return np.array([integrate.quad(lambda x: np.exp(-(x-yi)**2/d)*transition(x, al, ar, bl, br, xc), 0, np.inf)[0] +\
                     integrate.quad(lambda x: np.exp(-(x-yi)**2/d)*transition(x, al, ar, bl, br, xc),-np.inf, 0)[0] for yi in y])


x = np.arange(-10, 10, 0.1)
plt.figure()
plt.plot(x, unctransition(x, -0.8, -0.2, 0, 0, 0, 1))
plt.show()