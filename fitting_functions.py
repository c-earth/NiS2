import numpy as np
from scipy.special import exprel
import scipy.integrate as integrate

def gaussian(x, x0, sigma):
    return np.exp(-((x - x0) / sigma)**2 / 2) / sigma / np.sqrt(2 * np.pi)

def transition(x, al, ar, bl, br, xc):
    return (x-xc < 0)*(np.abs(x-xc) ** -al + bl) + (x-xc > 0)*(np.abs(x-xc) ** -ar + br)

def uncertain_transition(x, al, ar, bl, br, xc, sigma):
    return integrate.quad(lambda y: gaussian(x, y, sigma)*transition(y, al, ar, bl, br, xc), 0, np.inf)[0] +\
           integrate.quad(lambda y: gaussian(x, y, sigma)*transition(y, al, ar, bl, br, xc), -np.inf, 0)[0]

def Debye(x, xd):
    eta = xd/x
    return 3 * integrate.quad(lambda y: y**2 / (exprel(y) * exprel(-y)), 0, eta)[0] / eta**3

def Einstein(x, xe):
    eta = xe/x
    return 1 / (exprel(eta) * exprel(-eta))

