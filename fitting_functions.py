from scipy.special import exprel
import scipy.integrate as integrate

def Debye(x, xd):
    eta = xd/x
    return 3 * integrate.quad(lambda y: y**2 / (exprel(y) * exprel(-y)), 0, eta)[0] / eta**3

def Einstein(x, xe):
    eta = xe/x
    return 1 / (exprel(eta) * exprel(-eta))

def Electron(x, gamma):
    return gamma * x

def phonon_electron(x, Ad, xd, Ae, xe, gamma):
    return Ad * Debye(x, xd) + Ae * Einstein(x, xe) + Electron(x, gamma)

def residuals(ps, x, y, dy, func, cutoff = 0, penalty = 1, weight = 1):
    res = (y - func(x, *ps))/dy
    penalize = (res > 0) * (x < cutoff)
    return weight * res * (1 + (penalty - 1) * penalize)