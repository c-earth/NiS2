import numpy as np
import matplotlib.pyplot as plt

# def cutoff_log(x, *ps):
#     A = ps[0]
#     x_c = ps[1]
#     x_r = x - x_c
#     x_rp = x_r[x_r > 0]
#     out = np.zeros(x.shape)
#     out[x_r > 0] = A*np.log(x_rp)
#     return out

# def flip_cutoff_log(x, *ps):
#     A = ps[0]
#     x_c = ps[1]
#     B = ps[2]
#     x_r = x - x_c
#     x_rn = x_r[x_r < 0]
#     out = np.zeros(x.shape)
#     out[x_r < 0] = A*np.log(-x_rn)+B
#     return out

# def cutoff_poly(x, *ps):
#     x_c = ps[-1]
#     out = np.zeros(x.shape)
#     for i in range(len(ps[:-1])):
#         out[x < x_c] += ps[i] * x[x < x_c] ** i
#     return out

# x = np.arange(0, 10, 0.01)

# plt.figure()
# plt.plot(x, cutoff_poly(x, 0.1, 0.2, 0.3, 0.4, 5))
# plt.plot(x, flip_cutoff_log(x, -10, 5, 10))
# plt.plot(x, cutoff_poly(x, 0.1, 0.2, 0.3, 0.4, 5) + flip_cutoff_log(x, -10, 5, 10), '.')
# plt.show()

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