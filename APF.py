"""
construct the Artificial Potential Field (based on Numpy.ndarray)
"""
from numpy import cos, sin, exp, sqrt, pi, array, dot, arccos, arcsin
from numpy import ndarray, zeros_like, arange, meshgrid, min
from numpy.linalg import norm
from scipy.optimize import fsolve


def att_field(s1, s2, rho):
    s1, s2 = array(s1), array(s2)
    d = norm(s1 - s2)
    if d <= rho:
        return d ** 2 / 2
    else:
        return rho * d - rho ** 2 / 2


def adaptive_safe_distance(sat, obs, rho, kou):
    sat, obs = array(sat), array(obs)
    d, v = sat[:3] - obs[:3], sat[3:] - obs[3:]
    cos_sigma = dot(d, -v) / (norm(d) * norm(v))
    sigma = arccos(cos_sigma)
    beta = fsolve(lambda x, a, b: a / (a + b * cos(x)) - sin(x), array([pi / 6]), args=(rho, kou * norm(v)))
    beta = beta[0]
    if cos_sigma <= 0:
        return 1.0
    if sigma <= beta:
        return rho / (rho + kou * norm(v) * cos_sigma)
    if sigma > beta:
        return sin(sigma)


def dynamic_stretch_coefficient(s_sat, y_obs, rho_obs, kp, kv):
    d = s_sat[:3] - y_obs
    v = -s_sat[3:]
    if norm(v) <= 1e-6:
        return 1.0, 1.0
    cos_sigma = dot(d, v) / (norm(d) * norm(v))
    sigma = arccos(cos_sigma)
    sin_gamma = rho_obs / norm(d)
    gamma = arcsin(sin_gamma)
    if sigma <= gamma:
        la_p = rho_obs / (rho_obs + kp / norm(d) * sin(gamma - sigma) / sin_gamma)
        la_v = rho_obs / (rho_obs + kv * norm(v) * cos_sigma * sin(gamma - sigma) / sin_gamma)
        return la_p, la_v
    elif cos_sigma <= 0:
        return 2.0, 2.0
    else:
        return 1.0, 1.0


def dynamic_rep_field(p_sat, p_obs, rho, Rd, Rv, la_p, la_v):
    d = p_sat - p_obs
    d1_ = Rd @ d
    d2_ = Rv @ d
    d1_[0] *= la_p
    d2_[0] *= la_v
    return (exp(-norm(d1_) ** 2 / (2 * rho ** 2)) + exp(-norm(d2_) ** 2 / (2 * rho ** 2))) / (2 * sqrt(2 * pi * rho))


def gradient(fun, s: ndarray, args=()):
    ds = 1e-12
    grad_s = []
    for i in range(3):
        s1 = s.copy()
        s1[i] += ds
        fxh1 = fun(s1, *args)
        s2 = s.copy()
        s2[i] -= ds
        fxh2 = fun(s2, *args)
        grad_s.append((fxh1 - fxh2) / (2 * ds))
    return array(grad_s)


def intersat_distance(t, n, s10, s20):
    c1 = s10[4] + 2 * n * s10[0]
    d1 = s10[1] - 2 * s10[3] / n
    x1 = s10[3] / n * sin(n * t) - (3 * s10[0] + 2 * s10[4] / n) * cos(n * t) + 2 * c1 / n
    y1 = 2 * s10[3] / n * cos(n * t) + 2 * (3 * s10[0] + 2 * s10[4] / n) * sin(n * t) - 3 * c1 * t + d1
    z1 = s10[5] / n * sin(n * t) + s10[2] * cos(n * t)

    c2 = s20[4] + 2 * n * s20[0]
    d2 = s20[1] - 2 * s20[3] / n
    x2 = s20[3] / n * sin(n * t) - (3 * s20[0] + 2 * s20[4] / n) * cos(n * t) + 2 * c2 / n
    y2 = 2 * s20[3] / n * cos(n * t) + 2 * (3 * s20[0] + 2 * s20[4] / n) * sin(n * t) - 3 * c2 * t + d2
    z2 = s20[5] / n * sin(n * t) + s20[2] * cos(n * t)

    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def satobs_distance(t, n, s0, po):
    c = s0[4] + 2 * n * s0[0]
    d = s0[1] - 2 * s0[3] / n
    x = s0[3] / n * sin(n * t) - (3 * s0[0] + 2 * s0[4] / n) * cos(n * t) + 2 * c / n
    y = 2 * s0[3] / n * cos(n * t) + 2 * (3 * s0[0] + 2 * s0[4] / n) * sin(n * t) - 3 * c * t + d
    z = s0[5] / n * sin(n * t) + s0[2] * cos(n * t)
    return sqrt((x - po[0]) ** 2 + (y - po[1]) ** 2 + (z - po[2]) ** 2)


def field_surface(field_fun, args=()):
    x = arange(-15, 15, 0.3)
    y = arange(-15, 15, 0.3)
    X, Y = meshgrid(x, y)
    Z = zeros_like(X)
    for i in range(x.size):
        for j in range(y.size):
            Z[i, j] = field_fun([x[i], y[j]], *args)
    return X, Y, Z


if __name__ == "_main__":
    k_att, k_rep = 4e-5, 2.5e-4
    r_att = 5e-5 / k_att
    x1, y1, z1 = field_surface(att_field, args=([0, 0], r_att))
    z1 *= k_att
