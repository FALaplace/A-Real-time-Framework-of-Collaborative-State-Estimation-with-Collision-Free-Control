"""
建立J2摄动下的航天器运动方程和相对运动方程
"""
import numpy as np
from numpy.linalg import norm
from numpy import cos, sin, tan
from scipy.linalg import expm
from scipy.optimize import approx_fprime

r2d = lambda x: x * 180 / np.pi
Re = 6371.0
mu = 398600.4418
J2 = 1.08264 * 10 ** (-3)

k_J2 = 3 * J2 * mu * Re ** 2 / 2
ka = 0.0


class RelativeMotionModel_J2:
    """
    The center satellite state is denoted as sc; the j-th follow satellite state is denoted as sj
    sc = [r, vx, h, the, inc]
    sj = [xj, yj, zj, vxj, vyj, vzj]
    The non-linear motion equation of satellite clusters:
    dsj = fj(sj; sc) + B @ u
    """

    def __init__(self, chief_sat_elems: list, orbital_elems: list = None):
        self.a = orbital_elems[0]
        self.e = orbital_elems[1]
        self.inc = orbital_elems[2]
        self.pe = orbital_elems[3]
        self.RAAN = orbital_elems[4]
        self.TA = orbital_elems[5]
        self.csat_state = np.array(chief_sat_elems)
        self.c0 = self.csat_state.copy()
        self.csat_state_ECI = None
        self.fsat_state = np.nan
        self.fsat_num = 0
        self.B = np.block([[np.zeros((3, 3))], [np.eye(3)]])
        self.G = np.block([np.eye(3), np.zeros((3, 3))])

    def derivation_csat(self):
        r, vx, h, the, i = self.csat_state
        dr = vx
        dvx = -mu / r ** 2 + h ** 2 / r ** 3 - k_J2 * (1 - 3 * sin(i) ** 2 * sin(the) ** 2) / r ** 4
        dh = -k_J2 * (sin(i) ** 2 * sin(2 * the)) / r ** 3
        dthe = h / r ** 2 + 2 * k_J2 * (cos(i) ** 2 * sin(the) ** 2) / (h * r ** 3)
        dinc = -k_J2 * (sin(2 * i) * sin(2 * the)) / (2 * h * r ** 3)
        return np.array([dr, dvx, dh, dthe, dinc])

    def derivation_fsat(self, u):
        if self.fsat_num == 0:
            return np.nan
        dfsat_state_list = []
        for j in range(self.fsat_num):
            dfsat_state_j = self.ds_fsat(self.fsat_state[j, :], self.csat_state) + self.B @ u[j, :]
            dfsat_state_list.append(dfsat_state_j)
        return np.array(dfsat_state_list)

    def upgrade_state(self, u, h):
        # improved Euler method
        K1 = self.derivation_csat(), self.derivation_fsat(u)
        # upgrade the state
        self.csat_state = self.csat_state + h * K1[0]
        self.fsat_state = self.fsat_state + h * K1[1]
        K2 = self.derivation_csat(), self.derivation_fsat(u)
        self.csat_state = self.csat_state + h * (K2[0] - K1[0]) / 2
        self.fsat_state = self.fsat_state + h * (K2[1] - K1[1]) / 2

    def add_follow_sat(self, fsat_state):
        self.fsat_num += 1
        if self.fsat_state is np.nan:
            self.fsat_state = np.array([fsat_state])
        else:
            self.fsat_state = np.vstack((self.fsat_state, fsat_state))

    @staticmethod
    def ds_fsat(s, args: np.ndarray):
        r, vx, h, the, i = args
        wx = -k_J2 * (sin(2 * i) * sin(the)) / (h * r ** 3)
        wz = h / r ** 2
        ax = (-k_J2 * (sin(2 * i) * cos(the)) / r ** 5 + 3 * vx * k_J2 * (sin(2 * i) * sin(the)) / (r ** 4 * h) -
              8 * k_J2 ** 2 * sin(i) ** 3 * cos(i) * sin(the) ** 2 * cos(the) / (r ** 6 * h ** 2))
        az = -2 * h * vx / r ** 3 - k_J2 * (sin(i) ** 2 * sin(2 * the)) / r ** 5
        eta = mu / r ** 3 + k_J2 / r ** 5 - 5 * k_J2 * sin(i) ** 2 * sin(the) ** 2 / r ** 5
        kexi = 2 * k_J2 * (sin(i) * sin(the)) / r ** 4
        x, y, z, vx, vy, vz = s
        rf = np.sqrt((r + x) ** 2 + y ** 2 + z ** 2)
        rfz = (r + x) * sin(i) * sin(the) + y * sin(i) * cos(the) + z * cos(i)
        etaf = mu / rf ** 3 + k_J2 / rf ** 5 - 5 * k_J2 * rfz ** 2 / rf ** 7
        kexif = 2 * k_J2 * rfz / rf ** 5
        dx, dy, dz = vx, vy, vz
        dvx = (2 * vy * wz - x * (etaf - wz ** 2) + y * az - z * wx * wz -
               (kexif - kexi) * sin(i) * sin(the) - r * (etaf - eta))
        dvy = (-2 * vx * wz + 2 * vz * wx - x * az - y * (etaf - wz ** 2 - wx ** 2) + z * ax -
               (kexif - kexi) * sin(i) * cos(the))
        dvz = -2 * vy * wx - x * wx * wz - y * ax - z * (etaf - wx ** 2) - (kexif - kexi) * cos(i)
        dx = np.array([dx, dy, dz, dvx, dvy, dvz])
        return dx

    def Jacboi_matrix(self, x, args=None):
        if args is None:
            args = self.csat_state
        jac = approx_fprime(x, self.ds_fsat, np.array([1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9]), args)
        return jac

    def clear(self):
        self.csat_state = self.c0
        self.fsat_state = np.nan
        self.fsat_num = 0

    def local2ECI(self):
        RAAN = self.RAAN
        inc = self.csat_state[4]
        the = self.csat_state[3]
        C1 = np.array([[cos(RAAN), sin(RAAN), 0.0],
                       [-sin(RAAN), cos(RAAN), 0.0],
                       [0.0, 0.0, 1.0]])
        C2 = np.array([[1, 0, 0],
                       [0, cos(inc), sin(inc)],
                       [0.0, -sin(inc), cos(inc)]])
        C3 = np.array([[cos(the), sin(the), 0.0],
                       [-sin(the), cos(the), 0.0],
                       [0.0, 0.0, 1.0]])
        C = C3 @ C2 @ C1
        C_inv = np.linalg.inv(C)
        csat_r = C_inv @ np.array([self.csat_state[0], 0.0, 0.0])
        fsat_r = np.array([C_inv @ self.fsat_state[j, :3] for j in range(self.fsat_num)]) + csat_r

        return csat_r, fsat_r
