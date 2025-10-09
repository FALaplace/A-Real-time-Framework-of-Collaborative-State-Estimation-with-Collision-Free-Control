import numpy as np
from numpy.linalg import inv, norm
from numpy import cos, sin, arctan2, arcsin, arccos
from numpy.random import uniform, normal
from scipy.linalg import expm
from Topology import SensorNetwork, MotionModel
import Visualization as vv
from MotionModel import RelativeMotionModel_J2
import APF
from copy import deepcopy
from tqdm import trange
import json

T_total = 24000
T_sam = 10
N_sample = int(T_total / T_sam)
d2r = lambda x: x * np.pi / 180
rtf2xyz = lambda x: np.array([x[0] * cos(x[1]) * cos(x[2]), x[0] * sin(x[1]) * cos(x[2]), x[0] * sin(x[2])])
xyz2rtf = lambda x, dx: [norm(x) + normal(0.0, dx[0]),
                         arctan2(x[1], x[0]) + normal(0.0, dx[1]),
                         arcsin(x[2] / norm(x)) + normal(0.0, dx[2])]

Rz = lambda theta: np.array([[cos(theta), sin(theta), 0],
                             [-sin(theta), cos(theta), 0],
                             [0, 0, 1]])
Ry = lambda phi: np.array([[cos(phi), 0, -sin(phi)],
                           [0, 1, 0],
                           [sin(phi), 0, cos(phi)]])


def dp(p1, p2):
    return norm(p1[0] - p2[0], np.inf) + abs(p1[1] - p2[1])


mu = 398600.4418
np.random.seed(40)
umax = 2 * 10 ** (-3)
rs = 0.1
obs_nums = 4
obs_pos = np.zeros((obs_nums, N_sample + 1, 3))
tf = np.pi / 2
for i in range(obs_nums):
    tfi = np.random.random() * tf
    # tfi = 0.0
    t = np.linspace(2 * np.pi / obs_nums * i, tfi + 2 * np.pi / obs_nums * i, N_sample + 1)
    x_range = uniform(low=6.0, high=11.0)
    y_range = uniform(low=6.0, high=11.0)
    z_range = uniform(low=-0.2, high=0.2)
    obs_pos[i] = np.column_stack((x_range * cos(t), y_range * sin(t), z_range * sin(0.5 * t)))
obs_rad = uniform(low=0.5, high=1.0, size=obs_nums)
# print(obs_rad)
k_att, k_rep = 5e-5, 4e-4
r_att = 5e-5 / k_att


def convert_np_array(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError


def APF_controller(si: np.ndarray, ti: np.ndarray, Di) -> np.ndarray:
    kv = 0.01
    ua = k_att * APF.gradient(APF.att_field, si[:3], args=(ti, r_att))
    ur = np.zeros_like(ua)
    for y, rho in Di:
        if norm(si[:3] - y) <= rho * 4.0:
            la_p, la_v = APF.dynamic_stretch_coefficient(si, y, rho, 5, 400)
            # la_p, la_v = 1.0, 1.0
            # if la_p * la_v < 1:
            #     print(la_p, la_v)
            # la_p, la_v = 1.0, 1.0
            dr = si[:3] - y
            vr = -si[3:]
            _, td, pd = xyz2rtf(dr, [0, 0, 0])
            _, tv, pv = xyz2rtf(vr, [0, 0, 0])
            Rd = Ry(-pd) @ Rz(td)
            Rv = Ry(-pv) @ Rz(tv)
            if norm(vr) <= 1e-15:
                Rv = np.eye(3)
            ur += k_rep * APF.gradient(APF.dynamic_rep_field, si[:3], args=(y, rho, Rd, Rv, la_p, la_v))
    uv = kv * si[3:]
    return ua + ur + uv


def distributed_EKF_sc(sn: SensorNetwork, sc: RelativeMotionModel_J2, t_sam, tar, info=None, **kwargs):
    alpha = kwargs["a"] if "a" in kwargs else 0.05 / np.sqrt(norm(sn.MM.R))
    beta = kwargs["b"] if "b" in kwargs else 600
    pd = kwargs["period"] if "period" in kwargs else None
    uFk = np.zeros((sc.fsat_num, 3))
    U_total = 0.0
    for j in range(sc.fsat_num):
        u_apf = APF_controller(sn.G.nodes[j]["S_est"], tar[j], [])
        uFk[j] = np.clip(-u_apf, -umax, umax)
    for k in trange(1, 2401):
        """real time upgrade"""
        U_total += sum([norm(uFk[m], 1) for m in range(uFk.shape[0])])
        sc.upgrade_state(uFk, t_sam)
        for ni in sn.G.nodes():
            sn.G.nodes[ni]["S_rea"] = sc.fsat_state[ni]
        sn.Time += sn.T_sam

        """first loop: local state update and environment detection"""
        for ni in sn.G.nodes():
            qiF_kk = sn.G.nodes[ni]["InfoVec_fu"]
            OiF_kk = sn.G.nodes[ni]["InfoMat_fu"]
            Oi_k_1k = inv(sn.MM.F @ inv(OiF_kk) @ sn.MM.F.T + sn.MM.Q)
            qi_k_1k = Oi_k_1k @ (sn.MM.F @ inv(OiF_kk) @ qiF_kk + sn.MM.B @ uFk[ni])
            if sn.G.nodes[ni]["NodeType"]:
                """receive signal from GNSS or ground station"""
                wi = np.random.multivariate_normal([0] * sn.y_dim, sn.MM.R)
                yi_k_1 = sn.MM.H @ sn.G.nodes[ni]["S_rea"] + wi
                sn.G.nodes[ni]["InfoVec"] = qi_k_1k + sn.MM.H.T @ inv(sn.MM.R) @ yi_k_1
                sn.G.nodes[ni]["InfoMat"] = Oi_k_1k + sn.MM.H.T @ inv(sn.MM.R) @ sn.MM.H
                sn.G.nodes[ni]["P_est"] = inv(sn.G.nodes[ni]["InfoMat"])
                sn.G.nodes[ni]["S_est"] = sn.G.nodes[ni]["P_est"] @ sn.G.nodes[ni]["InfoVec"]
                """environment detection for guider satellites"""
                for io in range(obs_nums):
                    ro_obs, to_obs, fo_obs = xyz2rtf(obs_pos[io, k], [sn.MM.dr, sn.MM.dt, sn.MM.df])
                    rho_obs = obs_rad[io] + normal(0.0, sn.MM.dr)
                    y_obs = rtf2xyz([ro_obs, to_obs, fo_obs])
                    sn.collision_avoidance_update(ni, (y_obs, rho_obs))
                for vj in sn.G.nodes():
                    if vj != ni:
                        rj_obs, tj_obs, fj_obs = xyz2rtf(sn.G.nodes[vj]["S_rea"][:3], [sn.MM.dr, sn.MM.dt, sn.MM.df])
                        yji_k_1 = rtf2xyz([rj_obs, tj_obs, fj_obs])
                        sn.collision_avoidance_update(ni, (yji_k_1, rs))
                        if vj in sn.G.successors(ni):  # out-neighbors
                            Rji_k_1 = sn.MM.calculate_R(rj_obs, tj_obs, fj_obs)
                            qji_kk = sn.G.nodes[ni]["InfoVec_nei"][vj]  # qji_{k|k}
                            Oji_kk = sn.G.nodes[ni]["InfoMat_nei"][vj]  # Oji_{k|k}
                            sji_kk = inv(Oji_kk) @ qji_kk
                            uji_apf = APF_controller(sji_kk, tar[vj], sn.G.nodes[ni]["D_bar"][vj])
                            uji = np.clip(-uji_apf, -umax, umax)
                            Oji_k_1k = inv(sn.MM.F @ inv(Oji_kk) @ sn.MM.F.T + sn.MM.Q)
                            qji_k_1k = Oji_k_1k @ (sn.MM.F @ inv(Oji_kk) @ qji_kk + sc.B @ uji)
                            qji_k_1k_1 = qji_k_1k + sn.MM.H.T @ inv(Rji_k_1) @ yji_k_1
                            Oji_k_1k_1 = Oji_k_1k + sn.MM.H.T @ inv(Rji_k_1) @ sn.MM.H
                            sn.G.nodes[ni]["InfoVec_nei"][vj] = qji_k_1k_1
                            sn.G.nodes[ni]["InfoMat_nei"][vj] = Oji_k_1k_1
                            sn.G.nodes[ni]["D_nei"][vj] = sn.obtain_Dji(ni, yji_k_1)
                            sn.G.nodes[ni]["C2"][vj] = sn.collision_detect_trigger(ni, vj)
                            if sn.G.nodes[ni]["C2"][vj]:
                                sn.G.nodes[ni]["D_bar"][vj] = deepcopy(sn.G.nodes[ni]["D_nei"][vj])
                            sn.Recorder["C2"][ni][vj].append(sn.G.nodes[ni]["C2"][vj])
                if pd is not None:
                    sn.G.nodes[ni]["C1"] = int(sn.Time // pd != (sn.Time - sn.T_sam) // pd)
                else:
                    sn.state_estimation_trigger(ni, a=alpha, b=beta)
            else:
                Oi_k_1k_1 = Oi_k_1k
                qi_k_1k_1 = qi_k_1k
                sn.G.nodes[ni]["InfoVec"] = qi_k_1k_1
                sn.G.nodes[ni]["InfoMat"] = Oi_k_1k_1
                sn.G.nodes[ni]["P_est"] = inv(Oi_k_1k_1)
                sn.G.nodes[ni]["S_est"] = sn.G.nodes[ni]["P_est"] @ qi_k_1k_1
        """Second loop: information fusion: state estimate fusion and collision avoidance set fusion"""
        for ni in sn.G.nodes():
            wii = 1 / norm(sn.G.nodes[ni]["P_est"], 2)
            PiF_k_1k_1 = wii * sn.G.nodes[ni]["P_est"].copy()
            siF_k_1k_1 = wii * sn.G.nodes[ni]["S_est"].copy()
            w_total = wii
            for vj in list(sn.G.predecessors(ni)):  # in-neighbors
                if sn.G.nodes[vj]["C1"] == 1:
                    Oij_k_1k_1 = sn.G.nodes[vj]["InfoMat_nei"][ni]
                    qij_k_1k_1 = sn.G.nodes[vj]["InfoVec_nei"][ni]
                    Pij_k_1k_1 = inv(Oij_k_1k_1)
                    sij_k_1k_1 = Pij_k_1k_1 @ qij_k_1k_1
                    wij = 1 / norm(Pij_k_1k_1, 2)
                    siF_k_1k_1 += wij * sij_k_1k_1
                    PiF_k_1k_1 += wij * Pij_k_1k_1
                    w_total += wij
            siF_k_1k_1 = siF_k_1k_1 / w_total
            PiF_k_1k_1 = PiF_k_1k_1 / w_total
            OiF_k_1k_1 = inv(PiF_k_1k_1)
            qiF_k_1k_1 = OiF_k_1k_1 @ siF_k_1k_1
            Di = sn.obstacle_fusion(ni)
            sn.G.nodes[ni]["D"] = Di
            uiF_k_1_apf = APF_controller(siF_k_1k_1, tar[ni], Di)
            uiF_k_1 = np.clip(-uiF_k_1_apf, -umax, umax)
            sn.G.nodes[ni]["InfoVec_fu"] = qiF_k_1k_1
            sn.G.nodes[ni]["InfoMat_fu"] = OiF_k_1k_1
            if sn.G.nodes[ni]["NodeType"]:
                ci = sn.G.nodes[ni]["C1"]
                qi_k_1_bar = sn.G.nodes[ni]["InfoVec_bar"]
                Oi_k_1_bar = sn.G.nodes[ni]["InfoMat_bar"]
                qi_k_1_wt = ci * sn.G.nodes[ni]["InfoVec"] + (1 - ci) * qi_k_1_bar
                Oi_k_1_wt = ci * sn.G.nodes[ni]["InfoMat"] + (1 - ci) * Oi_k_1_bar
                sn.G.nodes[ni]["InfoMat_bar"] = inv(sn.MM.F @ inv(Oi_k_1_wt) @ sn.MM.F.T + sn.MM.Q)
                sn.G.nodes[ni]["InfoVec_bar"] = sn.G.nodes[ni]["InfoMat_bar"] @ sn.MM.F @ inv(Oi_k_1_wt) @ qi_k_1_wt
            uFk[ni] = uiF_k_1
        sn.MM.F = expm(t_sam * sc.Jacboi_matrix(np.zeros(6)))
        sn.Recorder["S_rea"].append(np.array([sn.G.nodes[ni]["S_rea"] for ni in sn.G.nodes()]))
        sn.Recorder["S_est"].append(np.array([sn.G.nodes[ni]["S_est"] for ni in sn.G.nodes()]))
        sn.Recorder["C1"].append([sn.G.nodes[ni]["C1"] for ni in sn.Ns_nodes])
        if sn.Time >= T_total:
            break
    if info is None:
        return deepcopy(sn.Recorder)
    else:
        return {k: sn.Recorder[k] for k in info}


def main():
    """definite the orbital parameter of reference satellite"""
    a, e, inc, w, RAAN, TA = 10000, 0.0, d2r(60), d2r(45), d2r(50.2), d2r(30)

    r = a * (1 - e ** 2) / (1 + e * cos(TA))
    h = np.sqrt(a * (1 - e ** 2) * mu)
    vx = np.sqrt(np.round(2 * mu / r - mu / a - h ** 2 / r ** 2, 10))
    the = w + TA
    sat_cluster = RelativeMotionModel_J2([r, vx, h, the, inc], [a, e, inc, w, RAAN, TA])
    fsat_num = 20
    alpha = np.random.uniform(-np.pi, np.pi)
    beta = np.random.uniform(np.pi / 2.1, np.pi / 2)
    n = [cos(alpha) * cos(beta), sin(alpha) * cos(beta), sin(beta)]
    an = np.array([-n[2], 0, n[0]]) / np.linalg.norm(np.array([-n[2], 0, n[0]]))
    bn = np.array([0, -n[2], n[1]]) / np.linalg.norm(np.array([0, -n[2], n[1]]))
    rt = 3.5
    ri = 15.0
    tar = np.zeros((fsat_num, 3))
    for j in range(fsat_num):
        the_j = j / fsat_num * 2 * np.pi
        tar[j] = rt * np.array([an[0] * sin(the_j) + bn[0] * cos(the_j),
                                an[1] * sin(the_j) + bn[1] * cos(the_j),
                                an[2] * sin(the_j) + bn[2] * cos(the_j)])
        pj = [(an[0] * sin(the_j + np.pi / 2.5) + bn[0] * cos(the_j + np.pi / 2.5)) * ri + uniform(-2.0, 2.0),
              (an[1] * sin(the_j + np.pi / 2.5) + bn[1] * cos(the_j + np.pi / 2.5)) * ri + uniform(-2.0, 2.0),
              (an[2] * sin(the_j + np.pi / 2.5) + bn[2] * cos(the_j + np.pi / 2.5)) * ri + uniform(-2.0, 2.0)]
        vj = [0, 0, 0]
        sj = pj + vj
        sat_cluster.add_follow_sat(sj)

    A = sat_cluster.Jacboi_matrix(np.zeros(6))
    Fk = expm(T_sam * A)
    sigma_v = 2e-6
    sigma_w = 4e-3
    V = np.concatenate([np.zeros((3, 3)), np.eye(3) * sigma_v], axis=0)
    Vd = T_sam * (expm(A * T_sam) + expm(A * T_sam / 2) * 4 + np.eye(6)) / 6 @ V
    Q = Vd @ Vd.T
    R = sigma_w ** 2 * np.eye(3)
    H = np.block([np.eye(3), np.zeros((3, 3))])
    mm = MotionModel(Fk, H, Q, R)
    mm.B = Vd / sigma_v
    sn = SensorNetwork(Ns=6, Nc=fsat_num - 6, Sr=sat_cluster.fsat_state, No=obs_nums, T_sam=T_sam, MM=mm)
    sn.plot_Graph(p=True)
    record = distributed_EKF_sc(sn, sat_cluster, T_sam, tar, period=int(10 / 0.17))
    # # record = {}
    record["O_pos"] = obs_pos[:, :int(N_sample) + 1, :]
    record["O_rad"] = obs_rad
    record["S_tar"] = tar
    tv = vv.satellite_cluster_visualization(record)
    tv.show_obstacle()
    # # # # # tv.plot_fsat_track([5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    tv.plot_fsat_track(sn.G.nodes())
    # for ne in range(6):
    #     tv.draw_signal_trigger1(ne, dt=T_sam)
    # tv.draw_signal_trigger2(ne, dt=T_sam)
    # tv.satellite_collision_avoidance_test(rs * 2)
    # tv.obstacle_collision_avoidance_test()


def mento_carlo_sim(N, ifsave=True, **kwargs):
    a, e, inc, w, RAAN, TA = 10000, 0.0, d2r(60), d2r(45), d2r(50.2), d2r(30)
    r = a * (1 - e ** 2) / (1 + e * cos(TA))
    h = np.sqrt(a * (1 - e ** 2) * mu)
    vx = np.sqrt(np.round(2 * mu / r - mu / a - h ** 2 / r ** 2, 10))
    the = w + TA
    sat_cluster = RelativeMotionModel_J2([r, vx, h, the, inc], [a, e, inc, w, RAAN, TA])

    fsat_num = 20
    A = sat_cluster.Jacboi_matrix(np.zeros(6))
    Fk = expm(T_sam * A)
    sigma_v = 2e-6
    sigma_w = 4e-3
    V = np.concatenate([np.zeros((3, 3)), np.eye(3) * sigma_v], axis=0)
    Vd = T_sam * (expm(A * T_sam) + expm(A * T_sam / 2) * 4 + np.eye(6)) / 6 @ V
    Q = Vd @ Vd.T
    R = sigma_w ** 2 * np.eye(3)
    H = np.block([np.eye(3), np.zeros((3, 3))])
    mm = MotionModel(Fk, H, Q, R)
    mm.B = Vd / sigma_v

    Recorder = []
    for _ in range(N):
        alpha = np.random.uniform(-np.pi, np.pi)
        beta = np.random.uniform(np.pi / 4, np.pi / 2)
        n = [cos(alpha) * cos(beta), sin(alpha) * cos(beta), sin(beta)]
        an = np.array([-n[2], 0, n[0]]) / np.linalg.norm(np.array([-n[2], 0, n[0]]))
        bn = np.array([0, -n[2], n[1]]) / np.linalg.norm(np.array([0, -n[2], n[1]]))
        rt = 3.0
        ri = 15.0
        tar = np.zeros((fsat_num, 3))
        for j in range(fsat_num):
            the_j = j / fsat_num * 2 * np.pi
            tar[j] = rt * np.array([an[0] * sin(the_j) + bn[0] * cos(the_j),
                                    an[1] * sin(the_j) + bn[1] * cos(the_j),
                                    an[2] * sin(the_j) + bn[2] * cos(the_j)])
            pj = [(an[0] * sin(the_j + np.pi / 2.5) + bn[0] * cos(the_j + np.pi / 2.5)) * ri + uniform(-2.0, 2.0),
                  (an[1] * sin(the_j + np.pi / 2.5) + bn[1] * cos(the_j + np.pi / 2.5)) * ri + uniform(-2.0, 2.0),
                  (an[2] * sin(the_j + np.pi / 2.5) + bn[2] * cos(the_j + np.pi / 2.5)) * ri + uniform(-1.0, 1.0)]
            vj = [0, 0, 0]
            sj = pj + vj
            sat_cluster.add_follow_sat(sj)
        sn = SensorNetwork(Ns=6, Nc=fsat_num - 6, Sr=sat_cluster.fsat_state, No=obs_nums, T_sam=T_sam, MM=mm)
        record = distributed_EKF_sc(sn, sat_cluster, T_sam, tar, info=("S_est", "S_rea", "C1"), **kwargs)
        Recorder.append(record)
        sat_cluster.clear()
    if ifsave:
        file_name = f"mento_carlo_data/PD-C1-{N}-rate={round(1000 / kwargs['period'])}.json"
        with open(file_name, "w") as f:
            json.dump(Recorder, f, default=convert_np_array)
            print(f"{file_name} has been successfully saved.")
    return


if __name__ == "__main__":
    # mento_carlo_sim(50, 5, 10)
    # mento_carlo_sim(50, 10, 10)
    # mento_carlo_sim(50, 15, 10)
    # mento_carlo_sim(50, 5, 5)
    # mento_carlo_sim(50, 5, 10)
    # mento_carlo_sim(50, 5, 15)
    # mento_carlo_sim(50, 15, 5)
    # mento_carlo_sim(50, 15, 10)
    # mento_carlo_sim(50, period=int(T_sam / 0.17))
    # for a_ in range(5, 16):
    #     mento_carlo_sim(8, a_, 15)
    # for b_ in range(5, 16):
    #     mento_carlo_sim(8, 15, b_)
    main()
