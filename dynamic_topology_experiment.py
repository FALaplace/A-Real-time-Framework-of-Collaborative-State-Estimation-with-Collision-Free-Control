from pathlib import Path
from copy import deepcopy
import json
import random

import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import expm
import networkx as nx
from MotionModel import RelativeMotionModel_J2
from Topology import MotionModel, SensorNetwork
from satellite_cluster_state_estimation import APF_controller, rtf2xyz, umax, rs, mu, d2r, T_total
from tqdm import trange

LEADERS = np.array([0, 3, 7, 8, 12, 16])
FOLLOWERS = np.array([1, 2, 4, 5, 6, 9, 10, 11, 13, 14, 15, 17, 18, 19])


def component_union(*D_family, gamma=0.2):
    p_list = [tuple(np.r_[y, rho]) for Di in D_family for y, rho in Di]
    if not p_list:
        return []
    p_all = np.array(sorted(set(p_list)), dtype=float)
    d = (np.max(np.abs(p_all[:, None, :3] - p_all[None, :, :3]), axis=2) + np.abs(
        p_all[:, None, 3] - p_all[None, :, 3]))
    A_D = d < gamma
    unseen = set(range(len(p_all)))
    D_mean = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        stack, component = [root], [root]
        while stack:
            current = stack.pop()
            neighbors = sorted(unseen.intersection(np.flatnonzero(A_D[current])))
            unseen.difference_update(neighbors)
            component.extend(neighbors)
            stack.extend(neighbors)
        p_mean = p_all[sorted(component)].mean(axis=0)
        D_mean.append((p_mean[:3], float(p_mean[3])))
    return D_mean


def fuse_state(sii, Pii, sij_list, Pij_list):
    if not sij_list:
        return sii.copy(), Pii.copy(), np.array([1.0])
    P_all = np.asarray([Pii, *Pij_list])
    P_norm = np.array([norm(Pj, 2) for Pj in P_all])
    wi = 1.0 / P_norm
    wi /= np.sum(wi)
    siF = wi @ np.asarray([sii, *sij_list])
    PiF = wi[0] ** 2 * Pii
    for wij, Pij in zip(wi[1:], Pij_list):
        PiF += (wij ** 2 + 2 * wi[0] * wij) * Pij
    return siF, PiF, wi


class DynamicTopologyExperiment:
    leaders = LEADERS
    followers = FOLLOWERS
    scenarios = ("switching", "fixed")
    fuse_state = staticmethod(fuse_state)
    APF_controller = staticmethod(APF_controller)

    def __init__(self):
        self.Ns, self.Nc = len(self.leaders), len(self.followers)
        self.N = self.Ns + self.Nc
        self.steps, self.dt, self.seed = 2400, 10.0, 1030
        self.time = np.arange(self.steps + 1) * self.dt
        self.graph_seed = 1030
        sc, mm = self.make_models()
        self.P0 = np.block([[mm.R, mm.R / self.dt], [mm.R / self.dt, 2 * mm.R / self.dt ** 2]])
        self.sigma_w = 0.004
        self.sigma_wr = np.array([mm.dr, mm.dt, mm.df])
        self.alpha = 0.05 / np.sqrt(norm(mm.R))
        self.beta = 10
        np.random.seed(0)
        random.seed(self.graph_seed)
        sn = self.make_sensor_network(np.zeros((self.N, 6)), mm)
        self.A0 = nx.to_numpy_array(sn.G, nodelist=range(self.N), dtype=bool)
        self.output_dir = "Dynamic_Topology_Results"
        obs_dir = Path(self.output_dir)
        obs_dir.mkdir(parents=True, exist_ok=True)
        pos_path = obs_dir / "obstacle_trajectory.npy"
        rad_path = obs_dir / "obstacle_radius.npy"
        if pos_path.is_file() and rad_path.is_file():
            self.obs_pos = np.load(pos_path)
            self.obs_rad = np.load(rad_path)
        else:
            obs_rng = np.random.RandomState(40)
            self.obs_pos = np.zeros((4, self.steps + 1, 3))
            for j in range(4):
                dt_obs = obs_rng.random() * np.pi / 2
                the_obs = np.linspace(j * np.pi / 2, j * np.pi / 2 + dt_obs * self.time[-1] / T_total, self.steps + 1)
                xr, yr, zr = obs_rng.uniform(6, 11), obs_rng.uniform(6, 11), obs_rng.uniform(-.2, .2)
                self.obs_pos[j] = np.c_[xr * np.cos(the_obs), yr * np.sin(the_obs), zr * np.sin(.5 * the_obs)]
            self.obs_rad = obs_rng.uniform(.5, 1., 4)
            np.save(pos_path, self.obs_pos)
            np.save(rad_path, self.obs_rad)

    def make_sensor_network(self, S0, mm):
        return SensorNetwork(self.Ns, self.Nc, S0, 4, self.dt, mm)

    def make_models(self):
        a, e, inc, w, RAAN, TA = 10000, 0.0, d2r(60), d2r(45), d2r(50.2), d2r(30)
        r = a * (1 - e ** 2) / (1 + e * np.cos(TA))
        h = np.sqrt(a * (1 - e ** 2) * mu)
        vx = np.sqrt(np.round(2 * mu / r - mu / a - h ** 2 / r ** 2, 10))
        sc = RelativeMotionModel_J2([r, vx, h, w + TA, inc], [a, e, inc, w, RAAN, TA])
        A = sc.Jacboi_matrix(np.zeros(6))
        Fk = expm(self.dt * A)
        V = np.vstack((np.zeros((3, 3)), np.eye(3) * 2e-6))
        Vd = self.dt * (Fk + 4 * expm(A * self.dt / 2) + np.eye(6)) / 6 @ V
        mm = MotionModel(Fk, np.c_[np.eye(3), np.zeros((3, 3))], Vd @ Vd.T, 0.004 ** 2 * np.eye(3))
        mm.B = Vd / 2e-6
        return sc, mm

    def generate_case(self, case_id, scenario):
        seed = [self.seed, case_id]
        scene_seed = seed + [0, 0]
        rng = np.random.default_rng(scene_seed)
        alpha, beta = rng.uniform(-np.pi, np.pi), rng.uniform(np.pi / 4, np.pi / 2)
        n = np.array([np.cos(alpha) * np.cos(beta), np.sin(alpha) * np.cos(beta), np.sin(beta)])
        an, bn = np.array([-n[2], 0., n[0]]), np.array([0., -n[2], n[1]])
        an, bn = an / norm(an), bn / norm(bn)
        the_j = np.arange(self.N) / self.N * 2 * np.pi
        tar = 3 * (np.sin(the_j)[:, None] * an + np.cos(the_j)[:, None] * bn)
        p0 = 15 * (np.sin(the_j + np.pi / 2.5)[:, None] * an + np.cos(the_j + np.pi / 2.5)[:, None] * bn)
        p0 += rng.uniform([-2., -2., -1.], [2., 2., 1.], size=(self.N, 3))
        S0 = np.c_[p0, np.zeros((self.N, 3))]
        init_rng = np.random.default_rng(seed + [1])
        noise_rng = np.random.default_rng(seed + [2])
        event_rng = np.random.default_rng(seed + [3])
        if scenario == "switching":
            dt_index = np.arange(3000, self.time[-1], 600)
            G_LIST = []
            for _ in dt_index:
                Gt = self.A0 & (event_rng.random((self.N, self.N)) < .7)
                for ni in self.followers:
                    Nei_i = np.flatnonzero(self.A0[:, ni])
                    while not Gt[Nei_i, ni].any():
                        Gt[Nei_i, ni] = event_rng.random(len(Nei_i)) < .7
                G_LIST.append(Gt)
            scenario_config = dict(times=dt_index, masks=np.array(G_LIST, dtype=bool).reshape(-1, self.N, self.N))
        elif scenario == "dropout":
            scenario_config = dict(node=int(event_rng.choice(self.leaders)),
                                   time=int(event_rng.integers(300, 601)) * self.dt)
        elif scenario == "outage":
            ni = int(event_rng.choice(self.followers))
            durations = np.array([100., 300., 600.])
            starts = np.array([3000., 9000., 15000.])
            scenario_config = dict(node=ni, start=starts, end=starts + durations)
        else:
            scenario_config = {}
        return dict(
            S0_real=S0, S0_esti=S0 + init_rng.multivariate_normal(np.zeros(6), self.P0, size=self.N), St=tar,
            w_list=noise_rng.normal(size=(self.steps, self.Ns, 3)) * self.sigma_w,
            wr_list=noise_rng.normal(size=(self.steps, self.Ns, self.N, 3)) * self.sigma_wr,
            wr_obs_list=noise_rng.normal(size=(self.steps, self.Ns, 4, 4)) * np.r_[self.sigma_wr, .001],
            scenario_config=scenario_config)

    def switching(self, case, A_available):
        schedule = case["scenario_config"]
        for start, mask in zip(schedule["times"], schedule["masks"]):
            A_available[self.time >= start] = mask

    def dropout(self, case, I_service):
        config = case["scenario_config"]
        ni, start = config["node"], config["time"]
        I_service[self.time >= start, int(np.flatnonzero(self.leaders == ni)[0])] = False

    def outage(self, case, A_available):
        config = case["scenario_config"]
        ni = config["node"]
        for start, end in zip(config["start"], config["end"]):
            select = np.flatnonzero((self.time >= start) & (self.time < end))
            A_available[select, :, ni] = False

    def availability(self, case, scenario):
        A_available = np.broadcast_to(self.A0, (self.steps + 1, self.N, self.N)).copy()
        I_service = np.ones((self.steps + 1, self.Ns), dtype=bool)
        if scenario == "switching":
            self.switching(case, A_available)
        elif scenario == "dropout":
            self.dropout(case, I_service)
        elif scenario == "outage":
            self.outage(case, A_available)
        elif scenario != "fixed":
            raise ValueError(scenario)
        return A_available, I_service

    def make_network(self, case):
        sc, mm = self.make_models()
        for si0 in case["S0_real"]:
            sc.add_follow_sat(si0.copy())
        np.random.seed(0)
        random.seed(self.graph_seed)
        sn = self.make_sensor_network(sc.fsat_state, mm)
        for ni in range(self.N):
            node = sn.G.nodes[ni]
            node["NodeType"] = int(ni in self.leaders)
            node["S_rea"] = sc.fsat_state[ni].copy()
            node["D"] = []
            node["S_est"] = case["S0_esti"][ni].copy()
            node["P_est"] = self.P0.copy()
            node["InfoMat"] = inv(self.P0)
            node["InfoVec"] = node["InfoMat"] @ node["S_est"]
            node["InfoMat_fu"] = node["InfoMat"].copy()
            node["InfoVec_fu"] = node["InfoVec"].copy()
            node["InfoMat_bar"] = inv(mm.F @ self.P0 @ mm.F.T + mm.Q)
            node["InfoVec_bar"] = node["InfoMat_bar"] @ mm.F @ node["S_est"]
            node["C1"] = 0
        for ni in self.leaders:
            node = sn.G.nodes[ni]
            Nei_i = list(sn.G.successors(ni))
            node["InfoMat_nei"] = {vj: sn.G.nodes[vj]["InfoMat"].copy() for vj in Nei_i}
            node["InfoVec_nei"] = {vj: sn.G.nodes[vj]["InfoVec"].copy() for vj in Nei_i}
            node["C2"] = {vj: 0 for vj in Nei_i}
            node["D_nei"] = {vj: [] for vj in Nei_i}
            node["D_bar"] = {vj: [] for vj in Nei_i}
        return sc, sn

    @staticmethod
    def observe(p, w):
        r = norm(p)
        return np.array([r, np.arctan2(p[1], p[0]), np.arcsin(p[2] / r)]) + w

    def sat_update(self, ni, k, case, sc, sn, uFk):
        """单星本地预测、观测更新、环境感知和触发判定；不执行融合。"""
        mm, nodes = sn.MM, sn.G.nodes
        H, R_inv = mm.H, inv(mm.R)
        node = nodes[ni]
        PiF_kk = inv(node["InfoMat_fu"])
        Oi_k_1k = inv(mm.F @ PiF_kk @ mm.F.T + mm.Q)
        qi_k_1k = Oi_k_1k @ (mm.F @ PiF_kk @ node["InfoVec_fu"] + mm.B @ uFk[ni])
        if node["NodeType"]:
            li = int(np.flatnonzero(self.leaders == ni)[0])
            yi_k_1 = H @ node["S_rea"] + case["w_list"][k - 1, li]
            node["InfoVec"] = qi_k_1k + H.T @ R_inv @ yi_k_1
            node["InfoMat"] = Oi_k_1k + H.T @ R_inv @ H
            node["P_est"] = inv(node["InfoMat"])
            node["S_est"] = node["P_est"] @ node["InfoVec"]
            for io in range(4):
                wi_obs = case["wr_obs_list"][k - 1, li, io]
                y_rtf = self.observe(self.obs_pos[io, k], wi_obs[:3])
                rho_obs = self.obs_rad[io] + wi_obs[3]
                sn.collision_avoidance_update(ni, (rtf2xyz(y_rtf), rho_obs))
            for vj in range(self.N):
                if vj == ni:
                    continue
                y_rtf = self.observe(nodes[vj]["S_rea"][:3], case["wr_list"][k - 1, li, vj])
                yji_k_1 = rtf2xyz(y_rtf)
                sn.collision_avoidance_update(ni, (yji_k_1, rs))
                if self.A0[ni, vj]:
                    Rji_k_1 = mm.calculate_R(*y_rtf)
                    Oji_kk = node["InfoMat_nei"][vj]
                    qji_kk = node["InfoVec_nei"][vj]
                    Pji_kk = inv(Oji_kk)
                    sji_kk = Pji_kk @ qji_kk
                    uji = np.clip(-self.APF_controller(sji_kk, case["St"][vj], node["D_bar"][vj]), -umax, umax)
                    Oji_k_1k = inv(mm.F @ Pji_kk @ mm.F.T + mm.Q)
                    qji_k_1k = Oji_k_1k @ (mm.F @ sji_kk + sc.B @ uji)
                    Rji_inv = inv(Rji_k_1)
                    node["InfoVec_nei"][vj] = qji_k_1k + H.T @ Rji_inv @ yji_k_1
                    node["InfoMat_nei"][vj] = Oji_k_1k + H.T @ Rji_inv @ H
                    node["D_nei"][vj] = sn.obtain_Dji(ni, yji_k_1)
                    node["C2"][vj] = sn.collision_detect_trigger(ni, vj)
                    if node["C2"][vj]:
                        node["D_bar"][vj] = deepcopy(node["D_nei"][vj])
            sn.state_estimation_trigger(ni, a=self.alpha, b=self.beta)
        else:
            node["InfoVec"], node["InfoMat"] = qi_k_1k, Oi_k_1k
            node["P_est"] = inv(Oi_k_1k)
            node["S_est"] = node["P_est"] @ qi_k_1k

    def sat_fusion(self, ni, case, sn, uFk, R_state, R_safe):
        """全部本地更新完成后，按实际收到的信息进行单星融合和控制更新。"""
        mm, nodes = sn.MM, sn.G.nodes
        node = nodes[ni]
        Nei_i = np.flatnonzero(R_state[:, ni])
        Pij_list = [inv(nodes[vj]["InfoMat_nei"][ni]) for vj in Nei_i]
        sij_list = [Pij_k_1k_1 @ nodes[vj]["InfoVec_nei"][ni] for vj, Pij_k_1k_1 in
                    zip(Nei_i, Pij_list)]
        siF_k_1k_1, PiF_k_1k_1, _ = self.fuse_state(node["S_est"], node["P_est"], sij_list, Pij_list)
        D_family = [nodes[vj]["D_nei"][ni] for vj in np.flatnonzero(R_safe[:, ni])]
        if node["NodeType"]:
            D_family.append(node["D"])
        Di = component_union(*D_family, gamma=sn.gamma)
        node["D"] = Di
        uFk[ni] = np.clip(-self.APF_controller(siF_k_1k_1, case["St"][ni], Di), -umax, umax)
        node["InfoMat_fu"] = inv(PiF_k_1k_1)
        node["InfoVec_fu"] = node["InfoMat_fu"] @ siF_k_1k_1
        if node["NodeType"]:
            ci = node["C1"]
            qi_k_1_wt = node["InfoVec"] if ci else node["InfoVec_bar"]
            Oi_k_1_wt = node["InfoMat"] if ci else node["InfoMat_bar"]
            Pi_k_1_wt = inv(Oi_k_1_wt)
            node["InfoMat_bar"] = inv(mm.F @ Pi_k_1_wt @ mm.F.T + mm.Q)
            node["InfoVec_bar"] = node["InfoMat_bar"] @ mm.F @ Pi_k_1_wt @ qi_k_1_wt

    def cluster_update(self, k, case, sc, sn, uFk, A_available, I_service):
        """推进一步：真实运动、本地更新、消息传递、融合和控制。"""
        mm, nodes = sn.MM, sn.G.nodes
        sc.upgrade_state(uFk, self.dt)
        sn.Time = self.time[k]
        for ni in range(self.N):
            nodes[ni]["S_rea"] = sc.fsat_state[ni].copy()
        for ni in range(self.N):
            self.sat_update(ni, k, case, sc, sn, uFk)
        C1 = np.zeros(self.N, bool)
        C2 = np.zeros((self.N, self.N), bool)
        for ni in self.leaders:
            C1[ni] = nodes[ni]["C1"]
            for vj in nodes[ni]["C2"]:
                C2[ni, vj] = nodes[ni]["C2"][vj]
        I_active = np.ones(self.N, bool)
        I_active[self.leaders] = I_service
        R_state = A_available & I_active[:, None] & C1[:, None]
        R_safe = A_available & I_active[:, None] & C2
        for ni in range(self.N):
            self.sat_fusion(ni, case, sn, uFk, R_state, R_safe)
        mm.F = expm(self.dt * sc.Jacboi_matrix(np.zeros(6)))
        return R_state

    def run(self, case, scenario):
        sc, sn = self.make_network(case)
        A_available, I_service = self.availability(case, scenario)
        uFk = np.array(
            [np.clip(-self.APF_controller(case["S0_esti"][ni], case["St"][ni], []), -umax, umax) for ni in range(self.N)])
        if scenario in ("switching", "dropout"):
            record = dict(S_est=[case["S0_esti"].tolist()], S_real=[case["S0_real"].tolist()],
                          C1=[[int(sn.G.nodes[ni]["C1"]) for ni in self.leaders]])
            if scenario == "switching":
                record["G_LIST"] = A_available.tolist()
            else:
                ni = case["scenario_config"]["node"]
                record["dropout_node"] = ni
                record["dropout_time"] = case["scenario_config"]["time"]
                record["out_neighbors"] = np.flatnonzero(self.A0[ni]).tolist()
        elif scenario == "outage":
            config = case["scenario_config"]
            ni = config["node"]
            record = dict(S_est=[case["S0_esti"][ni].tolist()], S_real=[case["S0_real"][ni].tolist()],
                          C=[0], outage_start=config["start"].tolist(),
                          outage_duration=(config["end"] - config["start"]).tolist())
        for k in range(1, self.steps + 1):
            R_state = self.cluster_update(k, case, sc, sn, uFk, A_available[k], I_service[k])
            if scenario in ("switching", "dropout"):
                record["S_est"].append([sn.G.nodes[ni]["S_est"].tolist() for ni in range(self.N)])
                record["S_real"].append(sc.fsat_state.tolist())
                record["C1"].append([int(sn.G.nodes[ni]["C1"]) for ni in self.leaders])
            elif scenario == "outage":
                record["S_est"].append(sn.G.nodes[ni]["S_est"].tolist())
                record["S_real"].append(sc.fsat_state[ni].tolist())
                record["C"].append(int(R_state[:, ni].any()))
        if scenario in ("switching", "dropout", "outage"):
            return record
        return None

    def run_Mento_Carlo(self, N_mc: int, scenario: str):
        if scenario not in self.scenarios:
            return None
        case_dir = Path(self.output_dir) / scenario
        case_dir.mkdir(parents=True, exist_ok=True)
        for i in trange(N_mc):
            case = self.generate_case(i, scenario)
            record = self.run(case, scenario)
            with (case_dir / f"case_{i}.json").open("w", encoding="utf-8") as f:
                json.dump(record, f)
        return None


class DropoutExperiment(DynamicTopologyExperiment):
    """三个领航星向一个跟随星供给信息，比较固定退出 0、1、2 个信息源。"""
    leaders = np.array([0, 1, 2])
    followers = np.array([3])
    scenarios = ("dropout_0", "dropout_1", "dropout_2")

    def __init__(self):
        super().__init__()
        self.dropout_time = 6000.0
        self.output_dir = str(Path(self.output_dir) / "dropout")

    def make_sensor_network(self, S0, mm):
        # 原构造器写死了 20 星编号；复用初始化后替换图，实际仿真仅推进 4 星。
        sn = SensorNetwork(6, 14, np.zeros((20, 6)), 4, self.dt, mm)
        sn.Ns, sn.Nc, sn.N = self.Ns, self.Nc, self.N
        sn.Ns_nodes, sn.Nc_nodes = self.leaders.tolist(), self.followers.tolist()
        sn.G = nx.DiGraph()
        sn.G.add_nodes_from(range(self.N))
        sn.G.add_edges_from((int(ni), 3) for ni in self.leaders)
        sn.Recorder = {}
        return sn

    def generate_case(self, case_id, scenario):
        case = super().generate_case(case_id, "fixed")
        nodes = {"dropout_0": [], "dropout_1": [0], "dropout_2": [0, 1]}[scenario]
        case["scenario_config"] = dict(nodes=nodes, time=self.dropout_time)
        return case

    def dropout(self, case, I_service):
        config = case["scenario_config"]
        for ni in config["nodes"]:
            li = int(np.flatnonzero(self.leaders == ni)[0])
            I_service[self.time >= config["time"], li] = False

    def availability(self, case, scenario):
        return super().availability(case, "dropout")

    def run(self, case, scenario):
        sc, sn = self.make_network(case)
        A_available, I_service = self.availability(case, scenario)
        uFk = np.array([np.clip(-self.APF_controller(case["S0_esti"][ni], case["St"][ni], []), -umax, umax)
                        for ni in range(self.N)])
        ni = int(self.followers[0])
        config = case["scenario_config"]
        record = dict(S_fused=[case["S0_esti"][ni].tolist()], S_real=[case["S0_real"][ni].tolist()],
                      C=[0], dropout_nodes=config["nodes"], dropout_time=config["time"])
        for k in range(1, self.steps + 1):
            R_state = self.cluster_update(k, case, sc, sn, uFk, A_available[k], I_service[k])
            node = sn.G.nodes[ni]
            siF = inv(node["InfoMat_fu"]) @ node["InfoVec_fu"]
            record["S_fused"].append(siF.tolist())
            record["S_real"].append(sc.fsat_state[ni].tolist())
            record["C"].append(int(R_state[:, ni].sum()))
        return record

    def run_Mento_Carlo(self, N_mc: int, scenario: str):
        case_dir = Path(self.output_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        for i in trange(N_mc):
            case = self.generate_case(i, scenario)
            record = self.run(case, scenario)
            with (case_dir / f"case_{i}_{scenario}.json").open("w", encoding="utf-8") as f:
                json.dump(record, f)


class OutageExperiment(DropoutExperiment):
    """每个 case 随机抽取入链路数量和领航星，考察跟随星完全断联及恢复。"""
    scenarios = ("outage",)

    def __init__(self):
        super().__init__()
        self.output_dir = str(Path(self.output_dir).with_name("outage"))
        self.outage_start = np.array([3000., 9000., 15000.])
        self.outage_duration = np.array([100., 300., 600.])

    def generate_case(self, case_id, scenario):
        case = DynamicTopologyExperiment.generate_case(self, case_id, "fixed")
        rng = np.random.default_rng([self.seed, case_id, 3])
        n = rng.integers(1, self.Ns + 1)
        Nei_i = rng.permutation(self.leaders)[:n]
        case["scenario_config"] = dict(node=int(self.followers[0]), in_neighbors=Nei_i.tolist(),
                                       start=self.outage_start.copy(),
                                       end=self.outage_start + self.outage_duration)
        return case

    def availability(self, case, scenario):
        config = case["scenario_config"]
        A_available = np.zeros((self.steps + 1, self.N, self.N), dtype=bool)
        A_available[:, config["in_neighbors"], config["node"]] = True
        I_service = np.ones((self.steps + 1, self.Ns), dtype=bool)
        self.outage(case, A_available)
        return A_available, I_service

    def run(self, case, scenario):
        sc, sn = self.make_network(case)
        A_available, I_service = self.availability(case, scenario)
        uFk = np.array([np.clip(-self.APF_controller(case["S0_esti"][ni], case["St"][ni], []), -umax, umax)
                        for ni in range(self.N)])
        config = case["scenario_config"]
        ni = config["node"]
        record = dict(S_fused=[], S_real=[], C=[], in_neighbors=config["in_neighbors"],
                      outage_start=config["start"].tolist(),
                      outage_duration=(config["end"] - config["start"]).tolist(),
                      d_ss=[], d_so=[], rs_safe=0.2, obs_rad=self.obs_rad.tolist())
        for k in range(self.steps + 1):
            ci = 0
            if k > 0:
                R_state = self.cluster_update(k, case, sc, sn, uFk, A_available[k], I_service[k])
                ci = int(R_state[:, ni].any())
            node = sn.G.nodes[ni]
            siF = inv(node["InfoMat_fu"]) @ node["InfoVec_fu"]
            pi = sc.fsat_state[ni, :3]
            record["S_fused"].append(siF.tolist())
            record["S_real"].append(sc.fsat_state[ni].tolist())
            record["C"].append(ci)
            # 距离单位为 km；只统计该 follower 与其他物体的真实距离。
            record["d_ss"].append(norm(sc.fsat_state[self.leaders, :3] - pi, axis=1).tolist())
            record["d_so"].append(norm(self.obs_pos[:, k] - pi, axis=1).tolist())
        return record


if __name__ == "__main__":
    N = 40
    experiment = OutageExperiment()
    experiment.run_Mento_Carlo(N, experiment.scenarios[0])
