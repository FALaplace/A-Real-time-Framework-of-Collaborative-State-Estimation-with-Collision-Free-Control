import numpy as np
from numpy import cos, sin
import networkx as nx
import random
from numpy.linalg import inv, norm, eigvals
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch
from typing import Dict, List, Tuple, Set
from itertools import chain

np.random.seed(42)
random.seed(41)
norm_MatrixInduced = lambda x, M: np.sqrt(np.inner(x, M @ x))


def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.isclose(c, -1.0):  # 180度翻转的特殊情况
        return -np.eye(3)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))


def plot_cone(ax, start, end, height, radius=0.1, resolution=30, color="green", alpha=1.0):
    start, end = np.array(start), np.array(end)
    direction = (end - start) * height / np.linalg.norm(end - start)
    # height = np.linalg.norm(direction)
    start_ = end - direction

    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(0, height, 2)  # 只要底面和顶点
    theta, z = np.meshgrid(theta, z)
    R = radius * (1 - z / height)

    X = R * np.cos(theta)
    Y = R * np.sin(theta)
    Z = z

    cone_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()])
    Rmat = rotation_matrix_from_vectors(np.array([0, 0, 1]), direction)
    cone_points_rot = Rmat @ cone_points

    Xr = cone_points_rot[0].reshape(X.shape) + start_[0]
    Yr = cone_points_rot[1].reshape(Y.shape) + start_[1]
    Zr = cone_points_rot[2].reshape(Z.shape) + start_[2]

    ax.plot_surface(Xr, Yr, Zr, color=color, alpha=alpha)


def collision_set_mean(D: List[Tuple[np.ndarray, float]]):
    y_list, rho_list = tuple(zip(*D))
    y_mean = np.mean(y_list, axis=0)
    rho_mean = np.mean(rho_list)
    return y_mean, rho_mean


def dp(p1, p2):
    return norm(p1[0] - p2[0], np.inf) + abs(p1[1] - p2[1])


def is_semi_positive(M):
    return np.all(eigvals(M) >= 0)


def clustered_mean_Union(*D: List[Tuple[np.ndarray, float]], g: float) -> List[Tuple[np.ndarray, float]]:
    S = []
    for Di in D:
        S.extend(Di)
    if not S:
        return []
    clusters = []
    for p in S:
        assigned = False
        for cluster in clusters:
            if any(dp(p, p_) < g for p_ in cluster):
                cluster.append(p)
                assigned = True
                break
        if not assigned:
            clusters.append([p])

    merged_clusters = []
    for cluster in clusters:
        merged = False
        for mc in merged_clusters:
            if any(dp(p1, p2) < g for p1 in cluster for p2 in mc):
                mc.extend(cluster)
                merged = True
                break
        if not merged:
            merged_clusters.append(cluster.copy())
    D_mean = []
    for cluster in merged_clusters:
        mean = collision_set_mean(cluster)
        D_mean.append(mean)
    return D_mean


class MotionModel:
    """
    s(k+1) = F @ s(k) + vi
    y(k) = H @ s(k) + wi
    yr(k) = h(s(k)) + wri
    vi ~ N(0, Q)
    wi ~ N(0, R)
    wri ~ N(0, Rr)
    """

    def __init__(self, F, H, Q, R):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.dr = 0.001
        self.dt = np.pi / 180 * 0.1
        self.df = np.pi / 180 * 0.1
        self.B = None

    def calculate_R(self, r, t, f):
        ct, st, cf, sf = cos(t), sin(t), cos(f), sin(f)
        R_obs = np.zeros((3, 3))
        R_obs[0, 0] = (ct * cf * self.dr) ** 2 + (r * st * cf * self.dt) ** 2 + (r * ct * sf * self.df) ** 2
        R_obs[0, 1] = ct * st * ((cf * self.dr) ** 2 - (cf * r * self.dt) ** 2 + (sf * r * self.df) ** 2)
        R_obs[0, 2] = ct * cf * sf * (self.dr ** 2 - (r * self.df) ** 2)
        R_obs[1, 1] = (st * cf * self.dr) ** 2 + (r * ct * cf * self.dt) ** 2 + (r * st * sf * self.df) ** 2
        R_obs[1, 2] = st * cf * sf * (self.dr ** 2 - (r * self.df) ** 2)
        R_obs[2, 2] = (self.dr * sf) ** 2 + (r * cf * self.df) ** 2
        R_obs[1, 0] = R_obs[0, 1]
        R_obs[2, 0] = R_obs[0, 2]
        R_obs[1, 2] = R_obs[2, 1]
        return R_obs.copy()


class SensorNetwork:
    def __init__(self, Ns, Nc, Sr, No, T_sam, MM: MotionModel):
        self.MM = MM
        self.Time = 0
        self.T_sam = T_sam
        self.Ns = Ns  # the number of sensor nodes
        self.Nc = Nc  # the number of communication nodes
        self.N = Ns + Nc  # total node number
        P0 = np.block([[MM.R, MM.R / T_sam], [MM.R / T_sam, 2 * MM.R / T_sam ** 2]])
        self.G: nx.DiGraph = nx.gnm_random_graph(self.N, 3 * self.N, seed=46, directed=True)
        # lst = [1] * Ns + [0] * Nc
        # random.shuffle(lst)  # allocate node types randomly
        # self.Nc_nodes = [i for i, l in enumerate(lst) if l == 0]  # record communication nodes
        # self.Ns_nodes = [i for i, l in enumerate(lst) if l == 1]  # record sensor nodes
        self.Nc_nodes = [1, 2, 4, 5, 6, 9, 10, 11, 13, 14, 15, 17, 18, 19]
        self.Ns_nodes = [0, 3, 7, 8, 12, 16]
        self.s_dim = self.MM.F.shape[0]  # record the dimension of state vector
        self.y_dim = self.MM.H.shape[0]  # record the dimension of observation vector
        self.gamma = 0.2
        # delete redundant edges
        redundant_edges = []
        for u, v in self.G.edges:
            if u in self.Nc_nodes:
                redundant_edges.append((u, v))
        self.G.remove_edges_from(redundant_edges)
        # guarantee each node is connected to 3 guider node
        for i in self.Nc_nodes:
            neis = set(self.G.predecessors(i))
            diff_set = set(self.Ns_nodes) - neis
            Ns_neis_count = len(neis.intersection(set(self.Ns_nodes)))
            if Ns_neis_count < 3:
                [k1, k2, k3] = random.sample(diff_set, 3)
                match Ns_neis_count:
                    case 0:
                        self.G.add_edge(k1, i)
                        self.G.add_edge(k2, i)
                        self.G.add_edge(k3, i)
                    case 1:
                        self.G.add_edge(k1, i)
                        self.G.add_edge(k2, i)
                    case 2:
                        self.G.add_edge(k1, i)
        # give node attribute value
        for i in self.G.nodes():
            self.G.nodes[i]["NodeType"]: int = int(i in self.Ns_nodes)  # 0-Follower Satellite, 1-Guider Satellite
            self.G.nodes[i]["S_rea"]: np.ndarray = Sr[i]
            self.G.nodes[i]["P_est"]: np.ndarray = P0
            self.G.nodes[i]["S_est"]: np.ndarray = np.random.multivariate_normal(mean=Sr[i], cov=P0)
            self.G.nodes[i]["InfoMat"]: np.ndarray = inv(P0)
            self.G.nodes[i]["InfoVec"]: np.ndarray = self.G.nodes[i]["InfoMat"] @ self.G.nodes[i]["S_est"]
            self.G.nodes[i]["InfoMat_fu"]: np.ndarray = inv(P0)
            self.G.nodes[i]["InfoVec_fu"]: np.ndarray = inv(P0) @ self.G.nodes[i]["S_est"]
            self.G.nodes[i]["InfoMat_bar"]: np.ndarray = inv(MM.F @ P0 @ MM.F.T + MM.Q)
            self.G.nodes[i]["InfoVec_bar"]: np.ndarray = self.G.nodes[i]["InfoMat_bar"] @ (
                    MM.F @ self.G.nodes[i]["S_est"])
            self.G.nodes[i]["D"]: List[Tuple[np.ndarray, float]] = []
            self.G.nodes[i]["C1"] = 1  # communication signals

        for i in self.Ns_nodes:
            neis = list(self.G.successors(i))
            self.G.nodes[i]["C2"] = {j: 0 for j in neis}
            self.G.nodes[i]["InfoMat_nei"] = {j: self.G.nodes[j]["InfoMat"] for j in neis}
            self.G.nodes[i]["InfoVec_nei"] = {j: self.G.nodes[j]["InfoVec"] for j in neis}
            self.G.nodes[i]["D_nei"] = {j: [] for j in neis}
            self.G.nodes[i]["D_bar"] = {j: [] for j in neis}
            self.G.nodes[i]["s_sat"] = {j: (Sr[j],) for j in self.G.nodes()}
            self.G.nodes[i]["s_obs"] = np.zeros((No, 7))

        self.AttriList = [
            "NodeType",
            "S_rea", "P_est", "S_est",
            "InfoMat", "InfoVec", "InfoMat_bar", "InfoVec_bar", "InfoMat_fu", "InfoVec_fu",
            "InfoMat_nei", "InfoVec_nei", "ObstPos_nei", "y_sat", "y_obs", "C1", "C2"
        ]
        self.Recorder = {
            "SS": self.Ns_nodes,
            "CS": self.Nc_nodes,
            "S_rea": [Sr],
            "S_est": [np.array([self.G.nodes[ni]["S_est"] for ni in self.G.nodes()])],
            "C1": [[self.G.nodes[ni]["C1"] for ni in self.Ns_nodes]],
            "C2": {ni: {nj: [] for nj in self.G.successors(ni)} for ni in self.Ns_nodes},
        }

    def collision_avoidance_update(self, ni, p_ca):
        """
        p_ca: collision avoidance parameter p_ca = (y_obs, rho_obs)
        """
        Di = self.G.nodes[ni]["D"]
        if not Di:
            Di.append(p_ca)
        else:
            dp_list = [dp(p_ca, pm) for pm in Di]
            if min(dp_list) > self.gamma:
                Di.append(p_ca)
            else:
                min_index = dp_list.index(min(dp_list))
                Di[min_index] = p_ca
        return

    def obtain_Dji(self, i, yji):
        kappa, rs = 3.0, 0.1
        Di = self.G.nodes[i]["D"]
        Dji = [(ym, rm) for ym, rm in Di if rs < norm(ym - yji) < kappa * rm]
        return Dji

    # determine whether node ni requires to communicate
    def state_estimation_trigger(self, ni, **kwargs):
        a = kwargs["a"]
        b = kwargs["b"]
        O_kk = self.G.nodes[ni]["InfoMat"]
        O_bar = self.G.nodes[ni]["InfoMat_bar"]
        s_kk = self.G.nodes[ni]["S_est"]
        s_bar = inv(O_bar) @ self.G.nodes[ni]["InfoVec_bar"]
        if (norm_MatrixInduced(s_kk - s_bar, O_kk) <= a and
                is_semi_positive(O_bar - O_kk / (1 + b)) and is_semi_positive(O_kk * (1 + b) - O_bar)):
            c1 = 0
        else:
            c1 = 1
        self.G.nodes[ni]["C1"] = c1

    def collision_detect_trigger(self, i: int, j: int):
        Dji = self.G.nodes[i]["D_nei"][j]
        Dji_bar = self.G.nodes[i]["D_bar"][j]
        if not (Dji or Dji_bar):
            return 0
        if len(Dji) > len(Dji_bar):
            return 1
        max_min_d = -np.inf
        for pm in Dji:
            min_d = min([dp(pm, pm_bar) for pm_bar in Dji_bar])
            if min_d >= max_min_d:
                max_min_d = min_d
        return int(max_min_d >= self.gamma)

    def obstacle_fusion(self, ni):
        D_family = [self.G.nodes[vj]["D_nei"][ni] for vj in self.G.predecessors(ni) if self.G.nodes[vj]["C2"][ni]]
        if ni in self.Ns_nodes:
            D_family.append(self.G.nodes[ni]["D"])
        DiF = clustered_mean_Union(*D_family, g=self.gamma)
        return DiF

    def plot_Graph(self, p=True):
        if p:
            self.__plot_3d_network()
        else:
            fig1: plt.Figure = plt.figure(1, figsize=(10, 10))
            ax1: plt.Axes = fig1.add_axes((0.15, 0.1, 0.8, 0.8))
            pos = {ni: self.G.nodes[ni]['S_rea'][:2] for ni in self.G.nodes()}
            nx.draw_networkx_nodes(self.G, pos, self.Ns_nodes, ax=ax1,
                                   node_color="red", node_shape="o", node_size=200, label="Guider satellite")
            nx.draw_networkx_nodes(self.G, pos, self.Nc_nodes, ax=ax1,
                                   node_color="blue", node_shape="s", node_size=200, label="Follower satellite")
            nx.draw_networkx_edges(self.G, pos, ax=ax1, width=2.0)
            ax1.axis('on')
            ax1.tick_params(axis="both", which="both", bottom=True, left=True, labelbottom=True, labelleft=True,
                            labelsize=20)
            ax1.legend(fontsize=12)
            ax1.set_xlabel("x(km)", fontsize=25)
            ax1.set_ylabel("y(km)", fontsize=25)
            plt.show()

    def __plot_3d_network(self):
        p = nx.get_node_attributes(self.G, "S_rea")
        pc = np.array(list(p.values()))[self.Nc_nodes, :3]
        ps = np.array(list(p.values()))[self.Ns_nodes, :3]
        f1: plt.Figure = plt.figure(1, figsize=(10, 10))
        ax: Axes3D = f1.add_axes((0.05, 0.05, 0.8, 0.8), projection="3d")
        ax.set_box_aspect([1, 1, 0.5])

        # 节点
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], alpha=.9, c="slateblue", s=100, marker="o", label="follower satellites")
        ax.scatter(ps[:, 0], ps[:, 1], ps[:, 2], alpha=.9, c="red", s=100, marker="s", label="leader satellites")

        xs, ys, zs = [], [], []
        for (i, j) in self.G.edges():
            xs.append(p[i][0])
            xs.append(p[j][0] - 0.1)
            xs.append(np.nan)
            ys.append(p[i][1])
            ys.append(p[j][1] - 0.1)
            ys.append(np.nan)
            zs.append(p[i][2])
            zs.append(p[j][2] - 0.1)
            zs.append(np.nan)
            plot_cone(ax, p[i][:3], p[j][:3] - 0.1, 1.5, radius=0.15, color="green")
            pass
        ax.plot(xs, ys, zs, linewidth=1.5, color="green", alpha=1.0, label="Communication Links")

        ax.legend(fontsize=15)
        ax.set_xlabel("x(km)", fontsize=15)
        ax.set_ylabel("y(km)", fontsize=15)
        ax.set_zlabel("z(km)", fontsize=15)
        ax.tick_params(axis="both", labelsize=14)
        plt.show()
        pass
