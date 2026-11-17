import numpy as np
from typing import List, Dict, Union, Tuple, Iterable
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
from mpl_toolkits.mplot3d import Axes3D
import json
import re

Re = 6371
color_map = ["rosybrown", "lightcoral", "darkred", "coral", "peru", "tan",
             "orange", "darkkhaki", "g", "teal", "b", "crimson"]

r2d = lambda x: 180 * x / np.pi


def plot_field_surface(X, Y, Z):
    Z = np.clip(Z, -0.003, 0.001)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    f: plt.Figure = plt.figure(figsize=(6, 6))
    ax: Axes3D = f.add_axes((0.1, 0.1, 0.8, 0.8), projection="3d")
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(-15, 15)
    ax.set_ylim3d(-15, 15)
    ax.set_box_aspect([1, 1, 0.5])
    plt.show()


def create_earth(x, y, R):
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i, j] ** 2 + Y[i, j] ** 2 >= R ** 2:
                Z[i, j] = np.nan
            else:
                Z[i, j] = np.sqrt(R ** 2 - (X[i, j] ** 2 + Y[i, j] ** 2))
    return Z


def plot_control_signal(u_arr: np.ndarray, umax=3e-5, t_sam=60):
    n = u_arr.shape[0]
    t_seq = [t_sam * i for i in range(n)]
    f: plt.Figure = plt.figure(figsize=(10, 5))
    ax: plt.Axes = f.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.plot(t_seq, u_arr[:, 0], label="ux", c="b")
    ax.plot(t_seq, u_arr[:, 1], label="uy", c="y")
    ax.plot(t_seq, u_arr[:, 2], label="uz", c="g")
    ax.set_xlim([0, n * t_sam])
    ax.set_ylim([-umax * 1.1, umax * 1.1])
    ax.set_xlabel("t(s)")
    ax.set_ylabel(r"u($km/s^2$)")
    ax.legend()
    plt.show()
    pass


def plot_track_ECI(cs: np.ndarray, fs: np.ndarray):
    n = fs.shape[1]
    f: plt.Figure = plt.figure(figsize=(10, 8))
    ax: Axes3D = f.add_axes((0, 0, 1, 1), projection="3d")
    ax.set_box_aspect([1, 1, 1])
    # plot the Earth
    phi, theta = np.mgrid[0.0:2.0 * np.pi:50j, 0.0:np.pi:25j]
    xe = Re * np.sin(theta) * np.cos(phi)
    ye = Re * np.sin(theta) * np.sin(phi)
    ze = Re * np.cos(theta)
    ax.plot_surface(xe, ye, ze, color='slateblue', alpha=0.6, edgecolors='khaki')
    # plot satellite track
    ax.plot3D(cs[:, 0], cs[:, 1], cs[:, 2], c="r", label="chief satellite")
    for i in range(n):
        si = fs[:, i, :]
        ax.plot3D(si[:, 0], si[:, 1], si[:, 2],
                  linestyle="--",
                  linewidth=0.8,
                  label=f"{i + 1}-th follow satellite")
    ax.legend()
    lim = np.linalg.norm(cs[0, :3]) + 1000
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(-lim, lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


class satellite_cluster_visualization:
    def __init__(self, data):
        self.data: Dict[str, list | np.ndarray | Dict[int, list]] = data
        self.fig1: plt.Figure = plt.figure(1, figsize=(10, 8))
        self.ax1: Axes3D = self.fig1.add_axes((0.1, 0.1, 0.8, 0.8), projection="3d")
        self.ax1.set_xlim3d(-10, 10)
        self.ax1.set_ylim3d(-10, 10)
        self.ax1.set_zlim3d(-5, 5)
        self.ax1.set_xlabel("x(km)", fontsize=15)
        self.ax1.set_ylabel("y(km)", fontsize=15)
        self.ax1.set_zlabel("z(km)", fontsize=15)
        self.ax1.tick_params("both", labelsize=15)
        self.ax1.set_box_aspect([1, 1, 0.5])

    def show_obstacle(self):
        pos = self.data["O_pos"]
        rad = self.data["O_rad"]
        p0 = pos[:, 0]
        pf = pos[:, -1]
        obs_num = rad.size
        phi, theta = np.mgrid[0.0:2.0 * np.pi:10j, 0.0:np.pi:5j]
        for i in range(obs_num):
            x0, xf = rad[i] * np.sin(theta) * np.cos(phi) + p0[i, 0], rad[i] * np.sin(theta) * np.cos(phi) + pf[i, 0]
            y0, yf = rad[i] * np.sin(theta) * np.sin(phi) + p0[i, 1], rad[i] * np.sin(theta) * np.sin(phi) + pf[i, 1]
            z0, zf = rad[i] * np.cos(theta) + p0[i, 2], rad[i] * np.cos(theta) + pf[i, 2]
            self.ax1.plot_surface(x0, y0, z0, color='slateblue', alpha=0.2, edgecolors='khaki', linestyles="--")
            self.ax1.plot_surface(xf, yf, zf, color='slateblue', alpha=0.5, edgecolors='khaki')
            self.ax1.plot(pos[i, :, 0], pos[i, :, 1], pos[i, :, 2], linestyle="--", c="b")
            self.draw_cylinder(pos[i, [i for i in range(0, pos.shape[1], 20)]], rad[i], self.ax1)
        # plt.show()
        return 0

    def plot_fsat_track(self, nodes, show_fig=True, **kwargs):
        s = np.array(self.data["S_est"])
        t = self.data["S_tar"]
        al = kwargs["alpha"] if "alpha" in kwargs.keys() else 1.0  # set transparency
        ls = kwargs["linestyle"] if "linestyle" in kwargs.keys() else "-"  # set line style
        lw = kwargs["linewidth"] if "linewidth" in kwargs.keys() else 1.0  # set line width
        lb = kwargs["label"] if "label" in kwargs.keys() else "final trajectory"  # set line label
        label_flag = 0
        for i in nodes:
            si = s[:, i, :]
            ti = t[i]
            # self.ax1.text(si[0, 0], si[0, 1], si[0, 2], f"{i}")
            if label_flag == 0:
                label_flag = 1
                self.ax1.scatter3D(si[0, 0], si[0, 1], si[0, 2],  # scatter initial position and label
                                   marker="d", color=color_map[i], label="initial pos")
                self.ax1.scatter3D(ti[0], ti[1], ti[2],  # scatter target position and label
                                   marker="o", color=color_map[i], label="target pos")
                self.ax1.plot3D(si[:, 0], si[:, 1], si[:, 2],  # plot trajectory
                                c=color_map[i], linewidth=lw, linestyle=ls, alpha=al, label=lb)
            else:
                self.ax1.scatter3D(si[0, 0], si[0, 1], si[0, 2], marker="d", color=color_map[i % len(color_map)])
                self.ax1.scatter3D(ti[0], ti[1], ti[2], marker="o", color=color_map[i % len(color_map)])
                self.ax1.plot3D(si[:, 0], si[:, 1], si[:, 2], c=color_map[i % len(color_map)], linewidth=lw,
                                linestyle=ls, alpha=al)
        if show_fig:
            self.ax1.scatter3D(0, 0, 0, marker="*", label="chief sat", color="r")
            hs, las = self.ax1.get_legend_handles_labels()
            las_uni = list(set(las))
            hs_uni = [hs[las.index(la)] for la in las_uni]
            self.ax1.legend(hs_uni, las_uni, fontsize=15)
            plt.figure(1)
            plt.show()

    def draw_xyz(self, ni, dt=10.0):
        sr = np.array(self.data["S_rea"])[:, ni]
        se = np.array(self.data["S_est"])[:, ni]
        ae = [r2d(np.arccos(np.dot(sr[k, :3], se[k, :3]) / (np.linalg.norm(sr[k, :3]) * np.linalg.norm(se[k, :3]))))
              for k in range(sr.shape[0])]
        t = [dt * i for i in range(sr.shape[0])]
        f1: plt.Figure = plt.figure(2, figsize=(16, 7))
        ax1: plt.Axes = f1.add_subplot(241)
        ax2: plt.Axes = f1.add_subplot(242)
        ax3: plt.Axes = f1.add_subplot(243)
        ax4: plt.Axes = f1.add_subplot(245)
        ax5: plt.Axes = f1.add_subplot(246)
        ax6: plt.Axes = f1.add_subplot(247)
        ax7: plt.Axes = f1.add_subplot(248)
        ax1.plot(t, sr[:, 0], label="real track")
        ax1.plot(t, se[:, 0], label="DKF estimated track")
        ax1.set_xlabel("time step")
        ax1.set_ylabel("x(km)")
        ax2.plot(t, sr[:, 1], label="real track")
        ax2.plot(t, se[:, 1], label="DKF estimated track")
        ax2.set_xlabel("time step")
        ax2.set_ylabel("y(km)")
        ax3.plot(t, sr[:, 2], label="real track")
        ax3.plot(t, se[:, 2], label="DKF estimated track")
        ax3.set_xlabel("time step")
        ax3.set_ylabel("z(km)")
        ax3.legend()
        ax4.plot(t, sr[:, 3], label="real track")
        ax4.plot(t, se[:, 3], label="DKF estimated track")
        ax4.set_xlabel("time step")
        ax4.set_ylabel("vx(km/s)")
        ax5.plot(t, sr[:, 4], label="real track")
        ax5.plot(t, se[:, 4], label="DKF estimated track")
        ax5.set_xlabel("time step")
        ax5.set_ylabel("vy(km/s)")
        ax6.plot(t, sr[:, 5], label="real track")
        ax6.plot(t, se[:, 5], label="DKF estimated track")
        ax6.set_xlabel("time step")
        ax6.set_ylabel("vz(km/s)")
        ax7.plot(t, ae)
        ax7.set_xlabel("time step")
        ax7.set_ylabel(r"<$v_r$, $v_e$>")
        plt.figure(2)
        plt.show()

    def draw_signal_trigger1(self, ni, dt=10.0):
        x = list(zip(*self.data["C1"]))[ni]
        t = [dt * i for i in range(len(x))]
        f3: plt.Figure = plt.figure(3, figsize=(10, 5))
        ax1: plt.Axes = f3.add_axes((0.08, 0.1, 0.84, 0.8))
        xy_pair_0 = [(ti, ci) for ti, ci in zip(t, x) if ci == 0]
        xy_pair_1 = [(ti, ci) for ti, ci in zip(t, x) if ci == 1]
        if xy_pair_0:
            ax1.scatter(*tuple(zip(*xy_pair_0)), marker="^", color="blue", label=f"No communication: {len(xy_pair_0)}")
        if xy_pair_1:
            ax1.scatter(*tuple(zip(*xy_pair_1)), marker="^", color="red", label=f"Communication: {len(xy_pair_1)}")
        ax1.set_yticks([0, 1])
        ax1.tick_params(labelsize=12)
        ax1.set_xlabel("t(s)", fontsize=14)
        ax1.set_ylabel("State Estimation Trigger Signal", fontsize=14)
        ax1.legend(fontsize=14, loc="upper right", bbox_to_anchor=(1.0, 0.9))
        plt.figure(3)
        plt.show()

    def draw_signal_trigger2(self, ni, dt=1.0):
        if ni not in self.data["SS"]:
            return
        c_dict = self.data["C2"][ni]
        nei_num = len(c_dict)
        if nei_num:
            t_seq = [dt * i for i in range(len(list(c_dict.values())[0]))]
            f4: plt.Figure = plt.figure(4, figsize=(10, 5 * nei_num))
            for i, (nei, c_seq) in enumerate(c_dict.items()):
                xy_pair_0 = [(ti, ci) for ti, ci in zip(t_seq, c_seq) if ci == 0]
                xy_pair_1 = [(ti, ci) for ti, ci in zip(t_seq, c_seq) if ci == 1]
                axi: plt.Axes = f4.add_subplot(nei_num, 1, i + 1)
                if xy_pair_0:
                    axi.scatter(*tuple(zip(*xy_pair_0)), marker="^", color="blue",
                                label=f"No communication:{len(xy_pair_0)}")
                if xy_pair_1:
                    axi.scatter(*tuple(zip(*xy_pair_1)), marker="^", color="red",
                                label=f"Communication:{len(xy_pair_1)}")
                axi.set_yticks([0, 1])
                axi.tick_params(labelsize=12)
                axi.set_title(rf"Satellite {ni} $\rightarrow$ Satellite {nei}")
                axi.set_xlabel("t(s)", fontsize=14)
                axi.set_ylabel("Collision Detection Trigger Signal", fontsize=14)
                axi.legend(fontsize=14, loc="upper right", bbox_to_anchor=(1.0, 0.9))
            plt.figure(4)
            plt.show()

    @staticmethod
    def draw_cylinder(x, r, ax):
        N = x.shape[0]
        theta = np.linspace(0, 2 * np.pi, 10)
        circle_x = r * np.cos(theta)
        circle_y = r * np.sin(theta)
        surface = []
        for i in range(N - 1):
            p1 = x[i]
            p2 = x[i + 1]
            direction = p2 - p1
            direction /= np.linalg.norm(direction)
            up = np.array([0, 0, 1])
            if np.all(direction == up):
                up = np.array([0, 1, 0])
            v1 = np.cross(direction, up)
            v1 /= np.linalg.norm(v1)
            v2 = np.cross(direction, v1)
            circle = v1.reshape((3, 1)) @ circle_x[None, :] + v2.reshape((3, 1)) @ circle_y[None, :]
            ax.plot3D(p1[0] + circle[0], p1[1] + circle[1], p1[2] + circle[2], color='lightblue', alpha=0.2)
            ax.plot3D(p2[0] + circle[0], p2[1] + circle[1], p2[2] + circle[2], color='lightblue', alpha=0.2)
            surface.append(np.array([p1[0] + circle[0], p1[1] + circle[1], p1[2] + circle[2]]))
            if i == N - 2:
                surface.append(np.array([p2[0] + circle[0], p2[1] + circle[1], p2[2] + circle[2]]))
        surface = np.array(surface)
        ax.plot_surface(surface[:, 0], surface[:, 1], surface[:, 2], color="b", alpha=0.2)
        return 0

    def obstacle_collision_avoidance_test(self):
        from numpy.linalg import norm
        x = np.array(self.data["S_est"])
        obs = self.data["O_pos"]
        rho = self.data["O_rad"]
        obs_count = obs.shape[0]
        n = x.shape[1]
        N = x.shape[0]
        obs_co = [[] for _ in range(obs_count)]
        for p in range(n):
            xnp = x[:, p, :3]
            for i in range(obs_count):
                oi = obs[i]
                d_min = min([norm(xnp[k] - oi[k]) for k in range(N)])
                d_min = d_min if d_min > rho[i] else d_min * 1.8
                obs_co[i].append(d_min)
        f2: plt.Figure = plt.figure(22, figsize=(12, 6))
        for i in range(1, obs_count + 1):
            axi: plt.Axes = f2.add_subplot(2, int(obs_count / 2), i)
            axi.scatter(list(range(1, n + 1)), obs_co[i - 1], label=f"minimum distance for obstacle {i}")
            axi.hlines(y=rho[i - 1], xmin=0, xmax=n + 1, colors="r", label="safe distance")
            axi.set_xlim(0, n + 1), axi.set_xlabel("Satellite No."), axi.set_ylabel("distance(km)")
            axi.legend(loc="upper right", fontsize=8)
            axi.set_xticks(list(range(0, n + 1, 2)))
        plt.figure(22)
        plt.show()

    def satellite_collision_avoidance_test(self, rs):
        from numpy.linalg import norm
        x = np.array(self.data["S_est"])
        n = x.shape[1]
        N = x.shape[0]
        sat_co = []
        for p in range(n):
            xnp = x[:, p, :3]
            nei1 = x[:, p - 1, :3]
            nei2 = x[:, (p + 1) % n, :3]
            sat_co.append(min([min([norm(xnp[k] - nei1[k]) for k in range(N)]),
                               min([norm(xnp[k] - nei2[k]) for k in range(N)])]))
        f1: plt.Figure = plt.figure(11)
        ax1: plt.Axes = f1.add_subplot(111)
        ax1.scatter(list(range(1, n + 1)), sat_co, label="minimum distance")
        ax1.hlines(rs, xmin=0, xmax=n + 1, colors="r", label="safe distance")
        ax1.set_xlim(0, n + 1)
        ax1.legend(loc="upper right", fontsize=8)
        ax1.set_xlabel("Satellite No."), ax1.set_ylabel("distance(km)")
        ax1.set_xticks(list(range(0, n + 1, 2)))
        plt.figure(11)
        plt.show()


class RecorderPlot:
    def __init__(self, filename):
        with open(filename, "r") as f:
            recorder = json.load(f)
        self.recorder = recorder
        self.n = len(recorder)  # the number of mento-carlo sims
        self.guider_nodes = [0, 3, 7, 8, 12, 16]
        self.follower_nodes = [1, 2, 4, 5, 6, 9, 10, 11, 13, 14, 15, 17, 18, 19]
        m = re.search(r'alpha=([-+]?\d*\.?\d+)', filename)
        self.alpha = m.group(1) if m else None
        m = re.search(r'beta=([-+]?\d*\.?\d+)', filename)
        self.beta = m.group(1) if m else None

    def draw_RMSE(self, t_sam=10.0, fig_name=None):
        x_rea_list: List[np.ndarray] = list(map(lambda ri: np.array(ri["S_rea"]), self.recorder))
        x_est_list: List[np.ndarray] = list(map(lambda ri: np.array(ri["S_est"]), self.recorder))
        rmse = np.sqrt(np.sum([(xr - xe) ** 2 / self.n for xr, xe in zip(x_rea_list, x_est_list)], axis=0))
        rmse_guider = np.average(rmse[:, self.guider_nodes, :], axis=1)
        rmse_follower = np.average(rmse[:, self.follower_nodes, :], axis=1)
        t_seq = [t_sam * i for i in range(x_rea_list[0].shape[0])]

        ls = "-"
        lw = 1.5

        plt.rcParams['xtick.labelsize'] = 13
        plt.rcParams['ytick.labelsize'] = 13

        f1: plt.Figure = plt.figure(1, figsize=(16, 7), constrained_layout=True)
        ax_px: plt.Axes = f1.add_subplot(2, 3, 1)
        ax_py: plt.Axes = f1.add_subplot(2, 3, 2)
        ax_pz: plt.Axes = f1.add_subplot(2, 3, 3)
        ax_vx: plt.Axes = f1.add_subplot(2, 3, 4)
        ax_vy: plt.Axes = f1.add_subplot(2, 3, 5)
        ax_vz: plt.Axes = f1.add_subplot(2, 3, 6)
        axin_px = ax_px.inset_axes((0.3, 0.3, 0.63, 0.25))
        axin_py = ax_py.inset_axes((0.3, 0.3, 0.63, 0.25))
        axin_pz = ax_pz.inset_axes((0.3, 0.3, 0.63, 0.25))
        axin_vx = ax_vx.inset_axes((0.3, 0.3, 0.63, 0.25))
        axin_vy = ax_vy.inset_axes((0.3, 0.3, 0.63, 0.25))
        axin_vz = ax_vz.inset_axes((0.3, 0.3, 0.63, 0.25))

        ax_px.plot(t_seq, rmse_guider[:, 0] * 1000, lw=lw, ls=ls)
        ax_py.plot(t_seq, rmse_guider[:, 1] * 1000, lw=lw, ls=ls)
        ax_pz.plot(t_seq, rmse_guider[:, 2] * 1000, lw=lw, ls=ls, label="Leader Satellites")
        ax_vx.plot(t_seq, rmse_guider[:, 3] * 1000, lw=lw, ls=ls)
        ax_vy.plot(t_seq, rmse_guider[:, 4] * 1000, lw=lw, ls=ls)
        ax_vz.plot(t_seq, rmse_guider[:, 5] * 1000, lw=lw, ls=ls)

        ax_px.plot(t_seq, rmse_follower[:, 0] * 1000, lw=lw, ls=ls)
        ax_py.plot(t_seq, rmse_follower[:, 1] * 1000, lw=lw, ls=ls)
        ax_pz.plot(t_seq, rmse_follower[:, 2] * 1000, lw=lw, ls=ls, label="Follower Satellites")
        ax_vx.plot(t_seq, rmse_follower[:, 3] * 1000, lw=lw, ls=ls)
        ax_vy.plot(t_seq, rmse_follower[:, 4] * 1000, lw=lw, ls=ls)
        ax_vz.plot(t_seq, rmse_follower[:, 5] * 1000, lw=lw, ls=ls)

        axin_px.plot(t_seq[600:], rmse_guider[600:, 0] * 1000, lw=lw + 0.5, ls=ls)
        axin_py.plot(t_seq[600:], rmse_guider[600:, 1] * 1000, lw=lw + 0.5, ls=ls)
        axin_pz.plot(t_seq[600:], rmse_guider[600:, 2] * 1000, lw=lw + 0.5, ls=ls)
        axin_vx.plot(t_seq[600:], rmse_guider[600:, 3] * 1000, lw=lw + 0.5, ls=ls)
        axin_vy.plot(t_seq[600:], rmse_guider[600:, 4] * 1000, lw=lw + 0.5, ls=ls)
        axin_vz.plot(t_seq[600:], rmse_guider[600:, 5] * 1000, lw=lw + 0.5, ls=ls)

        axin_px.plot(t_seq[600:], rmse_follower[600:, 0] * 1000, lw=lw + 0.5, ls=ls)
        axin_py.plot(t_seq[600:], rmse_follower[600:, 1] * 1000, lw=lw + 0.5, ls=ls)
        axin_pz.plot(t_seq[600:], rmse_follower[600:, 2] * 1000, lw=lw + 0.5, ls=ls)
        axin_vx.plot(t_seq[600:], rmse_follower[600:, 3] * 1000, lw=lw + 0.5, ls=ls)
        axin_vy.plot(t_seq[600:], rmse_follower[600:, 4] * 1000, lw=lw + 0.5, ls=ls)
        axin_vz.plot(t_seq[600:], rmse_follower[600:, 5] * 1000, lw=lw + 0.5, ls=ls)

        ax_px.set_ylabel(r"$RMSE_{x}$(m)", fontsize=14), ax_px.set_xlim(-300, t_seq[-1] - 300)
        ax_py.set_ylabel(r"$RMSE_{y}$(m)", fontsize=14), ax_py.set_xlim(-300, t_seq[-1] - 300)
        ax_pz.set_ylabel(r"$RMSE_{z}$(m)", fontsize=14), ax_pz.set_xlim(-300, t_seq[-1] - 300)
        ax_vx.set_ylabel(r"$RMSE_{vx}$(m/s)", fontsize=14), ax_vx.set_xlim(-300, t_seq[-1] - 300)
        ax_vy.set_ylabel(r"$RMSE_{vy}$(m/s)", fontsize=14), ax_vy.set_xlim(-300, t_seq[-1] - 300)
        ax_vz.set_ylabel(r"$RMSE_{vz}$(m/s)", fontsize=14), ax_vz.set_xlim(-300, t_seq[-1] - 300)

        axin_px.set_xlim(t_seq[600], t_seq[-30]), axin_px.set_ylim(0, 10), axin_px.set_yticks([0, 3, 6, 9])
        axin_py.set_xlim(t_seq[600], t_seq[-30]), axin_py.set_ylim(0, 10), axin_py.set_yticks([0, 3, 6, 9])
        axin_pz.set_xlim(t_seq[600], t_seq[-30]), axin_pz.set_ylim(0, 10), axin_pz.set_yticks([0, 3, 6, 9])
        axin_vx.set_xlim(t_seq[600], t_seq[-30]), axin_vx.set_ylim(0, 0.15), axin_vx.set_yticks([0, 0.05, 0.10, 0.15])
        axin_vy.set_xlim(t_seq[600], t_seq[-30]), axin_vy.set_ylim(0, 0.08), axin_vy.set_yticks([0, 0.03, 0.06])
        axin_vz.set_xlim(t_seq[600], t_seq[-30]), axin_vz.set_ylim(0, 0.10), axin_vz.set_yticks([0, 0.03, 0.06, 0.09])

        ax_vx.set_xlabel("time(s)", fontsize=14)
        ax_vy.set_xlabel("time(s)", fontsize=14)
        ax_vz.set_xlabel("time(s)", fontsize=14)
        ax_pz.legend(loc="upper right", fontsize=14)

        if fig_name is None:
            plt.show()
        else:
            plt.savefig(fig_name)
        pass


class MultiRecorderPlot:

    def __init__(self, *filenames):
        self.fns = filenames
        self.alphas = []
        self.betas = []
        self.rcs: Dict[Tuple, List[Dict[str, Union[List[np.ndarray], Dict[str, Dict[str, list]]]]]] = {}
        for fn in filenames:
            m = re.search(r'alpha=([-+]?\d*\.?\d+)', fn)
            alpha = int(m.group(1)) if m else None
            m = re.search(r'beta=([-+]?\d*\.?\d+)', fn)
            beta = int(m.group(1)) if m else None
            self.alphas.append(alpha), self.betas.append(beta)
            with open(fn, "r") as f:
                ri: List[Dict[str, Union[List[np.ndarray], Dict[int, Dict[int, list]]]]] = json.load(f)
            self.rcs[(alpha, beta)] = ri
        self.guider_nodes = [0, 3, 7, 8, 12, 16]
        self.follower_nodes = [1, 2, 4, 5, 6, 9, 10, 11, 13, 14, 15, 17, 18, 19]
        self.guider_numbers = len(self.guider_nodes)

    def draw_multi_RMSE(self, nodes, dt=10.0, fig_name=None):

        ls = "-"
        lw = 1.5

        m = ["o", "s", "d", "^", "v", "*", "X", "+", "<"]

        plt.rcParams['xtick.labelsize'] = 13
        plt.rcParams['ytick.labelsize'] = 13

        f1: plt.Figure = plt.figure(1, figsize=(16, 7), constrained_layout=True)
        ax_px: plt.Axes = f1.add_subplot(2, 3, 1)
        ax_py: plt.Axes = f1.add_subplot(2, 3, 2)
        ax_pz: plt.Axes = f1.add_subplot(2, 3, 3)
        ax_vx: plt.Axes = f1.add_subplot(2, 3, 4)
        ax_vy: plt.Axes = f1.add_subplot(2, 3, 5)
        ax_vz: plt.Axes = f1.add_subplot(2, 3, 6)
        axin_px = ax_px.inset_axes((0.3, 0.2, 0.63, 0.25))
        axin_py = ax_py.inset_axes((0.3, 0.2, 0.63, 0.25))
        axin_pz = ax_pz.inset_axes((0.3, 0.2, 0.63, 0.25))
        axin_vx = ax_vx.inset_axes((0.3, 0.2, 0.63, 0.25))
        axin_vy = ax_vy.inset_axes((0.3, 0.2, 0.63, 0.25))
        axin_vz = ax_vz.inset_axes((0.3, 0.2, 0.63, 0.25))

        i = 0
        t_seq = []
        for info, rc in self.rcs.items():
            alpha, beta = info
            r_bar = np.round(self.cal_average_c(rc) * 100, 2)
            label = rf"$\alpha$={alpha},$\beta$={beta},$\bar r$={r_bar}%"
            if alpha is None and beta is None:
                label = rf"periodic communication,$\bar r$={r_bar}%"
            x_rea_list: List[np.ndarray] = list(map(lambda ri: np.array(ri["S_rea"]), rc))
            x_est_list: List[np.ndarray] = list(map(lambda ri: np.array(ri["S_est"]), rc))
            rmse = np.sqrt(np.sum([(xr - xe) ** 2 / len(rc) for xr, xe in zip(x_rea_list, x_est_list)], axis=0))
            rmse = np.average(rmse[:, nodes, :], axis=1)

            t_seq = [dt * i for i in range(x_rea_list[0].shape[0])]
            ax_px.plot(t_seq, self.smooth(rmse[:, 0] * 1000), lw=lw, marker=m[i], markevery=50, alpha=0.6, ls=ls)
            ax_py.plot(t_seq, self.smooth(rmse[:, 1] * 1000), lw=lw, marker=m[i], markevery=50, alpha=0.6, ls=ls)
            ax_pz.plot(t_seq, self.smooth(rmse[:, 2] * 1000), lw=lw, marker=m[i], markevery=50, alpha=0.6, ls=ls,
                       label=label)
            ax_vx.plot(t_seq, self.smooth(rmse[:, 3] * 1000), lw=lw, marker=m[i], markevery=50, alpha=0.6, ls=ls)
            ax_vy.plot(t_seq, self.smooth(rmse[:, 4] * 1000), lw=lw, marker=m[i], markevery=50, alpha=0.6, ls=ls)
            ax_vz.plot(t_seq, self.smooth(rmse[:, 5] * 1000), lw=lw, marker=m[i], markevery=50, alpha=0.6, ls=ls)

            axin_px.plot(t_seq[600:], self.smooth(rmse[:, 0] * 1000)[600:], lw=lw + 0.5, marker=m[i], ls=ls, alpha=0.6,
                         markevery=200)
            axin_py.plot(t_seq[600:], self.smooth(rmse[:, 1] * 1000)[600:], lw=lw + 0.5, marker=m[i], ls=ls, alpha=0.6,
                         markevery=200)
            axin_pz.plot(t_seq[600:], self.smooth(rmse[:, 2] * 1000)[600:], lw=lw + 0.5, marker=m[i], ls=ls, alpha=0.6,
                         markevery=200)
            axin_vx.plot(t_seq[600:], self.smooth(rmse[:, 3] * 1000)[600:], lw=lw + 0.5, marker=m[i], ls=ls, alpha=0.6,
                         markevery=200)
            axin_vy.plot(t_seq[600:], self.smooth(rmse[:, 4] * 1000)[600:], lw=lw + 0.5, marker=m[i], ls=ls, alpha=0.6,
                         markevery=200)
            axin_vz.plot(t_seq[600:], self.smooth(rmse[:, 5] * 1000)[600:], lw=lw + 0.5, marker=m[i], ls=ls, alpha=0.6,
                         markevery=200)
            i += 1

        ax_px.set_ylabel(r"$RMSE_{x}$(m)", fontsize=14), ax_px.set_xlim(-300, t_seq[-1] - 300)
        ax_py.set_ylabel(r"$RMSE_{y}$(m)", fontsize=14), ax_py.set_xlim(-300, t_seq[-1] - 300)
        ax_pz.set_ylabel(r"$RMSE_{z}$(m)", fontsize=14), ax_pz.set_xlim(-300, t_seq[-1] - 300)
        ax_vx.set_ylabel(r"$RMSE_{vx}$(m/s)", fontsize=14), ax_vx.set_xlim(-300, t_seq[-1] - 300)
        ax_vy.set_ylabel(r"$RMSE_{vy}$(m/s)", fontsize=14), ax_vy.set_xlim(-300, t_seq[-1] - 300)
        ax_vz.set_ylabel(r"$RMSE_{vz}$(m/s)", fontsize=14), ax_vz.set_xlim(-300, t_seq[-1] - 300)

        axin_px.set_xlim(t_seq[600], t_seq[-30]), axin_px.set_ylim(5, 7), axin_px.set_yticks([5, 6, 7])
        axin_py.set_xlim(t_seq[600], t_seq[-30]), axin_py.set_ylim(2, 4), axin_py.set_yticks([2, 3, 4])
        axin_pz.set_xlim(t_seq[600], t_seq[-30]), axin_pz.set_ylim(3, 5), axin_pz.set_yticks([3, 4, 5])
        axin_vx.set_xlim(t_seq[600], t_seq[-30]), axin_vx.set_ylim(0.08, 0.10), axin_vx.set_yticks([0.08, 0.09, 0.10])
        axin_vy.set_xlim(t_seq[600], t_seq[-30]), axin_vy.set_ylim(0.02, 0.06), axin_vy.set_yticks([0.02, 0.04, 0.06])
        axin_vz.set_xlim(t_seq[600], t_seq[-30]), axin_vz.set_ylim(0.04, 0.08), axin_vz.set_yticks([0.04, 0.06, 0.08])

        ax_vx.set_xlabel("time(s)", fontsize=14)
        ax_vy.set_xlabel("time(s)", fontsize=14)
        ax_vz.set_xlabel("time(s)", fontsize=14)
        ax_pz.legend(loc="upper right", fontsize=14)
        if fig_name is None:
            plt.show()
        else:
            plt.savefig(fig_name)
        pass

    def draw_periodic_RMSE(self, pd_data, et_data, fig_name=None):
        nodes = self.follower_nodes
        t_seq = [i for i in range(401)]
        x_rea_pd = list(map(lambda ri: np.average(np.array(ri["x_real"])[:, nodes], axis=1), pd_data))
        x_nav_pd = list(map(lambda ri: np.average(np.array(ri["x_nav"])[:, nodes], axis=1), pd_data))
        rmse_pd: List[np.ndarray] = [np.sqrt((xr - xe) ** 2) for xr, xe in zip(x_rea_pd, x_nav_pd)]
        rmse_pd = np.average(rmse_pd, axis=0)
        x_rea_et = list(map(lambda ri: np.average(np.array(ri["x_real"])[:, nodes], axis=1), et_data))
        x_nav_et = list(map(lambda ri: np.average(np.array(ri["x_nav"])[:, nodes,], axis=1), et_data))
        rmse_et: List[np.ndarray] = [np.sqrt((xr - xe) ** 2) for xr, xe in zip(x_rea_et, x_nav_et)]
        rmse_et = np.average(rmse_et, axis=0)

        ls = ["-", "--", "-.", ":"]

        f1: plt.Figure = plt.figure(1, figsize=(16, 9))
        ax_px: plt.Axes = f1.add_subplot(2, 3, 1)
        ax_py: plt.Axes = f1.add_subplot(2, 3, 4)
        ax_vx: plt.Axes = f1.add_subplot(2, 3, 2)
        ax_vy: plt.Axes = f1.add_subplot(2, 3, 5)
        ax_ax: plt.Axes = f1.add_subplot(2, 3, 3)
        ax_ay: plt.Axes = f1.add_subplot(2, 3, 6)

        r_bar = np.round(self.cal_average_c(et_data) * 100, 2)
        ax_px.plot(t_seq, self.smooth(rmse_et[:, 0]), lw=1.5, ls=ls[0])
        ax_py.plot(t_seq, self.smooth(rmse_et[:, 1]), lw=1.5, ls=ls[0])
        ax_vx.plot(t_seq, self.smooth(rmse_et[:, 2], 5), lw=1.5, ls=ls[0])
        ax_vy.plot(t_seq, self.smooth(rmse_et[:, 3], 5), lw=1.5, ls=ls[0])
        ax_ax.plot(t_seq, self.smooth(rmse_et[:, 4], 10), lw=1.5, ls=ls[0],
                   label=rf"$\alpha$=1.0e-05,$\theta$=9.0,$\bar r$={r_bar}%")
        ax_ay.plot(t_seq, self.smooth(rmse_et[:, 5], 10), lw=1.5, ls=ls[0])

        r_bar = np.round(self.cal_average_c(pd_data) * 100, 2)
        ax_px.plot(t_seq, self.smooth(rmse_pd[:, 0], mode=1), lw=1.5, ls=ls[0])
        ax_py.plot(t_seq, self.smooth(rmse_pd[:, 1], mode=1), lw=1.5, ls=ls[0])
        ax_vx.plot(t_seq, self.smooth(rmse_pd[:, 2], mode=1), lw=1.5, ls=ls[0])
        ax_vy.plot(t_seq, self.smooth(rmse_pd[:, 3], mode=1), lw=1.5, ls=ls[0])
        ax_ax.plot(t_seq, self.smooth(rmse_pd[:, 4], 5, mode=1), lw=1.5, ls=ls[0],
                   label=rf"Periodic transmission,$\bar r$={r_bar}%")
        ax_ay.plot(t_seq, self.smooth(rmse_pd[:, 5], 5, mode=1), lw=1.5, ls=ls[0])

        # ax_px.set_xlabel("time(s)")
        ax_px.set_ylabel(r"$RMSE_{x}$")
        ax_px.set_title("Position(x)")
        ax_px.vlines(x=100, ymin=-0.1, ymax=4.0, ls="--", lw=1.0, colors="k", alpha=0.6)
        ax_px.vlines(x=250, ymin=-0.1, ymax=4.0, ls="--", lw=1.0, colors="k", alpha=0.6)
        ax_px.text(15, 1.5, r"$G=1.72$")
        ax_px.text(145, 1.5, r"$G=1.48$")
        ax_px.text(295, 1.5, r"$G=1.61$")
        ax_px.set_xlim(-5, 400)
        ax_px.set_ylim(0.0, 2.0)

        ax_py.set_xlabel("time(s)")
        ax_py.set_ylabel(r"$RMSE_{y}$")
        ax_py.set_title("Position(y)")
        ax_py.vlines(x=100, ymin=-0.1, ymax=4.0, ls="--", lw=1.0, colors="k", alpha=0.6)
        ax_py.vlines(x=250, ymin=-0.1, ymax=4.0, ls="--", lw=1.0, colors="k", alpha=0.6)
        ax_py.text(15, 1.5, r"$G=1.72$")
        ax_py.text(145, 1.5, r"$G=1.48$")
        ax_py.text(295, 1.5, r"$G=1.61$")
        ax_py.set_xlim(-5, 400)
        ax_py.set_ylim(0.0, 2.0)

        # ax_vx.set_xlabel("time(s)")
        ax_vx.set_ylabel(r"$RMSE_{vx}$")
        ax_vx.set_title("Velocity(x)")
        ax_vx.vlines(x=100, ymin=-0.1, ymax=4.0, ls="--", lw=1.0, colors="k", alpha=0.6)
        ax_vx.vlines(x=250, ymin=-0.1, ymax=4.0, ls="--", lw=1.0, colors="k", alpha=0.6)
        ax_vx.text(15, 1.5, r"$G=1.72$")
        ax_vx.text(145, 1.5, r"$G=1.48$")
        ax_vx.text(295, 1.5, r"$G=1.61$")
        ax_vx.set_xlim(-5, 400)
        ax_vx.set_ylim(0.0, 2.0)

        ax_vy.set_xlabel("time(s)")
        ax_vy.set_ylabel(r"$RMSE_{vy}$")
        ax_vy.set_title("Velocity(y)")
        ax_vy.vlines(x=100, ymin=-0.1, ymax=4.0, ls="--", lw=1.0, colors="k", alpha=0.6)
        ax_vy.vlines(x=250, ymin=-0.1, ymax=4.0, ls="--", lw=1.0, colors="k", alpha=0.6)
        ax_vy.text(15, 1.5, r"$G=1.72$")
        ax_vy.text(145, 1.5, r"$G=1.48$")
        ax_vy.text(295, 1.5, r"$G=1.61$")
        ax_vy.set_xlim(-5, 400)
        ax_vy.set_ylim(0.0, 2.0)

        # ax_ax.set_xlabel("time(s)")
        ax_ax.set_ylabel(r"$RMSE_{ax}$")
        ax_ax.set_title("Acceleration(x)")
        ax_ax.vlines(x=100, ymin=-0.1, ymax=4.0, ls="--", lw=1.0, colors="k", alpha=0.6)
        ax_ax.vlines(x=250, ymin=-0.1, ymax=4.0, ls="--", lw=1.0, colors="k", alpha=0.6)
        ax_ax.text(15, 1.5, r"$G=1.72$")
        ax_ax.text(145, 1.5, r"$G=1.48$")
        ax_ax.text(295, 1.5, r"$G=1.61$")
        ax_ax.set_xlim(-5, 400)
        ax_ax.set_ylim(0.0, 2.0)

        ax_ay.set_xlabel("time(s)")
        ax_ay.set_ylabel(r"$RMSE_{ay}$")
        ax_ay.set_title("Acceleration(y)")
        ax_ay.vlines(x=100, ymin=-0.1, ymax=4.0, ls="--", lw=1.0, colors="k", alpha=0.6)
        ax_ay.vlines(x=250, ymin=-0.1, ymax=4.0, ls="--", lw=1.0, colors="k", alpha=0.6)
        ax_ay.text(15, 1.5, r"$G=1.72$")
        ax_ay.text(145, 1.5, r"$G=1.48$")
        ax_ay.text(295, 1.5, r"$G=1.61$")
        ax_ay.set_xlim(-5, 400)
        ax_ay.set_ylim(0.0, 2.0)

        ax_ax.legend(loc="center right")
        if fig_name is None:
            plt.show()
        else:
            plt.savefig(fig_name)

    @staticmethod
    def smooth(x, i0=None, mode=1):
        if i0 is None:
            i0 = 0
        from scipy.signal import savgol_filter
        wl, k = 160, 5
        x_f = savgol_filter(x[i0:], wl, k)
        if mode == 1:
            return np.concatenate((x[:i0], x_f))
        else:
            return x

    @staticmethod
    def cal_average_c(recorders):
        ave_ci_list = []
        for ri in recorders:
            c_array = np.array(ri["C1"], dtype=int)
            tri = np.sum(c_array) / c_array.size
            ave_ci_list.append(tri)
        return sum(ave_ci_list) / len(ave_ci_list)

    def transmit_rate_analysis(self, keys):
        r1_list = []
        r2_list = []
        for k in keys:
            rc = self.rcs[k]
            N_sim = len(rc)
            r1, r2 = 0.0, 0.0
            for ri in rc:
                c_array = np.array(ri["C1"], dtype=int)
                r1 += np.sum(c_array[:500]) / c_array[:500].size / N_sim
                r2 += np.sum(c_array[500:]) / c_array[500:].size / N_sim
            r1_list.append(r1 * 100), r2_list.append(r2 * 100)
        return r1_list, r2_list

    def trans_rate_plot(self, mode="bar", params=None, info=None):

        plt.rcParams['xtick.labelsize'] = 13
        plt.rcParams['ytick.labelsize'] = 13

        f4 = plt.figure(4, figsize=(10, 6))
        ax: plt.Axes = f4.add_axes((0.1, 0.1, 0.8, 0.8))
        r1, r2 = self.transmit_rate_analysis(params)
        if mode == "bar":
            tick_labels = [rf"$\alpha={a}$" + "\n" + rf"$\beta$={b}" for a, b in params if a is not None]
            if (None, None) in params:
                tick_labels.append("Periodic\nr=17.24%")
            x = list(range(1, len(params) + 1))
            width = 0.4
            x1, x2 = [i - width / 2 for i in x], [i + width / 2 for i in x]
            ax.bar(x1, r1, width=width, label=r"$\bar r_1$")
            ax.bar(x2, r2, width=width, label=r"$\bar r_2$")
            for i in range(len(params)):
                ax.text(x1[i] - 0.2, r1[i] + 0.5, f"{np.round(r1[i], 2)}", fontsize=14)
                ax.text(x2[i] - 0.2, r2[i] + 0.5, f"{np.round(r2[i], 2)}", fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels, fontsize=14)
        if mode == "plot":
            match info:
                case "alpha":
                    x = tuple(zip(*self.rcs.keys()))[0]
                case "beta":
                    x = tuple(zip(*self.rcs.keys()))[0]
                case _:
                    x = list(range(1, len(self.rcs) + 1))
            ax.plot(x, r1, label=r"$\bar r_1$", marker="x")
            ax.plot(x, r2, label=r"$\bar r_2$", marker="s")
            ax.set_xlabel(rf"$\{info}$")
            ax.set_xticks(x)
        ax.set_ylim(0, 40)
        ax.set_ylabel("Transmission rate (%)", fontsize=14)
        ax.legend(fontsize=14)
        plt.show()


def plot_dynamic_topology_RMSE(*recorders: Iterable[dict]):
    Ns = np.array([0, 3, 7, 8, 12, 16])
    Nc = np.array([1, 2, 4, 5, 6, 9, 10, 11, 13, 14, 15, 17, 18, 19])
    e2 = np.array([(np.asarray(rc["S_est"]) - np.asarray(rc["S_real"])) ** 2 for rc in recorders])
    t_seq = np.arange(e2.shape[1]) * 10
    lw = 1.5
    f, axes = plt.subplots(2, 1, figsize=(6, 3.8), sharex=True)
    for ax, components, ylabel in zip(axes, (slice(0, 3), slice(3, 6)), ("Position RMSE (m)", "Velocity RMSE (m/s)")):
        e2_sum = np.sum(e2[:, :, :, components], axis=3)
        RMSE_c = np.sqrt(np.mean(e2_sum[:, :, Nc], axis=(0, 2))) * 1000
        RMSE_s = np.sqrt(np.mean(e2_sum[:, :, Ns], axis=(0, 2))) * 1000
        ax.axvspan(3000, 24000, color="gray", alpha=0.15, zorder=0)
        ax.text(13500, 0.94, "Dynamic Topology Duration", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, color="dimgray")
        ax.axvline(3000, color="k", ls="--", lw=1.0, alpha=0.6)
        ax.plot(t_seq, RMSE_c, color="b", lw=lw, ls="-", marker="o",
                markersize=4, markevery=100, alpha=0.6, label="Follower Satellites")
        ax.plot(t_seq, RMSE_s, color="darkred", lw=lw, ls="-", marker="s",
                markersize=4, markevery=100, alpha=0.6, label="Leader Satellites")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(0, t_seq[-1])
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="both", labelsize=11)
        select = (t_seq >= 5000) & (t_seq <= 24000)
        axins = ax.inset_axes([0.44, 0.28, 0.53, 0.30])
        axins.plot(t_seq[select], RMSE_c[select], color="b", lw=lw, ls="-",
                   marker="o", markersize=4, markevery=200, alpha=0.6)
        axins.plot(t_seq[select], RMSE_s[select], color="darkred", lw=lw, ls="-",
                   marker="s", markersize=4, markevery=200, alpha=0.6)
        axins.set_xlim(5000, 24000)
        axins.set_xticks([5000, 15000, 24000])
        axins.tick_params(axis="both", labelsize=11)
        axins.grid(alpha=0.2)
    axes[0].legend(fontsize=9, loc="upper right", bbox_to_anchor=(1, 0.85), ncol=2)
    axes[1].set_xlabel("Time (s)", fontsize=12)
    f.tight_layout(pad=0.6, h_pad=0.5)
    plt.show()
    return f, axes


def plot_dropout_RMSE(recorders: Dict[str, List[dict]]):
    scenarios = ("dropout_0", "dropout_1", "dropout_2")
    labels = ("No dropout", "1 leader dropout", "2 leader dropouts")
    colors = ("b", "darkred", "teal")
    markers = ("o", "s", "d")
    lw = 1.5
    f, axes = plt.subplots(2, 1, figsize=(6, 3.8), sharex=True)
    f_ss, axes_ss = plt.subplots(2, 1, figsize=(6, 3.8), sharex=True)
    insets = [ax.inset_axes([0.44, 0.28, 0.53, 0.30]) for ax in axes]
    dropout_time = recorders["dropout_1"][0]["dropout_time"]
    k_dropout = int(dropout_time / 10)

    for scenario, label, color, marker in zip(scenarios, labels, colors, markers):
        rcs = recorders[scenario]
        e2 = np.array([(np.asarray(rc["S_fused"]) - np.asarray(rc["S_real"])) ** 2 for rc in rcs])
        C = np.array([rc["C"] for rc in rcs])
        t_seq = np.arange(e2.shape[1]) * 10
        select = (t_seq > 20000) & (t_seq <= 24000)
        zoom = (t_seq >= 5000) & (t_seq <= 24000)
        # C 是每步收到的消息条数；稳态窗口 (20000, 24000]，采样间隔 10 s。
        rate = C[:, select].mean(axis=1) / 10
        for ax, ax_ss, axins, components in zip(axes, axes_ss, insets, (slice(0, 3), slice(3, 6))):
            e2_sum = np.sum(e2[:, :, components], axis=2)
            RMSE = np.sqrt(np.mean(e2_sum, axis=0)) * 1000
            RMSE_ss = np.sqrt(np.mean(e2_sum[:, select], axis=1)) * 1000
            # 退出前后分别平滑，避免跨越退出时刻混合数据。
            RMSE_plot = np.r_[MultiRecorderPlot.smooth(RMSE[:k_dropout]),
            MultiRecorderPlot.smooth(RMSE[k_dropout:])]
            ax.plot(t_seq, RMSE_plot, color=color, lw=lw, ls="-", marker=marker,
                    markersize=4, markevery=100, alpha=0.6, label=label)
            axins.plot(t_seq[zoom], RMSE_plot[zoom], color=color, lw=lw, ls="-",
                       marker=marker, markersize=4, markevery=200, alpha=0.6)
            ax_ss.scatter(rate, RMSE_ss, color=color, s=18, alpha=0.4)
            ax_ss.scatter(rate.mean(), np.sqrt(np.mean(RMSE_ss ** 2)),
                          color=color, s=75, marker="D", edgecolors="k", linewidths=0.5, label=label)

    for ax, ax_ss, axins, ylabel in zip(axes, axes_ss, insets, ("Position RMSE (m)", "Velocity RMSE (m/s)")):
        ax.axvspan(dropout_time, t_seq[-1], color="gray", alpha=0.15, zorder=0)
        ax.axvline(dropout_time, color="k", ls="--", lw=1.0, alpha=0.6)
        ax.text(dropout_time + 400, 0.76, "Leader dropout", transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=10, color="dimgray")
        ax.set_xlim(0, t_seq[-1])
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="both", labelsize=11)
        axins.axvline(dropout_time, color="k", ls="--", lw=0.8, alpha=0.6)
        axins.set_xlim(5000, 24000)
        axins.set_xticks([5000, 15000, 24000])
        axins.tick_params(axis="both", labelsize=11)
        axins.grid(alpha=0.2)
        ax_ss.set_ylabel(ylabel, fontsize=12)
        ax_ss.set_ylim(bottom=0)
        ax_ss.margins(x=0.12)
        ax_ss.tick_params(axis="both", labelsize=11)
        ax_ss.grid(alpha=0.2)
    axes[0].legend(fontsize=9, loc="upper right", ncol=3)
    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes_ss[1].legend(fontsize=9, loc="lower right", ncol=3)
    axes_ss[1].set_xlabel("Received messages (messages/s)", fontsize=12)
    f.tight_layout(pad=0.6, h_pad=0.5)
    f_ss.tight_layout(pad=0.6, h_pad=0.5)
    plt.show()
    return (f, axes), (f_ss, axes_ss)


def plot_fusion_ablation_RMSE(recorders: Dict[str, List[dict]], dt=10.0):
    Nc = np.array([1, 2, 4, 5, 6, 9, 10, 11, 13, 14, 15, 17, 18, 19])
    f, axes = plt.subplots(2, 1, figsize=(6, 3.8), sharex=True)
    insets = [ax.inset_axes([0.44, 0.28, 0.53, 0.30]) for ax in axes]
    labels = ("Matrix-norm weighted fusion", "Averaged weight fusion")
    colors = ("b", "darkred")
    markers = ("o", "s")
    for s, label, c, m in zip(("normal", "fusion"), labels, colors, markers):
        e2 = np.array([(np.asarray(rc["S_fused"])[:, Nc] - np.asarray(rc["S_real"])[:, Nc]) ** 2 for rc in recorders[s]])
        t_seq = np.arange(e2.shape[1]) * dt
        zoom = (t_seq >= 5000) & (t_seq <= 24000)
        for ax, axins, components in zip(axes, insets, (slice(0, 3), slice(3, 6))):
            RMSE = np.sqrt(np.mean(np.sum(e2[:, :, :, components], axis=3), axis=(0, 2))) * 1000
            RMSE_plot = MultiRecorderPlot.smooth(RMSE)
            ax.plot(t_seq, RMSE_plot, color=c, lw=1.5, marker=m, markersize=4, markevery=100, alpha=0.6, label=label)
            axins.plot(t_seq[zoom], RMSE_plot[zoom], color=c, lw=1.5, marker=m,
                       markersize=4, markevery=200, alpha=0.6)

    for ax, ylabel in zip(axes, ("Position RMSE (m)", "Velocity RMSE (m/s)")):
        ax.set_xlim(0, t_seq[-1])
        ax.set_ylim(bottom=0)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
    for ax, axins in zip(axes, insets):
        axins.set_xlim(5000, 24000)
        axins.set_xticks([5000, 15000, 24000])
        axins.tick_params(axis="both", labelsize=11)
        axins.xaxis.label.set_size(12)
        axins.yaxis.label.set_size(12)
        axins.grid(alpha=0.2)
    axes[0].legend(fontsize=11, loc="upper right")
    axes[1].set_xlabel("Time (s)", fontsize=12)
    insets[1].set_ylim(0.06, 0.13)
    f.tight_layout(pad=0.6, h_pad=0.5)
    plt.show()
    return f, axes


def plot_outage_RMSE(*recorders: Iterable[dict]):
    e2 = np.array([(np.asarray(rc["S_fused"]) - np.asarray(rc["S_real"])) ** 2 for rc in recorders])
    t_seq = np.arange(e2.shape[1]) * 10
    starts = recorders[0]["outage_start"]
    durations = recorders[0]["outage_duration"]
    f, axes = plt.subplots(2, 1, figsize=(6, 3.8), sharex=True)
    for ax, components, ylabel in zip(axes, (slice(0, 3), slice(3, 6)),
                                      ("Position RMSE (m)", "Velocity RMSE (m/s)")):
        RMSE = np.sqrt(np.mean(np.sum(e2[:, :, components], axis=2), axis=0)) * 1000
        ax.plot(t_seq, RMSE, color="b", lw=1.5, ls="-", marker="o", markersize=4, markevery=100, alpha=0.6)
        for start, duration in zip(starts, durations):
            ax.axvspan(start, start + duration, color="red", alpha=0.15, zorder=0)
        ax.set_xlim(0, t_seq[-1])
        ax.set_ylim(bottom=0)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
    axes[1].set_xlabel("Time (s)", fontsize=12)
    f.tight_layout(pad=0.6, h_pad=0.5)
    plt.show()
    return f, axes
