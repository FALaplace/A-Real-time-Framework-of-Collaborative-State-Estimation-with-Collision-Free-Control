"""
消融实验：
只做两个实验：
1，基于矩阵范数的融合退化成平均weight的融合，这是对“矩阵范数融合规则”的消融分析；
2，将DAPF退化成传统的APF，这是对“DAPF controller”的消融分析
"""

from pathlib import Path
import json
from tqdm import trange

import numpy as np
from numpy.linalg import inv, norm
from dynamic_topology_experiment import DynamicTopologyExperiment
from satellite_cluster_state_estimation import APF_controller, umax

"""
fuse state通过调整weighted就能实现切换；DAPF消融通过函数def APF_controller(si: np.ndarray, ti: np.ndarray, Di, d_flag=True)
的d_flag来切换。
"""


def fuse_state(sii, Pii, sij_list, Pij_list, weighted=True):
    if not sij_list:
        return sii.copy(), Pii.copy(), np.array([1.0])
    if weighted:
        P_all = np.asarray([Pii, *Pij_list])
        P_norm = np.array([norm(Pj, 2) for Pj in P_all])
        wi = 1.0 / P_norm
        wi /= np.sum(wi)
    else:
        wi = np.ones(len(Pij_list) + 1) / (len(Pij_list) + 1)
    siF = wi @ np.asarray([sii, *sij_list])
    PiF = wi[0] ** 2 * Pii
    for wij, Pij in zip(wi[1:], Pij_list):
        PiF += (wij ** 2 + 2 * wi[0] * wij) * Pij
    return siF, PiF, wi


class AblationExperiment(DynamicTopologyExperiment):
    """
    消融实验大类
    """
    scenarios = ("fusion", "apf", "normal")

    def __init__(self):
        super().__init__()
        self.output_dir = Path("AblationExperiment")
        self.weighted, self.d_flag = True, True

    def fusion_config(self):
        self.weighted, self.d_flag = False, True

    def APF_config(self):
        self.weighted, self.d_flag = True, False

    def fuse_state(self, sii, Pii, sij_list, Pij_list):
        return fuse_state(sii, Pii, sij_list, Pij_list, weighted=self.weighted)

    def APF_controller(self, si, ti, Di):
        return APF_controller(si, ti, Di, d_flag=self.d_flag)

    def run(self, case, scenario):
        if scenario == "fusion":
            self.fusion_config()
        elif scenario == "apf":
            self.APF_config()
        elif scenario == "normal":
            self.weighted, self.d_flag = True, True
        else:
            raise ValueError(scenario)

        sc, sn = self.make_network(case)
        A_available, I_service = self.availability(case, "fixed")
        uFk = np.array([np.clip(-self.APF_controller(case["S0_esti"][ni], case["St"][ni], []), -umax, umax)
                       for ni in range(self.N)])
        record = dict(S_fused=[case["S0_esti"].tolist()], S_real=[case["S0_real"].tolist()],
                      C1=[[int(sn.G.nodes[ni]["C1"]) for ni in self.leaders]], St=case["St"].tolist(), U=[])
        for k in range(1, self.steps + 1):
            # U[k-1] 是从 S_real[k-1] 推进到 S_real[k] 时实际施加的控制输入。
            record["U"].append(uFk.tolist())
            self.cluster_update(k, case, sc, sn, uFk, A_available[k], I_service[k])
            siF = [inv(sn.G.nodes[ni]["InfoMat_fu"]) @ sn.G.nodes[ni]["InfoVec_fu"] for ni in range(self.N)]
            record["S_fused"].append(np.asarray(siF).tolist())
            record["S_real"].append(sc.fsat_state.tolist())
            record["C1"].append([int(sn.G.nodes[ni]["C1"]) for ni in self.leaders])
        return record

    def run_Mento_Carlo(self, N_mc: int, scenario: str):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"============={scenario}============")
        for i in trange(N_mc):
            case = self.generate_case(i, "fixed")
            record = self.run(case, scenario)
            with (self.output_dir / f"ablation_data_{scenario}_MC={i}.json").open("w", encoding="utf-8") as f:
                json.dump(record, f)


if __name__ == "__main__":
    exper = AblationExperiment()
    exper.run_Mento_Carlo(N_mc=10, scenario="normal")
    exper.run_Mento_Carlo(10, "fusion")
    exper.run_Mento_Carlo(40, "apf")
    
