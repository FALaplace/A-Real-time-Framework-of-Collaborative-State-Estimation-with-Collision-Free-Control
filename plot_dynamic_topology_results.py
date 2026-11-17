import json
from pathlib import Path

from Visualization import plot_dynamic_topology_RMSE, plot_dropout_RMSE, plot_outage_RMSE


if __name__ == "__main__":
    scenario = "outage"  # switching / dropout / outage
    N_mc = 10
    data_dir = Path(__file__).resolve().parent / "Dynamic_Topology_Results" / scenario
    if scenario == "switching":
        case_files = sorted(data_dir.glob("case_*.json"), key=lambda p: int(p.stem.split("_")[-1]))
        recorders = []
        for path in case_files:
            with path.open(encoding="utf-8") as f:
                recorders.append(json.load(f))
        fig, axes = plot_dynamic_topology_RMSE(*recorders)
    elif scenario == "dropout":
        recorders = {}
        for group in ("dropout_0", "dropout_1", "dropout_2"):
            recorders[group] = []
            for i in range(N_mc):
                with (data_dir / f"case_{i}_{group}.json").open(encoding="utf-8") as f:
                    recorders[group].append(json.load(f))
        figures = plot_dropout_RMSE(recorders)
    elif scenario == "outage":
        case_files = sorted(data_dir.glob("case_*_outage.json"), key=lambda p: int(p.stem.split("_")[1]))
        recorders = []
        for path in case_files:
            with path.open(encoding="utf-8") as f:
                recorders.append(json.load(f))
        fig, axes = plot_outage_RMSE(*recorders)
