import json
from pathlib import Path

from Visualization import plot_fusion_ablation_RMSE


if __name__ == "__main__":
    N_mc = 10
    data_dir = Path(__file__).resolve().parent / "AblationExperiment"
    recorders = {}
    for scenario in ("normal", "fusion"):
        recorders[scenario] = []
        for i in range(N_mc):
            with (data_dir / f"ablation_data_{scenario}_MC={i}.json").open(encoding="utf-8") as f:
                recorders[scenario].append(json.load(f))
    fig, axes = plot_fusion_ablation_RMSE(recorders)
