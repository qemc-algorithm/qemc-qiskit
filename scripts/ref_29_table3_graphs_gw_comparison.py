import json
from dataclasses import asdict
from pathlib import Path

import networkx as nx

from qemc.gw_maxcut import obtain_gw_max_cut


def main():
    data_path = Path(
        "/home/ohad/work/quantum_variational_combinatorial_optimization/qemc/EXP_DATA/ref_29_table3_comparison__iters_20000__chosen_combinations"
    )
    num_gw_reps = 7

    for graph_dir in sorted(list(data_path.iterdir())):
        graph = nx.read_graphml(graph_dir / "graph.graphml")

        print()
        print(f"Executing GW for {graph.name}.")

        gw_results = obtain_gw_max_cut(graph, num_repetitions=num_gw_reps)
        print(f"GW MaxCut results for {graph.name}:")
        print(f"{gw_results.mean_cut=:.3f}")
        print(f"{gw_results.best_cut=}")

        with open(Path(graph_dir, "gw_results.json"), "w") as f:
            json.dump(asdict(gw_results), f, indent=4)

if __name__ == "__main__":
    main()