import json
from dataclasses import asdict
from pathlib import Path

import networkx as nx

from qemc.gw_maxcut import obtain_gw_max_cut


def main():
    graphml_path = Path(
        "/home/ohad/work/quantum_variational_combinatorial_optimization/qemc/",
        "EXP_DATA/ref_29_fig3b_comparison__iters_2000__layers_5to35/",
        "graph_random_graph__p_0.660__density_0.65__seed_0/graph.graphml"
    )
    num_gw_reps = 25

    graph = nx.read_graphml(graphml_path)

    print()
    print(f"Executing GW for {graph.name}.")

    gw_results = obtain_gw_max_cut(graph, num_repetitions=num_gw_reps)
    print(f"GW MaxCut results for {graph.name}:")
    print(f"{gw_results.mean_cut=:.3f}")
    print(f"{gw_results.best_cut=}")

    graph_dir = graphml_path.parent

    with open(Path(graph_dir, "gw_results.json"), "w") as f:
        json.dump(asdict(gw_results), f, indent=4)

if __name__ == "__main__":
    main()