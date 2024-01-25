"""TODO COMPLETE."""


from typing import Optional, List, Dict, Union
from itertools import combinations

import numpy as np
import networkx as nx


def compute_cut(graph: nx.Graph, partition_bitstring: str) -> int:
    """TODO COMPLETE."""

    cut = 0
    for node_i, node_j in graph.edges:
        cut += 1 if partition_bitstring[node_i] != partition_bitstring[node_j] else 0

    return cut


def get_maxcut_brute_force(graph: nx.Graph, blue_nodes: Optional[int] = None):
    """TODO COMPLETE."""

    num_nodes = graph.number_of_nodes()
    cuts = []

    for node_index in range(2**num_nodes):
        bitstring = format(node_index, "b").zfill(num_nodes)

        # The way to iterate only over the bitstrings with `blue_nodes` blue nodes. It's
        # not an efficient method, but a simple one. Need to replace with an efficient
        # method in the future.
        if blue_nodes is not None and bitstring.count("1") != blue_nodes:
            continue

        cut = compute_cut(graph, bitstring)
        cuts.append((bitstring, cut))

    return sorted(cuts, key=lambda x: x[1], reverse=True)


def get_efficient_maxcut_brute_force_spec_b(graph: nx.Graph, num_blue_nodes: int):
    """TODO COMPLETE."""

    num_nodes = graph.number_of_nodes()
    cuts = []

    for ones_combo in combinations(range(num_nodes), num_blue_nodes):

        bitstring = ""
        for char_index in range(num_nodes):
            if char_index in ones_combo:
                bitstring += "1"
            else:
                bitstring += "0"

        cut = compute_cut(graph, bitstring)
        cuts.append((bitstring, cut))

    return sorted(cuts, key=lambda x: x[1], reverse=True)


def get_cuts_approx_ratios_distribution(
        graph: nx.Graph,
        num_blue_nodes: int
) -> Dict[str, Union[Dict[float, int], int]]:
    """
        Returns a histogram of all N choose `num_blue_nodes` approximation ratios (each approximation
        ratio is associated with a single cut and partition).
    """
    
    bitstring_cut = dict(
        get_efficient_maxcut_brute_force_spec_b(graph, num_blue_nodes)
    )
    
    cuts = list(bitstring_cut.values())
    
    unique_cuts = np.array(
        sorted(
            set(cuts),
            reverse=True
        )
    )
    
    unique_approx_ratios = unique_cuts / unique_cuts[0]
    
    approx_ratios_distribution = {
        round(ratio, ndigits=3): cuts.count(cut) for ratio, cut in zip(unique_approx_ratios, unique_cuts)
    }

    mean_approx_ratio = sum(
        abundance * approx_ratio for approx_ratio, abundance in approx_ratios_distribution.items()
    ) / sum(approx_ratios_distribution.values())
    
    return {
        "distribution": approx_ratios_distribution,
        "maxcut": unique_cuts[0],
        "mean_approx_ratio": mean_approx_ratio
    }    


def get_random_partition(num_nodes: int, blue_nodes: Optional[int] = None):
    """TODO COMPLETE."""
    
    # The case when drawing a partition with a random number of blue nodes
    if blue_nodes is None:
        blue_nodes = np.random.randint(num_nodes + 1)

    ops = list(range(num_nodes))
    bitstring = ["0" for _ in range(num_nodes)]

    for i in range(blue_nodes):
        bitstring[ops.pop(np.random.randint(num_nodes - i))] = "1"

    return "".join(bitstring)


def get_random_cut(graph: nx.Graph, num_iters: int, num_blue_nodes: Optional[int] = None) -> List[int]:
    """TODO COMPLETE."""

    num_nodes = graph.number_of_nodes()

    cuts = np.array(
        [
            compute_cut(
                graph,
                get_random_partition(num_nodes, num_blue_nodes)
            ) for _ in range(num_iters)
        ]
    )

    return cuts, cuts.mean(), cuts.max(), np.argmax(cuts)


def randomized_online_k_nk_maxcut_solver():
    pass