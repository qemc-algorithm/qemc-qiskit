"""TODO COMPLETE."""


from typing import Optional, List

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

    best_cut = 0
    best_bitstring = None

    for node_index in range(2**num_nodes):
        bitstring = format(node_index, "b").zfill(num_nodes)

        # The way to iterate only over the bitstrings with `blue_nodes` blue nodes. It's
        # not an efficient method, but a simple one. Need to replace with an efficient
        # method in the future.
        if blue_nodes is not None and bitstring.count("1") != blue_nodes:
            continue

        cut = compute_cut(graph, bitstring)

        if cut > best_cut:
            best_cut = cut
            best_bitstring = bitstring

    return (best_bitstring, best_cut)


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