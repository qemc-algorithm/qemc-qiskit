"""TODO COMPLETE."""


from typing import Set

import numpy as np


def get_adj_nodes(node_index: int, num_nodes: int) -> Set[int]:
    return {
        (node_index + 1) % num_nodes,
        (node_index + 2) % num_nodes,
        (node_index + 3) % num_nodes,
        (node_index + 4) % num_nodes,
        (node_index + 5) % num_nodes
    }


def gen_whole_graph(num_nodes: int) -> Set[int]:
    """TODO COMPLETE."""

    edges = set()
    distinct_nodes = set()

    i = 0
    while len(distinct_nodes) < num_nodes:

        adj_nodes = get_adj_nodes(i, num_nodes)

        edges.update({(i, j) for j in adj_nodes})
        distinct_nodes.update(adj_nodes)

        i += 1

    return edges


# TODO REMOVE
if __name__ == "__main__":
    print()
    print(gen_whole_graph(32))