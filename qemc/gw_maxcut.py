from dataclasses import dataclass, asdict

import cvxpy as cvx
import networkx as nx

import numpy as np
from numpy.typing import NDArray


def split(cut):
    n = len(cut)
    S = [i for i in range(n) if cut[i]]
    T = [i for i in range(n) if not cut[i]]
    return S, T


def cut_cost(x, L):
    return 0.25 * x @ L @ x


def int_to_binary(n, int_cut):
    """Converts bitmask(==int) cut representation to list of bits"""

    return np.array([int(c) for c in bin(int_cut)[2:].zfill(n)])


def brute_force_max_cut(G):
    """Compute maximum cut of a graph considering all the possible cuts."""""

    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes))

    max_cut_value = 0
    max_cut = 0

    for int_cut in range(1, 2**(n-1) + 1):
        cut = int_to_binary(n, int_cut)
        value = cut_cost(cut * 2 - 1, L)

        if value > max_cut_value:
            max_cut_value = value
            max_cut = cut

    return max_cut_value


@dataclass
class GWMaxCutResult:
    num_repetitions: int
    cuts: list[np.float64]
    mean_cut: float
    best_cut: float
    partitions: list[NDArray[np.int32]]


def obtain_gw_max_cut(
    graph: nx.Graph,
    num_repetitions: int,
) -> GWMaxCutResult:
    """
    Implements the Goemans-Williamson (GW) algorithm for approximating the MaxCut of a graph.
    Solves the SDP relaxation and performs randomized hyperplane rounding.

    Args:
        graph (nx.Graph): Input graph.
        num_repetitions (int): Number of random hyperplane rounds to perform.

    Returns:
        GWMaxCutResult: Dataclass containing cuts (dimension = num_repetitions),
        mean_cut, best_cut, and partitions(dimension = num_repetitions).
    """

    num_nodes = graph.number_of_nodes()
    L = nx.laplacian_matrix(graph, nodelist=sorted(graph.nodes))

    # SDP solution
    X = cvx.Variable((num_nodes, num_nodes), PSD=True)
    obj = 0.25 * cvx.trace(L.toarray() @ X)
    constr = [cvx.diag(X) == 1]
    problem = cvx.Problem(cvx.Maximize(obj), constraints=constr)
    problem.solve(solver=cvx.SCS)

    # GW algorithm
    u, s, v = np.linalg.svd(X.value)
    U = u * np.sqrt(s)

    cuts = np.zeros(num_repetitions)
    partitions = []
    for i in range(num_repetitions):
        r = np.random.randn(num_nodes)
        r = r / np.linalg.norm(r)

        partition = np.sign(r @ U.T)
        partitions.append([int(sign) for sign in partition])
        
        cuts[i] = float(cut_cost(partition, L))

    return GWMaxCutResult(
        num_repetitions=num_repetitions,
        cuts=[float(cut) for cut in cuts],
        mean_cut=float(np.mean(cuts)),
        best_cut=int(np.max(cuts)),
        partitions=partitions
    )