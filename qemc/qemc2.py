""" QEMC2 class. """


from typing import Optional, List, Dict
from copy import deepcopy

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.providers import Backend
from scipy.optimize import minimize

from qemc_implementation import QEMCResult
from classical_functions import compute_cut_score


class QEMC2:
    """A MaxCut solver object based on the QEMC algorithm."""

    def __init__(self, graph: nx.Graph, num_blue_nodes: Optional[int] = None) -> None:
        self.graph = graph

        self.num_nodes = graph.number_of_nodes()
        self.num_qubits = int(np.ceil(np.log2(self.num_nodes)))

        if num_blue_nodes is None:
            num_blue_nodes = int(self.num_nodes / 2)
        self.num_blue_nodes = num_blue_nodes

        self.full_prob = 1 / self.num_blue_nodes
        self.prob_th = self.full_prob / 2

    def construct_ansatz(
        self,
        num_layers: Optional[int] = None,
        external_ansatz: Optional[QuantumCircuit] = None
    ) -> None:
        """
        Defines the ansatz for the QEMC algorithm. Can either create a QEMC 'default'
        ansatz with `num_layers` layers as depicted in the QEMC paper, or it can get an ansatz
        from the user via the parameter `external_ansatz`. Only one of the parameters should have
        a value which is not `None`.
        """

        if external_ansatz is not None and num_layers is not None:
            raise ValueError("Only one of the parameters should be assigned with a value.")
        
        elif external_ansatz is None:
            initial_layer = QuantumCircuit(self.num_qubits)
            initial_layer.h(initial_layer.qubits)

            self.ansatz = EfficientSU2(
                num_qubits = self.num_qubits,
                su2_gates=["rx", "ry", "rz"],
                entanglement="sca",
                reps=num_layers,
                initial_state=initial_layer,
                insert_barriers=True
            ).decompose()

        else:
            self.ansatz = external_ansatz

    def run(
        self,
        backend: Backend,
        num_shots: Optional[int] = 0,
        x0_params: Optional[List[float]] = None,
        optimizer_method: Optional[str] = "COBYLA"
    ) -> QEMCResult: # TODO THAT'S CURRENTY NOT TRUE
        """Runs the QEMC algorithm."""

        if x0_params is None:
            x0_params = [np.random.uniform(0, 2*np.pi) for _ in range(self.ansatz.num_parameters)]

        self.cost_values = []
        self.best_cost = 100
        self.best_prob_dist = None

        optimizer_result = minimize(
            fun=self.compute_cost,
            x0=x0_params,
            args=(backend, num_shots),
            method=optimizer_method
        )

        best_partition_bitstring = self.nodes_probs_to_partition_bitstring(self.best_prob_dist)
        best_cut = compute_cut_score(self.graph, best_partition_bitstring)

        return optimizer_result, best_partition_bitstring, best_cut

    def compute_cost(self, params: List[float], backend: Backend, num_shots: int) -> float:
        """Computes the cost function of the QEMC algorithm."""

        binded_ansatz = deepcopy(self.ansatz).bind_parameters(params)

        counts = backend.run(binded_ansatz).result().get_counts()
        if num_shots == 0:
            nodes_probs = {int(bin_node_id, 2): prob for bin_node_id, prob in counts.items()}
        else:
            nodes_probs = {
                int(bin_node_id, 2): count / num_shots for bin_node_id, count in counts.items()
            }

        cost = 0
        for node_i, node_j in self.graph.edges:
            p_i = nodes_probs[node_i]
            p_j = nodes_probs[node_j]

            cost += (p_i + p_j - self.full_prob)**2 + (np.abs(p_i - p_j) - self.full_prob)**2

        self.cost_values.append(cost)
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_prob_dist = nodes_probs

        return cost

    def nodes_probs_to_partition_bitstring(self, nodes_probes: Dict[int, float]) -> str:
        """Turns a dictionary of the form `{node_id: node_prob}` into a bitstring that represents
        the partition of a graph's nodes into 2 subsets."""

        bits_list = ["0" for _ in range(self.num_nodes)]

        print(nodes_probes)
        print("PROB_TH:", self.prob_th)
        print("NUM BLUE NODES:", self.num_blue_nodes)
        print("1/B:", self.full_prob)
        
        for node_index in graph.nodes:
            if nodes_probes[node_index] >= self.prob_th:
                bits_list[-node_index] = "1"
            print(bits_list)

        return "".join(bits_list)

# TODO REMOVE

if __name__ == "__main__":
    from qiskit_aer import StatevectorSimulator
    from classical_functions import brute_force_maxcut

    graph = nx.random_regular_graph(d=3, n=8)
    qemc_solver = QEMC2(graph)
    qemc_solver.construct_ansatz(num_layers=4)
    print(qemc_solver.run(backend=StatevectorSimulator()))

    bfmc = brute_force_maxcut(graph)
    print(bfmc.optimal_partitions)
    print(bfmc.best_score)

    print()
    print()

    from qemc_implementation import QEMC

    solver_2 = QEMC(graph)
    solver_2.construct_ansatz(num_layers=4, meas=False)
    print(solver_2.run())

    # print(compute_cut_score(graph,))