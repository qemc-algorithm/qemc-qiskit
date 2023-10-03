"""TODO COMPLETE."""


from typing import List, Dict, Any

import numpy as np
from qiskit import QuantumCircuit
from scipy.optimize import minimize

from online_graph_gen import get_adj_nodes


class OnlineQEMC:
    """TODO COMPLETE."""

    def __init__(self, num_nodes: int, num_blue_nodes: int):
        """TODO COMPLETE."""

        self.num_nodes = num_nodes
        self.num_blue_nodes = num_blue_nodes

        # TODO ARE WE SURE
        self.num_shots = self.num_blue_nodes ** 2
        self.desired_prob = 1 / num_blue_nodes

    def objective(
        self,
        params: List[float],
        ansatz: QuantumCircuit,
        backend,
    ):
        """
            Online QEMC's objective function.

            Args:
                params: parameters as input for the objective function.

            TODO COMPLETE
        """

        ansatz_with_params = ansatz.bind_parameters(params)
        prob_dist = backend.run(ansatz_with_params, shots=self.num_shots).result().get_counts()

        obj_value = 0
        for node_bitstring, node_prob in prob_dist.items():
            node_index = int(node_bitstring, 2)
            p_i = node_prob

            for adj_node_index in get_adj_nodes(node_index, self.num_nodes):
                p_j = prob_dist.get(adj_node_index, 0)

                obj_value += (p_i + p_j - self.desired_prob)**2 + \
                             (np.abs(p_i - p_j) - self.desired_prob)**2

        if obj_value < self.best_obj_value:
            self.best_obj_value = obj_value
            self.best_prob_dist = prob_dist

        return obj_value
    
    def run(
        self,
        ansatz: QuantumCircuit,
        opt_method: str,
        opt_options: Dict[str, Any],
        backend,
    ):
        """TODO COMPLETE."""

        self.best_obj_value = 99999 # Arbitrary high initial value
        self.best_prob_dist = None

        return minimize(
            fun=self.objective,
            x0=np.random.uniform(0, np.pi, size=ansatz.num_parameters),
            args=(ansatz, backend),
            method=opt_method,
            options=opt_options
        )
        

# TODO REMOVE
if __name__ == "__main__":

    from qiskit.circuit.library import RealAmplitudes
    from qiskit_aer import AerSimulator


    NUM_NODES = 8
    NUM_BLUE_NODES = 8

    num_qubits = int(np.log2(NUM_NODES))

    online_qemc_solver = OnlineQEMC(NUM_NODES, NUM_BLUE_NODES)

    ansatz = RealAmplitudes(num_qubits=num_qubits, entanglement="linear").decompose()
    ansatz.measure_all()

    print(ansatz)

    res = online_qemc_solver.run(
            ansatz=ansatz,
            opt_method="COBYLA",
            opt_options=None,
            shots=1024,
            backend=AerSimulator(),
    )

    print(res)



######################################
        # for _ in range(shots):
        #     bitstring_mes = backend.run(ansatz_with_params, shots=1).result().get_counts().popitem()[0]

        #     if bitstring_mes in prob_dist.keys():
        #         prob_dist[bitstring_mes] += 1 / shots
        #     elif len(prob_dist) < self.num_nodes_to_keep:
        #         prob_dist[bitstring_mes] = 1 / shots
        #     else:
        #         pass # NEED TO THINK AGAIN ON THE MEANING OF ALL THIS

        # saved_prob_total = sum(prob_dist.values())

        # if self.num_nodes_to_keep == self.num_nodes:
        #     default_prob_value = 0
        # else:    
        #     default_prob_value = (1 - saved_prob_total) / (self.num_nodes - self.num_blue_nodes)