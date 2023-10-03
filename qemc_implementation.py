"""QEMCResult (dataclass) and QEMC classes."""

import math
from dataclasses import dataclass
from collections import OrderedDict
from typing import List, Optional, Union, Dict, Set

import numpy as np
from networkx import Graph
from scipy.optimize import minimize, OptimizeResult

from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend
from qiskit.circuit import ParameterVector
from qiskit.result import Counts
from qiskit_aer import AerSimulator

from classical_functions import compute_cut_score

@dataclass
class QEMCResult:
    """
    Dataclass to be returned by `QEMC_obj.run()`.

    Attributes:
        optimizer_result (OptimizeResult): the result object returned by the QEMC's optimizer.
        best_counts (Dict[str, int]): counts of the best QEMC iteration.
        best_params (List[float]): parameters used in the best QEMC iteration.
        best_cost_value (float): cost function's value in the best QEMC iteration.
        best_partition_bitstring (str): partition defined in the best QEMC iteration.
        best_score (int): cut-core for the best partition.
    """

    optimizer_result: OptimizeResult
    best_counts: Dict[str, int]
    best_params: List[float]
    best_cost_value: float
    best_partition_bitstring: str
    best_score: int
    
class QEMC:
    """
    Interface for approximating a solution for the MaxCut problem w.r.t to a given graph,
    based on the QEMC algorithm.
    """

    def __init__(self, graph: Graph) -> None:
        """
        Args:
            graph (Graph): the graph object to approximate a MaxCut solution for.
        """

        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        self.num_qubits = int(math.log2(self.num_nodes)) # TODO Works only for powers of 2

        self.probability_threshold = 1 / self.num_nodes

    def construct_ansatz(self, num_layers: int) -> QuantumCircuit:
        """
        Creates a parameterized quantum circuit (AKA "Ansatz") in a heuristically manner.

        Args:
            num_layers (int): the number of layers in the Ansatz.
        
        Returns:
            (QuantumCircuit): the Ansatz.
        """

        num_sub_layers = 3
        params_per_layer = self.num_qubits * num_sub_layers
        theta = ParameterVector(name="theta", length=num_layers * params_per_layer)

        ansatz = QuantumCircuit(self.num_qubits)
        ansatz.h(ansatz.qubits)

        for layer_index in range(num_layers):
            ansatz.barrier()

            for qubit_index in range(self.num_qubits):
                theta_index = (layer_index * params_per_layer) + (qubit_index * num_sub_layers)
                ansatz.rx(theta[theta_index], qubit_index)
                ansatz.ry(theta[theta_index + 1], qubit_index)
                ansatz.rz(theta[theta_index + 2], qubit_index)

            for qubit_index in range(self.num_qubits):
                ansatz.cx(qubit_index, (qubit_index + 1) % self.num_qubits)
        
        ansatz.measure_all()

        self.ansatz = ansatz
        return ansatz
    
    def compute_cost_function(self, params: List[float]) -> float:
        """
        Computes the cost function of the QEMC algorithm w.r.t to `self.graph`.

        Args:
            params (List[float]): parameters to assign to `self.ansatz`.

        Returns:
            (float): the value of the cost function.
        """

        counts = self.backend.run(
            self.ansatz.bind_parameters(params),
            shots=self.shots
        ).result().get_counts()

        # `int_counts_probs` = counts with integers as keys and probabilities as values
        int_counts = counts.int_outcomes()
        int_counts_probs = {num: counts / self.shots for num, counts in int_counts.items()}

        # Computing and storing the current cut-score
        blue_nodes = self.classify_nodes(
            counts=counts,
            counts_threshold=self.shots * self.probability_threshold
        )
        partition_bitstring = self.classification_to_bitstring(blue_nodes)
        score = compute_cut_score(self.graph, partition_bitstring)
        self.cut_scores.append(score)

        # Documenting best score's properties
        score_index = len(self.cut_scores)
        try:
            if score > next(reversed(self.best_scores.values())):
                self.best_scores[score_index] = score
                self.best_partition_bitstring = partition_bitstring
                self.best_counts = counts
                self.best_params = params
        except StopIteration:
            self.best_scores[score_index] = score

        # 1/B from the cost function equation, while B is assumed to be half of the nodes
        one_over_B = 1 / (self.num_nodes / 2)

        cost = 0
        for edge in self.graph.edges():

            # Trying to retrieve probabilites from the counts, if not measured assigning p = 0
            try:
                p_j = int_counts_probs[edge[0]]
            except KeyError:
                p_j = 0
            try:
                p_k = int_counts_probs[edge[1]]
            except KeyError:
                p_k = 0

            # Single iteration of the cost function summation
            cost += ((np.abs(p_j - p_k) - one_over_B) ** 2) + ((p_j + p_k - one_over_B) ** 2)
    
        return cost
    
    def classify_nodes(self, counts: Dict[str, int], counts_threshold: float) -> Set[int]:
        """
        Given counts and a threshold line, classifies `self.graph`'s nodes into "blue" (above
        threshold line) and red (below threshold line) groups.

        Args:
            counts (Dict[str, int]): counts to classify.
            counts_threshold (float): threshold line value.

        Retruns:
            (Set[int]): a container for the "blue" nodes.
        """

        return {node[0] for node in counts.items() if node[1] >= counts_threshold}
    
    def classification_to_bitstring(self, blue_nodes: Set[int]) -> str:
        """
        Given the set of "nodes" and a graph, translates it to a bitstring that
        defines the partition into blue ("1") and red ("0") groups.

        Args:
            blue_nodes (Set[int]): a container for the "blue" nodes.

        Returns:
            str: the partition's bitstring.
        """

        bitstring_list = ["0" for _ in range(self.num_nodes)]
 
        for blue_node in blue_nodes:
            bitstring_list[-int(blue_node, 2) - 1] = "1"

        return "".join(bitstring_list)

    def run(
        self,
        x0: Optional[List[Union[int, float]]] = None,
        shots: Optional[int] = 1024,
        backend: Optional[Backend] = AerSimulator(),
        optimizer_method: Optional[str] = "COBYLA"
    ) -> QEMCResult:
        """
        Executes the QEMC algorithm for the given MaxCut problem.

        Args:
            x0 (Optional[List[Union[int, float]]] = None): the initial value of the parameters.
            shots (Optional[Backend] = AerSimulator()): number of simulations executions.
            backend (Backend = AerSimulator()): backend to run simulations upon.
            optimizer_method (Optional[str] = "COBYLA"): optimizer to use (see docstrings
            of `scipy.optimize.minimize` for all available options).

        Returns:
            (QEMCResult): A packed data object contains the optimizer's result and best
            QEMC iteration properties. See QEMCResult's docstrings for additional info.
        """

        self.shots = shots
        self.backend = backend

        if x0 is None:
            x0 = [np.random.uniform(0, 2*np.pi) for _ in range(self.ansatz.num_parameters)]

        # Initiating containers to be filled by the `self.compute_cost_function` method
        self.cut_scores = []
        self.best_scores = OrderedDict()
        self.best_cost_value = self.graph.number_of_edges() * 2 # Safe upper-bound value
        self.best_params = None
        self.best_counts = None
        self.best_partition_bitstring = None
        
        optimizer_result = minimize(
            fun=self.compute_cost_function,
            x0=x0,
            method=optimizer_method
        )

        best_score = next(reversed(self.best_scores.values()))
        self.best_scores[len(self.cut_scores)] = best_score

        return QEMCResult(
            optimizer_result = optimizer_result,
            best_counts=self.best_counts,
            best_params=self.best_params,
            best_cost_value=self.best_cost_value,
            best_score=best_score,
            best_partition_bitstring=self.best_partition_bitstring
        )