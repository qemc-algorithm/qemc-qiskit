"""QEMCResult (dataclass) and QEMCSolver classes."""

import math
from dataclasses import dataclass
from collections import OrderedDict
from typing import List, Optional, Dict, Any

import numpy as np
import networkx as nx
from scipy.optimize import minimize, OptimizeResult

from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend
from qiskit.circuit import ParameterVector
from qiskit_aer import StatevectorSimulator

from qemc.classical_functions import compute_cut_score


@dataclass
class QEMCResult:
    """
    Dataclass to be returned by the method `run()` of a `QEMCSolver` object.

    Attributes:
        optimizer_result (OptimizeResult): the result object returned by the chosen SciPy optimizer.
        best_counts (Dict[str, int]): `Counts` object of the best QEMC iteration.
        best_params (List[float]): parameter vector used in the best QEMC iteration.
        best_cost_value (float): cost function's value in the best QEMC iteration.
        best_score (int): cut-score for the best partition.
        best_partition_bitstring (str): partition defined in the best QEMC iteration.
        cuts (List[int]): all cut-scores (ordered).
        cost_values (List[float]): all the values of the cost function (ordered).
    """

    optimizer_result: OptimizeResult

    best_counts: Dict[str, int]
    best_params: List[float]
    best_cost_value: float
    best_partition_bitstring: str
    best_score: int

    cuts: List[int]
    cost_values: List[float]


class QEMCSolver:
    """
    A solver interface for approximating a solution for the MaxCut
    problem w.r.t to a given graph, based on the QEMC algorithm.
    """

    def __init__(
        self,
        graph: nx.Graph,
        num_blue_nodes: Optional[int] = None,
    ) -> None:
        """
        Args:
            graph (Graph): the graph object to approximate a MaxCut solution for.
            num_blue_nodes: Optional[int] = None TODO
        """

        self.graph = graph
        self.num_nodes = graph.number_of_nodes()
        
        # TODO ANNOTATE?
        if num_blue_nodes is None:
            self.num_blue_nodes = self.num_nodes / 2
        else:
            self.num_blue_nodes = num_blue_nodes

        # TODO ANNOTATE?
        self.desired_blue_prob = 1 / self.num_blue_nodes
        self.probability_threshold = self.desired_blue_prob / 2

        # TODO ANNOTATE?
        self.num_qubits = math.ceil(math.log2(self.num_nodes))
        
    def construct_ansatz(self, num_layers: int, meas: Optional[bool] = True) -> QuantumCircuit:
        """
        Creates a parameterized quantum circuit (AKA "Ansatz") in a heuristically manner.

        Args:
            num_layers (int): the number of layers in the Ansatz.
            meas: Optional[bool] = True TODO
        
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

        if meas:
            ansatz.measure_all()

        self.ansatz = ansatz
        return ansatz
    
    def run(
        self,
        x0: list[int | float] | None = None,
        shots: int | None = None,
        backend: Backend = StatevectorSimulator(),
        optimizer_method: str = "COBYLA",
        optimizer_options: dict[str, Any] | None = None
    ) -> QEMCResult:
        """
        Executes the QEMC algorithm for the given MaxCut problem.

        Args:
            x0: the initial value of the parameters.
            shots: number of simulations executions.
            backend: backend to run simulations upon.
            optimizer_method: optimizer to use (see docstrings of `scipy.optimize.minimize` for all available options).
            optimizer_options: Additional options dictionary to pass for the optimizer 
            (see docstrings of `scipy.optimize.minimize` for all available options).

        Returns:
            (QEMCResult): A packed data object contains the optimizer's result and best
            QEMC iteration properties. See QEMCResult's docstrings for additional info.
        """

        self.shots = shots
        self.backend = backend

        if x0 is None:
            x0 = np.random.uniform(0, 2*np.pi, size=self.ansatz.num_parameters)

        # Initiating containers to be filled by the `self.compute_cost_function` method
        self.cut_scores = []
        self.cost_values = []
        self.best_scores = OrderedDict()
        self.best_params = []
        self.best_counts = dict()
        self.best_partition_bitstring = None
        
        optimizer_result = minimize(
            fun=self.compute_cost_function,
            x0=x0,
            method=optimizer_method,
            options=optimizer_options
        )

        # TODO ANNOTATE
        best_score = next(reversed(self.best_scores.values()))
        self.best_scores[len(self.cut_scores)] = best_score

        return QEMCResult(
            optimizer_result=optimizer_result,
            best_counts=self.best_counts,
            best_params=list(self.best_params),
            best_cost_value=min(self.cost_values),
            best_score=best_score,
            best_partition_bitstring=self.best_partition_bitstring,
            cuts=self.cut_scores,
            cost_values=self.cost_values
        )
    
    def compute_cost_function(self, params: list[float]) -> float:
        """
        Computes the cost function of the QEMC algorithm w.r.t to `self.graph`.

        Args:
            params (List[float]): parameters to assign to `self.ansatz`.

        Returns:
            (float): the value of the cost function.
        """

        counts = self.backend.run(
            self.ansatz.assign_parameters(params),
            shots=self.shots
        ).result().get_counts()

        if self.shots is None:
            shots_dividing_factor = 1 # The case of ideal simulation
        else:
            shots_dividing_factor = self.shots

        # Counts with integers as keys and probabilities as values
        int_counts_probs = {
            int(bin_num, 2): counts / shots_dividing_factor for bin_num, counts in counts.items()
        }

        # Computing and storing the current cut-score
        blue_nodes = self.classify_nodes(
            counts=counts,
            counts_threshold=shots_dividing_factor * self.probability_threshold
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

        cost = 0
        for edge in self.graph.edges():

            # Trying to retrieve probabilites from the counts, if not measured assigning p = 0
            p_j = int_counts_probs.get(edge[0], 0)
            p_k = int_counts_probs.get(edge[1], 0)

            # Single iteration of the QEMC's cost function summation
            cost += (
                (np.abs(p_j - p_k) - self.desired_blue_prob) ** 2
            ) + (
                (p_j + p_k - self.desired_blue_prob) ** 2
            )
    
        # Storing cost value
        self.cost_values.append(cost)

        return cost
    
    def classify_nodes(self, counts: dict[str, int], counts_threshold: float) -> set[int]:
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
    
    def classification_to_bitstring(self, blue_nodes: set[int]) -> str:
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
            try:
                bitstring_list[-int(blue_node, 2) - 1] = "1"
            except IndexError:
                pass

        return "".join(bitstring_list)