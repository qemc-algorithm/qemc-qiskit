import numpy as np
import networkx as nx
from qiskit_aer import StatevectorSimulator

from qemc.benchmarking.executer import QEMCExecuter


probs_vec = np.linspace(0.66, 1, 5)
graphs = []

for p in probs_vec:
    graph = nx.fast_gnp_random_graph(n=32, p=p, seed=0)
    graph.name = f"random_graph_p_{p:.3f}"

    print()
    print(f"Graph: {graph.name}")
    print(f"{graph.number_of_edges()=}")
    print(f"{nx.is_connected(graph)=}")
    print(f"{nx.density(graph)=}")

    graphs.append(graph)


executer = QEMCExecuter(experiment_name="random_graphs_32_nodes_increasing_densities")
executer.define_graphs(graphs)
executer.define_qemc_parameters(
    shots=[None],
    num_layers=[5],
    num_blue_nodes=[None],
)
executer.define_optimization_process(
    optimization_method="COBYLA",
    optimization_options=[{"maxiter": 10_000}]
)
executer.define_backends([StatevectorSimulator()])
executer.execute_export(num_samples=3, export_path="EXP_DATA")