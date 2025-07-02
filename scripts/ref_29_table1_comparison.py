import networkx as nx
from qiskit_aer import StatevectorSimulator

from qemc.benchmarking.executer import QEMCExecuter


# Configuration for the experiment
num_algorithm_repeats = 10
maxiter = 1_000
num_layers = [7, 10] # [3, 4, 5, 6, 7]
rhobegs = [1] # [0.8, 0.9, 1, 1.1, 1.2]

# Graphs generation parameters
num_nodes = 64
probs_vec = [0.307, 0.35, 0.398, 0.441]
seed = 0


graphs = []
for p in probs_vec:
    graph: nx.Graph = nx.fast_gnp_random_graph(n=num_nodes, p=p, seed=seed)
    graph_density = nx.density(graph)
    graph.name = f"random_graph__p_{p:.3f}__density_{graph_density:.2f}__seed_{seed}"

    print()
    print(f"Graph: {graph.name}")
    print(f"{graph.number_of_edges()=}")
    print(f"{nx.is_connected(graph)=}")
    print(f"{graph_density=}")

    graphs.append(graph)

executer = QEMCExecuter(experiment_name="ref_29_table1_comparison_C")
executer.define_graphs(graphs)
executer.define_qemc_parameters(
    shots=[None],
    num_layers=num_layers,
    num_blue_nodes=[None],
)
executer.define_optimization_process(
    optimization_method="COBYLA",
    optimization_options=[{"maxiter": maxiter, "rhobeg": rhobeg} for rhobeg in rhobegs],
)
executer.define_backends([StatevectorSimulator()])
executer.execute_export(num_samples=num_algorithm_repeats, export_path="EXP_DATA")