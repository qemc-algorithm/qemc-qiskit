# QEMC (Qubit Efficient MaxCut) Algorithm Implementation

Implementation (with Qiskit) of the QEMC algorithm described in the paper **A Variational Qubit-Efficient MaxCut Heuristic Algorithm**: [arXiv:2308.10383](https://arxiv.org/abs/2308.10383).

## Installation

```
git clone <SSL/HTTPS HERE AS YOU PREFER>
pip install -r requirements.txt
pip install .
```

## Basic Usage

```
import networkx as nx
from qiskit_aer import StatevectorSimulator
from qemc.qemc_solver import QEMCSolver


graph = nx.random_regular_graph(d=4, n=14)
qemc_solver = QEMCSolver(graph=graph)
qemc_solver.construct_ansatz(num_layers=3, meas=False)

qemc_result = qemc_solver.run(backend=StatevectorSimulator())
```

Now `qemc_result` holds a dataclass of type `QEMCResult` (see `qemc.qemc_solver.QEMCResult` for details).

## Benchmarking Framework Usage

```
import networkx as nx
from qiskit_aer import StatevectorSimulator

from qemc.benchmarking.executer import QEMCExecuter


ex = QEMCExecuter(experiment_name="NAME_OF_YOUR_CHOICE")

graph1 = nx.random_regular_graph(d=3, n=16)
graph2 = nx.random_regular_graph(d=5, n=16)
ex.define_graphs([graph1, graph2])

ex.define_qemc_parameters(
    shots=[None],
    num_layers=[10, 15],
    num_blue_nodes=[None]
)

ex.define_backends([StatevectorSimulator()])

ex.define_optimization_process(
    optimization_method="COBYLA",
    optimization_options=[{"maxiter": 1_000}],
)

# Experiment data and matadata will be saved into `export_path/experiment_name` directory
ex.execute_export(num_samples=3, export_path="DATA_PATH_OF_YOUR_CHOICE")
```

`QEMCExecuter` framework allows for a nested, multi-configuration execution of QEMC. Data will be saved in a logical nested way for each configuration combination.
