""" QEMCExecuter class. """


import os
import json
from typing import Iterable, Any

import networkx as nx
import pandas as pd
from qiskit.providers import Backend

from qemc.qemc_solver import QEMCSolver


class QEMCExecuter:
    """
        A framework intended for systematically executing the QEMC algorithm and storing
        its output data, with the option to tune the QEMC algorithm's paramaters
        in order to test different, multiple configurations with ease.
    """

    def __init__(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name

    def define_graphs(self, graphs: Iterable[nx.Graph]) -> None:
        """A method used to define the graph instances to be executed by the framework."""

        self.graphs = graphs

    def define_qemc_parameters(
        self,
        shots: Iterable[int],
        num_layers: Iterable[int],
        num_blue_nodes: Iterable[int]
    ) -> None:
        """
        TODO
        """
    
        self.shots = shots # S
        self.num_layers = num_layers # L
        self.num_blue_nodes = num_blue_nodes # B

    def define_optimization_process(
        self,
        optimization_method: str,
        optimization_options: list[dict[str, Any]]
    ) -> None:
        """
        TODO
        """

        self.optimization_method = optimization_method
        self.optimization_options = optimization_options

    def define_backends(self, backends: Iterable[Backend]) -> None:
        """
        TODO
        """

        self.backends = backends

    def execute_export(self, num_samples: int, export_path: str) -> None:
        """
        TODO
        """

        path = f"{export_path}/{self.experiment_name}"
        os.mkdir(path)

        print("Executing experiment...")
        print()

        for graph in self.graphs:
            graph_path = path + f"/graph_{graph.name}"
            os.mkdir(graph_path)
            nx.write_graphml(graph, f"{graph_path}/graph.graphml")

            for backend in self.backends:               

                if isinstance(backend.name, str):
                    backend_name = backend.name
                else:
                    backend_name = backend.name()

                num_configurations_per_backend = 0
                total_backend_configurations_metadata = {
                    "date_time": pd.Timestamp.now().isoformat(),
                    "graph": graph.name,
                    "backend_name": backend_name,
                    "best_configuration": None,
                    "best_average_best_cut": 0,
                    "best_cut": 0,
                    "best_partition_bitstring": None,
                    "configurations": dict(),
                }

                backend_path = graph_path + f"/backend_{backend_name}"
                os.mkdir(backend_path)

                for num_blue_nodes in self.num_blue_nodes:
                    blue_nodes_path = backend_path + f"/blue_nodes_{num_blue_nodes}"
                    os.mkdir(blue_nodes_path)

                    for num_layers in self.num_layers:
                        num_layers_path = blue_nodes_path + f"/num_layers_{num_layers}"
                        os.mkdir(num_layers_path)

                        for shots in self.shots:                            
                            shots_path = num_layers_path + f"/shots_{shots}"
                            os.mkdir(shots_path)

                            # If `shots` is `None` = noiseless simulation takes place, no measurements
                            meas = False if shots is None else True

                            for opt_options in self.optimization_options:
                                rhobeg = opt_options.get("rhobeg")
                                opt_options_path = shots_path + f"/rhobeg_{rhobeg}"
                                os.mkdir(opt_options_path)

                                # TODO ANNOTATE
                                configuration_data = pd.DataFrame()

                                # TODO ANNOTATE
                                configuration_metadata = dict()
                                best_cut = 0
                                average_best_cut = 0

                                configuration_metadata["summary"] = {
                                    "date_time": pd.Timestamp.now().isoformat(),
                                    "graph": graph.name,
                                    "backend": backend_name,
                                    "num_blue_nodes": num_blue_nodes,
                                    "num_layers": num_layers,
                                    "shots": shots,
                                    "optimization_method": self.optimization_method,
                                    "optimization_options": opt_options
                                }

                                configuration_stamp = (
                                    f"graph={graph.name}, backend={backend}," \
                                    f" num_blue_nodes={num_blue_nodes}, num_layers={num_layers}," \
                                    f" shots={shots}, optimization_options={opt_options}," \
                                    f" optimization_method={self.optimization_method}"
                                )
                                num_configurations_per_backend += 1
                                total_backend_configurations_metadata["configurations"][num_configurations_per_backend] = {
                                    "stamp": configuration_stamp,
                                    "setting": {
                                        "optimization_method": self.optimization_method,
                                        "num_layers": num_layers,
                                        "optimization_options": opt_options,
                                        "num_blue_nodes": num_blue_nodes,
                                        "shots": shots
                                    },
                                    "data_path": opt_options_path
                                }
                                print(f"Executing configuration {num_configurations_per_backend}: {configuration_stamp}.")

                                for sample in range(num_samples):
                                    print(f"Executing algorithm_repeat={sample}.")

                                    qemc_solver = QEMCSolver(graph, num_blue_nodes=num_blue_nodes)
                                    qemc_solver.construct_ansatz(num_layers, meas=meas)
                                    qemc_res = qemc_solver.run(
                                        shots=shots,
                                        backend=backend,
                                        optimizer_method=self.optimization_method,
                                        optimizer_options=opt_options
                                    )

                                    sample_title = f"sample_{sample}"
                                    configuration_data = pd.concat(
                                        [
                                            configuration_data,
                                            pd.DataFrame(
                                                {
                                                    f"{sample_title}_costs": qemc_res.cost_values,
                                                    f"{sample_title}_cuts": qemc_res.cuts
                                                }
                                            )
                                        ],
                                        axis=1
                                    )

                                    configuration_metadata[sample_title] = {
                                        "best_cut": qemc_res.best_score,
                                        "best_partition_bitstring": qemc_res.best_partition_bitstring,
                                        "best_cost_value": qemc_res.best_cost_value,
                                        "best_params_vector": qemc_res.best_params,
                                        "best_counts": qemc_res.best_counts,
                                    }

                                    average_best_cut += (qemc_res.best_score / num_samples)
                                    if qemc_res.best_score > best_cut:
                                        best_cut = qemc_res.best_score
                                        best_partition_bitstring = qemc_res.best_partition_bitstring
                                        best_sample_id = sample

                                curated_configuration_metadata = {
                                    "best_algorithm_sample": best_sample_id,
                                    "best_cut": best_cut,
                                    "average_best_cut": average_best_cut,
                                    "best_partition_bitstring": best_partition_bitstring
                                }
                                configuration_metadata["summary"].update(curated_configuration_metadata)
                                total_backend_configurations_metadata["configurations"][num_configurations_per_backend].update(curated_configuration_metadata)

                                if average_best_cut > total_backend_configurations_metadata["best_average_best_cut"]:
                                    total_backend_configurations_metadata["best_configuration"] = num_configurations_per_backend
                                    total_backend_configurations_metadata["best_average_best_cut"] = average_best_cut

                                if best_cut > total_backend_configurations_metadata["best_cut"]:
                                    total_backend_configurations_metadata["best_cut"] = best_cut
                                    total_backend_configurations_metadata["best_partition_bitstring"] = best_partition_bitstring

                                with open(f"{opt_options_path}/configuration_metadata.json", "w") as f:
                                    json.dump(configuration_metadata, f, indent=4)

                                configuration_data.to_csv(f"{opt_options_path}/configuration_data.csv")

                                print(f"Done executing {num_samples} repeats for this configuration.")
                                print()

                with open(f"{backend_path}/total_backend_configurations_metadata.json", "w") as f:
                    json.dump(total_backend_configurations_metadata, f, indent=4)

        print("Done executing experiment.")


if __name__ == "__main__":
    from qiskit_aer import StatevectorSimulator
    from gset_graph_to_nx import gset_to_nx

    ex = QEMCExecuter("TRY_04.06.2025")

    # G1 = gset_to_nx("gset_graphs/G1.txt", graph_name="G1")
    G14 = gset_to_nx("gset_graphs/G14.txt", graph_name="G14")
    G15 = gset_to_nx("gset_graphs/G15.txt", graph_name="G15")
    G16 = gset_to_nx("gset_graphs/G16.txt", graph_name="G16")

    rrg = nx.random_regular_graph(d=5, n=16)
    ex.define_graphs([rrg])

    ex.define_qemc_parameters(
        shots=[None],
        num_layers=[10, 50, 100, 150],
        num_blue_nodes=[None]
    )

    ex.define_backends([StatevectorSimulator()])

    ex.define_optimization_process(
        optimization_method="COBYLA",
        optimization_options=[{"maxiter": 10_000}],
    )

    ex.execute_export(num_samples=3, export_path="EXP_DATA")