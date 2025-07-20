import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from qemc.classical_functions import get_random_partition, compute_cut_score


def find_graphml_file(config_path: Path) -> Path:
    """Find the corresponding .graphml file up the hierarchy for each config file."""

    for parent in config_path.parents:
        graphml_files = list(parent.glob("*.graphml"))
        if graphml_files:
            return graphml_files[0]
        
    raise FileNotFoundError(f"No .graphml file found for {config_path}")


def main(updated_bool_field_name: str = "is_updated_optimization_process") -> None:
    data_files_paths = list(Path("EXP_DATA").rglob("configuration_data.csv"))
    graphml_file = None
    avg_random_cut = []

    tail_std_threshold = 5e-4
    tail_window_size = 50

    for data_file_path in data_files_paths:
        data_dir_path = data_file_path.parent
        print()
        print(f"Processing configuration: {data_dir_path}...")

        if check_if_updated(data_dir_path, updated_bool_field_name=updated_bool_field_name):
            print(
                f"THIS CONFIGURATION HAS ALREADY BEEN UPDATED WITH {updated_bool_field_name} METADATA. "
                "MOVING ON TO THE NEXT CONFIGURATION."
            )
            continue

        data = pd.read_csv(data_file_path)

        # Number of algorithm repetitions, a.k.a number of samples
        num_algorithm_repetitions = (data.shape[1] - 1) / 2
        if num_algorithm_repetitions != int(num_algorithm_repetitions):
            raise ValueError("The number of algorithm iterations is not an integer.")
        num_algorithm_repetitions = int(num_algorithm_repetitions)

        # Number optimizer iterations
        num_optimizer_iterations = data.shape[0]

        avg_best_random_cut = np.zeros(num_optimizer_iterations)
        just_update_best_random_cut = True

        # Average cost and cut over all algorithm repetitions (sum of columns divided by `num_algorithm_repetitions`)
        avg_cost = data.iloc[:, 1::2].mean(axis=1)
        avg_cut = data.iloc[:, 2::2].mean(axis=1)

        # STD
        avg_cost_window = avg_cost[-tail_window_size:]
        cost_tail_std = np.std(avg_cost_window)

        # Compute average random cut per optimizer iteration
        new_graphml_filepath = find_graphml_file(data_file_path)
        if new_graphml_filepath != graphml_file:
            graphml_file = new_graphml_filepath
            graph = nx.read_graphml(graphml_file)

            avg_best_random_cut = obtain_best_random_cut_vector(
                graph=graph,
                num_optimizer_iterations=num_optimizer_iterations,
                num_algorithm_repetitions=num_algorithm_repetitions,
            )

            if updated_bool_field_name == "is_updated_optimization_process":
                just_update_best_random_cut = False

                avg_random_cut = []
                for _ in range(num_optimizer_iterations):
                    avg_current_cut = 0
                    for _ in range(num_algorithm_repetitions):
                        random_partition = get_random_partition(num_nodes=graph.number_of_nodes())
                        avg_current_cut += compute_cut_score(graph=graph, bitstring=random_partition) / num_algorithm_repetitions

                    avg_random_cut.append(avg_current_cut)
            else:
                metadata_file = Path(data_dir_path, "configuration_metadata.json")

                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                avg_random_cut = metadata["optimization_process"].get("y_avg_random_cut", [0 for _ in range(num_optimizer_iterations)])

        kwargs = dict(
            num_optimizer_iterations=num_optimizer_iterations,
            num_algorithm_repetitions=num_algorithm_repetitions,
            avg_cost=avg_cost,
            avg_cut=avg_cut,
            avg_random_cut=avg_random_cut,
            avg_best_random_cut=avg_best_random_cut,
            data_dir_path=data_dir_path,
            cost_tail_std=cost_tail_std,
            tail_window_size=tail_window_size,
            just_update_best_random_cut=just_update_best_random_cut,
        )

        alter_metadata(**kwargs)        
        plot(**kwargs)

        print(f"Done with {data_dir_path}.")


def obtain_best_random_cut_vector(
    graph: nx.Graph,
    num_optimizer_iterations: int,
    num_algorithm_repetitions: int,
) -> NDArray[np.float64]:
    """TODO COMPLETE."""

    avg_random_best_cut = np.zeros(num_optimizer_iterations)
    for repetition_index in range(num_algorithm_repetitions):
        current_best_random_cut = np.zeros(num_optimizer_iterations)

        for iteration_index in range(num_optimizer_iterations):
            random_partition = get_random_partition(num_nodes=graph.number_of_nodes())
            cut = compute_cut_score(graph=graph, bitstring=random_partition)

            if iteration_index == 0 or cut > current_best_random_cut[iteration_index - 1]:
                current_best_random_cut[iteration_index] = cut
            else:
                current_best_random_cut[iteration_index] = current_best_random_cut[iteration_index - 1]

        avg_random_best_cut += (current_best_random_cut / num_algorithm_repetitions)

    return avg_random_best_cut


def plot(
    num_optimizer_iterations: int,
    num_algorithm_repetitions: int,
    avg_cost: np.ndarray | pd.Series,
    avg_cut: np.ndarray | pd.Series,
    avg_random_cut: np.ndarray | pd.Series | list[float],
    data_dir_path: Path,
    cost_tail_std: float,
    tail_window_size: int,
    avg_best_random_cut: NDArray[np.float64],
    just_update_best_random_cut: bool = False,
) -> None:
    """Plot the average cost and cut per optimizer iteration."""

    x_axis = range(num_optimizer_iterations)
    linewidth = 0.5

    fig, ax1 = plt.subplots()

    color = "tab:blue"
    ax1.set_xlabel("Optimizer Iterations")
    ax1.set_ylabel("Cost", color=color)
    ax1.plot(x_axis, avg_cost, label="avg_cost", color=color, linewidth=linewidth)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:orange"
    ax2.set_ylabel("Cut", color=color)
    ax2.plot(x_axis, avg_cut, label="avg_cut", color=color, linewidth=linewidth)
    ax2.tick_params(axis="y", labelcolor=color)

    # Plot avg_random_cut on the same axis as avg_cut
    ax2.plot(x_axis, avg_random_cut, label="avg_random_cut", color="tab:green", linewidth=linewidth)

    # Plot avg_best_random_cut on the same axis as avg_cut
    ax2.plot(x_axis, avg_best_random_cut, label="avg_best_random_cut", color="tab:red", linewidth=linewidth)

    # Add legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="lower left")

    plt.title(
        f"Average Cost and Cut per Optimizer Iteration, over {num_algorithm_repetitions} algorithm repetitions \n"
        f"Cost tail STD: {cost_tail_std:.6f} over last {tail_window_size} iterations"
    )
    fig.tight_layout()

    plt.savefig(Path(data_dir_path, "optimization_process.png"))
    plt.close(fig)


def check_if_updated(data_dir_path: Path, updated_bool_field_name: str = "is_updated_optimization_process") -> bool:
    """Check if the metadata file has already been updated with optimization process data."""
    
    metadata_file = Path(data_dir_path, "configuration_metadata.json")

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    return metadata["summary"].get(updated_bool_field_name, False)


def alter_metadata(
    num_optimizer_iterations: int,
    num_algorithm_repetitions: int,
    avg_cost: NDArray[np.float64],
    avg_cut: NDArray[np.float64],
    avg_random_cut: NDArray[np.float64] | list[float],
    avg_best_random_cut: NDArray[np.float64],
    data_dir_path: Path,
    cost_tail_std: float,
    tail_window_size: int,
    just_update_best_random_cut: bool = False,
) -> bool:
    """Alter the metadata file to include the optimization process data."""
    
    metadata_file = Path(data_dir_path, "configuration_metadata.json")

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    if just_update_best_random_cut:
        metadata["optimization_process"].update(
            {
                "y_avg_best_random_cut": list(avg_best_random_cut)
            }
        )

        metadata["summary"].update(
            {
                "is_updated_best_random_cut": True,
            }
        )

    else:
        metadata["summary"].update(
            {
                "num_algorithm_repetitions": num_algorithm_repetitions,
                "is_updated_optimization_process": True,
                "cost_tail_std": cost_tail_std,
                "tail_window_size": tail_window_size,
            }
        )

        metadata["optimization_process"] = {
            "x_iterations": list(range(num_optimizer_iterations)),
            "y_avg_cost": list(avg_cost),
            "y_avg_cut": list(avg_cut),
            "y_avg_best_random_cut": list(avg_best_random_cut),
            "y_avg_random_cut": list(avg_random_cut)
        }

    with open(metadata_file, "w") as file:
        json.dump(metadata, file, indent=4)

    return True

    
if __name__ == "__main__":
    main(updated_bool_field_name="is_updated_best_random_cut")