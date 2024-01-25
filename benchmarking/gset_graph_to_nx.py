"""TODO COMPLETE."""


from typing import Optional

import networkx as nx
import pandas as pd


def gset_to_nx(graph_txt_file_path: str, graph_name: Optional[str] = None) -> nx.Graph:
    """TODO COMPLETE."""

    graph_df = pd.read_csv(graph_txt_file_path, delimiter=" ", header=None)
    graph_df.drop(columns=2, inplace=True)
    graph_df.rename(columns={0: "node_1", 1: "node_2"}, inplace=True)

    # Changing the first index to 0 instead of 1
    graph_df["node_1"] = graph_df["node_1"] - 1
    graph_df["node_2"] = graph_df["node_2"] - 1

    graph = nx.from_pandas_edgelist(graph_df, "node_1", "node_2")

    if graph_name is not None:
        graph.name = graph_name

    return graph