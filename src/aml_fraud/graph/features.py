from __future__ import annotations

import collections
import collections.abc

for _name in ["Mapping", "MutableMapping", "Sequence", "Set", "Iterable"]:
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

import networkx as nx
import pandas as pd

from aml_fraud.preprocessing.schema import AMLColumns


def build_transaction_graph(df: pd.DataFrame, columns: AMLColumns = AMLColumns()):
    graph = nx.DiGraph()
    for row in df[[columns.sender, columns.receiver, columns.amount]].itertuples(index=False):
        sender, receiver, amount = row
        if graph.has_edge(sender, receiver):
            graph[sender][receiver]["weight"] += float(amount)
            graph[sender][receiver]["count"] += 1
        else:
            graph.add_edge(sender, receiver, weight=float(amount), count=1)
    return graph


def build_graph_features(df: pd.DataFrame, columns: AMLColumns = AMLColumns()):
    graph = build_transaction_graph(df, columns)
    undirected = graph.to_undirected()
    pagerank = nx.pagerank(graph, weight="weight") if graph.number_of_nodes() else {}
    clustering = nx.clustering(undirected, weight="weight") if graph.number_of_nodes() else {}
    betweenness = nx.betweenness_centrality(undirected, k=min(500, graph.number_of_nodes()) or None, seed=42)
    rows = []
    for node in graph.nodes:
        rows.append(
            {
                "account": node,
                "in_degree": graph.in_degree(node),
                "out_degree": graph.out_degree(node),
                "pagerank": pagerank.get(node, 0.0),
                "clustering": clustering.get(node, 0.0),
                "betweenness": betweenness.get(node, 0.0),
            }
        )
    return pd.DataFrame(rows).set_index("account") if rows else pd.DataFrame()
