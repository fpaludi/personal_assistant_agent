from langchain_core.stores import BaseStore
from graph.graph import create_graph

def initialize_app(store: BaseStore):
    """Initialize the application with the graph."""
    graph = create_graph(store)
    return graph
