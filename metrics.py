import networkx as nx
from scipy.stats import ks_2samp, skew

def average_clustering_coefficient(graph):
    """
    Calculates the average clustering coefficient of a graph.

    Parameters:
    -----------
    graph : NetworkX graph object
        The graph whose average clustering coefficient is to be calculated.

    Returns:
    --------
    float
        The average clustering coefficient of the graph.

    Notes:
    ------
    The clustering coefficient of a node in a graph measures the degree to
    which its neighbors are also connected to each other. The average clustering
    coefficient of a graph is the average of the clustering coefficients of all
    its nodes. It is a measure of the degree of local clustering or cohesion in
    the graph. This function calculates the average clustering coefficient of
    a graph using the NetworkX library.
    """
    return nx.average_clustering(graph)



def global_clustering_coefficient(graph):
    """
    Calculates the global clustering coefficient of a graph.

    Parameters:
    -----------
    graph : NetworkX graph object
        The graph whose global clustering coefficient is to be calculated.

    Returns:
    --------
    float
        The global clustering coefficient of the graph.

    Notes:
    ------
    The global clustering coefficient of a graph measures the degree to which
    nodes in the graph tend to form triangles or clusters of nodes with high
    connectivity. It is the ratio of the number of triangles in the graph to
    the number of connected triples of nodes. The global clustering coefficient
    is a measure of the degree of global clustering or cohesion in the graph.
    This function calculates the global clustering coefficient of a graph
    using the NetworkX library.
    """
    return nx.transitivity(graph)



def number_of_connected_components(graph):
    """
    Calculates the number of connected components in a graph.

    Parameters:
    -----------
    graph : NetworkX graph object
        The graph whose number of connected components is to be calculated.

    Returns:
    --------
    int
        The number of connected components in the graph.

    Notes:
    ------
    A connected component of a graph is a subset of its nodes such that each
    node is connected to at least one other node in the subset. The number of
    connected components in a graph is a basic topological property that
    characterizes its structure and connectivity. This function calculates
    the number of connected components in a graph using the NetworkX library.
    """
    return nx.number_connected_components(graph)



def kolmogorov_smirnove_distance(graph, s_graph):
    """
    Calculates the Kolmogorov-Smirnov distance between the degree distributions
    of the original graph and a sampled graph.

    Parameters:
    -----------
    graph : NetworkX graph object
        The original graph.
    s_graph : NetworkX graph object
        The sampled graph.

    Returns:
    --------
    float
        The Kolmogorov-Smirnov distance between the degree distributions of
        the two graphs.

    Notes:
    ------
    The Kolmogorov-Smirnov distance is a non-parametric statistical test that
    compares two probability distributions. In this case, it is used to compare
    the degree distributions of the original graph and a sampled graph. The
    distance is a measure of the maximum difference between the cumulative
    distribution functions of the two degree distributions. A smaller distance
    indicates a closer match between the two distributions.
    """
    return ks_2samp(nx.degree_histogram(graph), nx.degree_histogram(s_graph))[0]



def skew_divergence_distance(graph, s_graph):
    """
    Calculates the absolute difference in the skewness of the degree distributions
    of the original graph and a sampled graph.

    Parameters:
    -----------
    graph : NetworkX graph object
        The original graph.
    s_graph : NetworkX graph object
        The sampled graph.

    Returns:
    --------
    float
        The absolute difference in the skewness of the degree distributions of
        the two graphs.

    Notes:
    ------
    The skewness of a distribution measures its asymmetry. A positive skewness
    indicates that the distribution is skewed to the right, while a negative
    skewness indicates that the distribution is skewed to the left. The skewness
    of a degree distribution is often used as a measure of its shape.

    The skew divergence distance is a simple measure of the difference between
    the skewness of two degree distributions. It is calculated as the absolute
    difference in the skewness of the two distributions. A smaller distance
    indicates a closer match between the two distributions.
    """
    return abs(skew(nx.degree_histogram(graph)) - skew(nx.degree_histogram(s_graph)))