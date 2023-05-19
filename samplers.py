import numpy as np
import networkx as nx
from utils import MCGS, MCGS_batch_major, MCGS_enhanced_minor, MCGS_batch_major_enhanced_minor
import random



def random_edge_node_sampler(G, sampling_rate):
    G_s = nx.Graph()

    edges = list(G.edges)
    random.shuffle(edges)

    for edge in edges:
        if len(G_s.nodes) >= len(G.nodes)*sampling_rate:
            break

        G_s.add_edge(*edge)

    for edge in G.edges:
        if edge[0] in G_s.nodes and edge[1] in G_s.nodes:
            G_s.add_edge(*edge)

    return G_s



def random_jump_sampler(G, sampling_rate, jump_probability=0.01):
    G_s = nx.Graph()

    current = np.random.choice(G.nodes) 
    while len(G_s.nodes) < len(G.nodes)*sampling_rate:
        if np.random.choice([0,1], p=[1-jump_probability, jump_probability]) == 1:
            current = np.random.choice(G.nodes) 
            continue

        next_n = np.random.choice(list(G.neighbors(current)))
        G_s.add_edge(current, next_n)
        current = next_n


    for edge in G.edges:
        if edge[0] in G_s.nodes and edge[1] in G_s.nodes:
            G_s.add_edge(*edge)

    return G_s



def spectral_sampler_vertices(G, sampling_rate):
    L = nx.laplacian_matrix(G).todense()

    eig_vals, eig_vecs = np.linalg.eig(L)
    sorted_eig_vals = sorted(eig_vals)

    resistances = [ ((i,j), 1 / (sorted_eig_vals[i] - sorted_eig_vals[j]))  for i in range(len(G)) for j in range(i, len(G)) if i != j and L[i,j] != 0]
    resistances = sorted(resistances, key=lambda x:x[1], reverse=True)
    resistance_dict = {r[0]:r[1] for r in resistances}

    vertex_dict = {}

    for key, value in resistance_dict.items():
        for node in key:
            if node not in vertex_dict:
                vertex_dict[node] = value
            else:
                vertex_dict[node] += value

    vertex_resistances = [(key, value) for key, value in vertex_dict.items()]

    vertex_resistances = sorted(vertex_resistances, key=lambda x:x[1], reverse=True)
    sampled_vertices = [vr[0] for vr in vertex_resistances[:int(len(vertex_resistances)*sampling_rate)]]

    G_s = nx.Graph()
    for vertex in sampled_vertices:
        G_s.add_node(vertex)

    for edge in G.edges:
        if edge[0] in G_s.nodes and edge[1] in G_s.nodes:
            G_s.add_edge(*edge)

    return G_s



def MCGS_sampler(G, sampling_rate):
    return MCGS().run_sampling(G, sampling_rate)



### Proposed algorithms ###

# Base algorithms #
def MCGS_batch_major_sampler(G, sampling_rate):
    return MCGS_batch_major().run_sampling(G, sampling_rate)


def MCGS_cc_sampler(G, sampling_rate):
    selected_nodes = []

    for comp in nx.connected_components(G):
        GC = G.subgraph(comp)
        g_s = MCGS_sampler(GC, sampling_rate)
        selected_nodes.extend(g_s.nodes)

    return G.subgraph(selected_nodes)


def MCGS_enhanced_minor_sampler(G, sampling_rate):
    return MCGS_enhanced_minor().run_sampling(G, sampling_rate)


# Ensembled algorithms #
def MCGS_batch_major_cc_sampler(G, sampling_rate):
    selected_nodes = []

    for comp in nx.connected_components(G):
        GC = G.subgraph(comp)
        g_s = MCGS_batch_major_sampler(GC, sampling_rate)
        selected_nodes.extend(g_s.nodes)

    return G.subgraph(selected_nodes)


def MCGS_batch_major_enhanced_minor_sampler(G, sampling_rate):
    return MCGS_batch_major_enhanced_minor().run_sampling(G, sampling_rate)


def MCGS_enhanced_minor_cc_sampler(G, sampling_rate):
    selected_nodes = []

    for comp in nx.connected_components(G):
        GC = G.subgraph(comp)
        g_s = MCGS_enhanced_minor_sampler(GC, sampling_rate)
        selected_nodes.extend(g_s.nodes)

    return G.subgraph(selected_nodes)


def MCGS_batch_major_enhanced_minor_cc_sampler(G, sampling_rate):
    selected_nodes = []

    for comp in nx.connected_components(G):
        GC = G.subgraph(comp)
        g_s = MCGS_batch_major_enhanced_minor_sampler(GC, sampling_rate)
        selected_nodes.extend(g_s.nodes)

    return G.subgraph(selected_nodes)