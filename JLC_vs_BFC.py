__author__ = "Jhony H. Giraldo"
__license__ = "MIT"

import argparse

import numpy as np
import pickle

import torch
from torch_geometric.utils import to_undirected

from models.JLC_curvature import jost_liu_curvature
from models.BFC_curvature import balanced_forman_curvature

import networkx as nx

import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='stochastic_block',
                    help='Name of dataset.')
parser.add_argument('--sub_dataset', type=str, default='')
parser.add_argument('--rand_split', action='store_true',
                    help='use random splits')
parser.add_argument('--undirected', action='store_true', default=True,
                    help='set to not symmetrize adjacency')
args = parser.parse_args()
print(args)

vector_nodes = np.arange(10, 201, 5)
repetitions = 10

times_BFC = np.zeros((repetitions, vector_nodes.shape[0]))
times_JLC = np.zeros((repetitions, vector_nodes.shape[0]))

dataset_name = args.dataset

for iRep in range(0, repetitions):
    print("Repetition: " + str(iRep))
    cont_nodes = 0
    for number_nodes in vector_nodes:
        ### Load and preprocess data ###
        if dataset_name == 'erdos':
            N = number_nodes
            prob_edges = 0.1
            erdos_renyi_graph = nx.generators.random_graphs.erdos_renyi_graph(N,prob_edges)
            number_edges = erdos_renyi_graph.number_of_edges()
            edge_index_orig = np.zeros((2,number_edges), dtype=int)
            cont = 0
            for u, v in erdos_renyi_graph.edges:
                edge_index_orig[:,cont] = np.array([u,v], dtype=int)
                cont += 1
            edge_index_orig = to_undirected(torch.LongTensor(edge_index_orig))
            edge_index_base = edge_index_orig
            number_edges = edge_index_orig.shape[1]
        elif dataset_name == 'stochastic_block':
            sizes = [int(number_nodes/5), int(number_nodes/5), int(number_nodes/5), int(number_nodes/5), int(number_nodes/5)]
            N = sum(sizes)
            prob_inner_cluster = 0.3
            prob_out_cluster = 0.01
            probs_edges = [[prob_inner_cluster, prob_out_cluster, prob_out_cluster, prob_out_cluster, prob_out_cluster],
                           [prob_out_cluster, prob_inner_cluster, prob_out_cluster, prob_out_cluster, prob_out_cluster],
                           [prob_out_cluster, prob_out_cluster, prob_inner_cluster, prob_out_cluster, prob_out_cluster],
                           [prob_out_cluster, prob_out_cluster, prob_out_cluster, prob_inner_cluster, prob_out_cluster],
                           [prob_out_cluster, prob_out_cluster, prob_out_cluster, prob_out_cluster, prob_inner_cluster]]
            stochastic_graph = nx.stochastic_block_model(sizes, probs_edges)
            number_edges = stochastic_graph.number_of_edges()
            edge_index_orig = np.zeros((2, number_edges), dtype=int)
            cont = 0
            for u, v in stochastic_graph.edges:
                edge_index_orig[:, cont] = np.array([u, v], dtype=int)
                cont += 1
            edge_index_orig = to_undirected(torch.LongTensor(edge_index_orig))
            edge_index_base = edge_index_orig
            number_edges = edge_index_orig.shape[1]  # Number of edges
        elif dataset_name == 'stochastic_block_2_clusters':
            sizes = [int(number_nodes/2), int(number_nodes/2)]
            N = sum(sizes)
            prob_inner_cluster = 0.3
            prob_out_cluster = 0.01
            probs_edges = [[prob_inner_cluster, prob_out_cluster],
                           [prob_out_cluster, prob_inner_cluster]]
            stochastic_graph = nx.stochastic_block_model(sizes, probs_edges)
            number_edges = stochastic_graph.number_of_edges()
            edge_index_orig = np.zeros((2, number_edges), dtype=int)
            cont = 0
            for u, v in stochastic_graph.edges:
                edge_index_orig[:, cont] = np.array([u, v], dtype=int)
                cont += 1
            edge_index_orig = to_undirected(torch.LongTensor(edge_index_orig))
            edge_index_base = edge_index_orig
            number_edges = edge_index_orig.shape[1]  # Number of edges

        if args.undirected:
            edge_index = edge_index_orig.cpu()
            row, col = edge_index
            index_undirected = torch.where(row > col)[0]
            edge_index = edge_index[:, index_undirected]
        else:
            edge_index = edge_index_orig.cpu()
            row, col = edge_index
        start_time = time.time()
        BFC = balanced_forman_curvature(edge_index, number_nodes, args.undirected).cpu()
        end_time = time.time()
        times_BFC[iRep, cont_nodes] = end_time - start_time
        start_time = time.time()
        JLC = jost_liu_curvature(edge_index, number_nodes, args.undirected).cpu()
        end_time = time.time()
        times_JLC[iRep, cont_nodes] = end_time - start_time
        cont_nodes += 1

file_name = 'results/' + args.dataset + '_times_BFC_vs_JLC.pkl'
with open(file_name, 'wb') as f:
    pickle.dump([times_BFC, times_JLC], f)
