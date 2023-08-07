__author__ = "Jhony H. Giraldo"
__license__ = "MIT"

import argparse

import pickle

import torch
from torch_geometric.utils import to_undirected, remove_self_loops

from models.JLC_curvature import compute_JLC_matrices
from load_data.data import get_dataset

DATA_PATH = 'data'

# Arguments to run the experiment
parser = argparse.ArgumentParser()

parser.add_argument("--JostLiuCurvature_Online", action='store_true', default=True,
                    help='Activate JostLiuCurvature_Online.')
parser.add_argument("--pA", type=float, default=1,
                    help='Percentage of added edges.')
parser.add_argument('--dataset', type=str, default='squirrel',
                    help='Name of dataset, options: {Cornell, Texas, Wisconsin, chameleon,'
                         'Actor, squirrel, Cora, Citeseer, Pubmed}')
parser.add_argument('--undirected', action='store_true', default=True,
                    help='set to not symmetrize adjacency')
args = parser.parse_args()
print(args)

dataset_name = args.dataset
# Load and preprocess data
if dataset_name in ['chameleon', 'squirrel', 'Actor']:
    file_name = 'data/' + dataset_name + '.pkl'
    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)
    dataset = dataset[0]
else:
    dataset = get_dataset(name=dataset_name, use_lcc=True)
if args.undirected:
    dataset.data.edge_index = to_undirected(dataset.data.edge_index)
dataset.data.edge_index = remove_self_loops(dataset.data.edge_index)[0]
n_edges = dataset.data.edge_index.shape[1]  # Number of edges
n = dataset.data.num_nodes  # Number of nodes


if args.JostLiuCurvature_Online:
    file_name_matrices = DATA_PATH + '/' + args.dataset + '/undirected_' + \
                         str(args.undirected) + '_adding_dropping_matrices.pkl'
    matrix_JLC, JLCc_indices, JLCc_values = compute_JLC_matrices(dataset.data.edge_index, n, n_edges, args.pA,
                                                   force_undirected=args.undirected)
    # Normalization matrix_JLC
    JLC_values = matrix_JLC.coalesce().values()
    JLC_values_normalized = JLC_values - torch.min(JLC_values)
    JLC_values_normalized = JLC_values_normalized / torch.max(JLC_values_normalized)
    JLC_indices = matrix_JLC.coalesce().indices()
    # Normalization matrix_JLCc
    JLCc_values_normalized = JLCc_values - torch.min(JLCc_values)
    JLCc_values_normalized = JLCc_values_normalized / torch.max(JLCc_values_normalized)
    with open(file_name_matrices, 'wb') as f:
        pickle.dump([JLCc_values_normalized, JLC_values_normalized, JLCc_indices, JLC_indices], f)
