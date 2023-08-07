__author__ = "Jhony H. Giraldo"
__license__ = "MIT"

import argparse

import numpy as np
import pickle
import seaborn as sns

from os.path import exists

import torch
import torch.nn.functional as functional
from torch_geometric.utils import to_undirected, remove_self_loops

from models.GNN_models import GCN, SGC
from models.JLC_curvature import compute_JLC_matrices
from models.BFC_curvature import stochastic_discrete_ricci_flow_rewiring
from load_data.data import get_dataset, PPRDataset, set_train_val_test_split
from load_data.seeds import val_seeds, test_seeds

import time
import sys

# Arguments to run the experiment
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='Show results.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden_units', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument("--n_layers_set", nargs="+", type=int, default=[2, 8, 32],
                    help='List of number of layers.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument("--ResidualDenseConnection", action='store_true', default=False,
                    help='Activate residual/dense connections.')
parser.add_argument("--PairNorm", action='store_true', default=False,
                    help='Activate pairNorm.')
parser.add_argument('--s', type=float, default=1,
                    help='Scaling factor s of PairNorm.')
parser.add_argument("--DiffGroupNorm", action='store_true', default=False,
                    help='Activate DiffGroupNorm Layers.')
parser.add_argument('--clusters', type=int, default=10,
                    help='Number of clusters of DGN.')
parser.add_argument('--lambda_p', type=float, default=0.01,
                    help='The balancing factor lambda of DGN.')
parser.add_argument("--DropEdge", action='store_true', default=False,
                    help='Activate DropEdge Layers.')
parser.add_argument("--JostLiuCurvature_Online", action='store_true', default=False,
                    help='Activate JostLiuCurvature_Online.')
parser.add_argument("--pA", type=float, default=0.1,
                    help='Percentage of added edges.')
parser.add_argument("--pD", type=float, default=0.1,
                    help='Percentage of dropped edges.')
parser.add_argument("--tau", type=float, default=100,
                    help='Tau of rewiring algorithm.')
parser.add_argument("--alpha", type=float, default=0.5,
                    help='Importance of Jost-Liu curvature value, between 0 and 1.')
parser.add_argument("--FALayer", action='store_true', default=False,
                    help='Activate Fully Adjacent Layer.')
parser.add_argument("--GraphDifussion", action='store_true', default=False,
                    help='Activate Graph Difussion preprocessing.')
parser.add_argument("--alphaGDC", type=float, default=0.05,
                    help='alpha value for graph difussion method.')
parser.add_argument("--k", type=int, default=128,
                    help='k value for graph difussion method.')
parser.add_argument("--RicciCurvature", action='store_true', default=False,
                    help='Activate Ricci flow curvature rewiring method.')
parser.add_argument("--iterRicci", type=int, default=1000,
                    help='Max number of iterations for SDRF.')
parser.add_argument("--c", type=float, default=0.5,
                    help='Ricci upper-bound C+.')
parser.add_argument('--dataset', type=str, default='Cornell',
                    help='Name of dataset, options: {Cornell, Texas, Wisconsin, chameleon,'
                         'Actor, squirrel, Cora, Citeseer, Pubmed}')
parser.add_argument('--GNN', type=str, default='GCN',
                    help='Name of graph neural network: GCN or SGC.')
parser.add_argument('--undirected', action='store_true', default=True,
                    help='set to not symmetrize adjacency')
parser.add_argument("--hyperparameterTunning_mode", action='store_true', default=False,
                    help='Activate hyperparameter tunning mode.')
parser.add_argument("--is_deep_experiment", action='store_true', default=False,
                    help='Activate experiments with deep GNNs.')
args = parser.parse_args()
print(args)


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    loss = functional.nll_loss(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    logits, accs_losses = model(data), []
    keys = ['train', 'val', 'test']
    for key in keys:
        mask = data[f'{key}_mask']
        predictions = logits[mask].max(1)[1]
        acc = predictions.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs_losses.append(acc)
        loss = functional.nll_loss(logits[mask], data.y[mask])
        accs_losses.append(loss)
    return accs_losses


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.no_cuda:
    device = 'cpu'
args.device = device
print(device)

dataset_name = args.dataset
# Load and preprocess data
if args.GraphDifussion:
    dataset = PPRDataset(
        name=dataset_name,
        use_lcc=True,
        alpha=args.alphaGDC,
        k=args.k
    )
    if args.undirected:
        dataset.data.edge_index, dataset.data.edge_attr = to_undirected(dataset.data.edge_index, dataset.data.edge_attr)
    dataset.data.edge_index, dataset.data.edge_attr = remove_self_loops(dataset.data.edge_index, dataset.data.edge_attr)
else:
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
dataset.data = dataset.data.to(device)
n_edges = dataset.data.edge_index.shape[1]  # Number of edges
n = dataset.data.num_nodes  # Number of nodes


if args.JostLiuCurvature_Online:
    file_name_matrices = 'data/' + args.dataset + '/undirected_' + str(args.undirected) + '_adding_dropping_matrices.pkl'
    if exists(file_name_matrices):
        with open(file_name_matrices, 'rb') as f:
            JLCc_values_normalized, JLC_values_normalized, JLCc_indices, JLC_indices = pickle.load(f)
        n_possible_edges = int(2 * n_edges * args.pA)
        JLCc_values_normalized = JLCc_values_normalized[0:n_possible_edges]
        JLCc_indices = JLCc_indices[:, 0:n_possible_edges]
    else:
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
    args.JLC_values_normalized = JLC_values_normalized.to(device)
    args.JLCc_values_normalized = JLCc_values_normalized.to(device)
    args.JLC_indices = JLC_indices.to(device)
    args.JLCc_indices = JLCc_indices.to(device)
    # Compute number of edges to drop and to add.
    if args.undirected:
        n_edges2drop = int(n_edges/2 * args.pD)
        n_edges2add = int(n_edges/2 * args.pA)
    else:
        n_edges2drop = int(n_edges * args.pD)
        n_edges2add = int(n_edges * args.pA)
    args.number_edges2drop = n_edges2drop
    args.number_edges2add = n_edges2add

if args.RicciCurvature:
    dataset.data.edge_index = stochastic_discrete_ricci_flow_rewiring(dataset.data.edge_index, args.iterRicci, args.tau,
                                                                      args.c, n, n_edges, device, dataset_name,
                                                                      force_undirected=args.undirected)

if args.hyperparameterTunning_mode:
    seeds = val_seeds
else:
    seeds = test_seeds

for n_layers in args.n_layers_set:
    cont_repetition = 0

    loss_train_vec = np.zeros((len(seeds), args.epochs), )
    loss_val_vec = np.zeros((len(seeds), args.epochs), )
    loss_test_vec = np.zeros((len(seeds), args.epochs), )
    best_acc_test_vec = np.zeros((len(seeds), args.epochs), )
    best_acc_val_vec = np.zeros((len(seeds), args.epochs), )
    err_train_vec = np.zeros((len(seeds), args.epochs), )
    err_test_vec = np.zeros((len(seeds), args.epochs), )
    err_val_vec = np.zeros((len(seeds), args.epochs), )

    for seed in seeds:
        print('Executing repetition ' + str(cont_repetition))

        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)

        if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
            num_development = 1500
            dataset.data = set_train_val_test_split(seed, dataset.data, dataset_name=dataset_name,
                                                    num_development=num_development).to(device)
        else:
            num_development = int(0.8*n)
            dataset.data = set_train_val_test_split(seed, dataset.data, dataset_name=dataset_name,
                                                    num_development=num_development).to(device)

        if args.GNN == 'GCN':  # We can add more models later in the file GNN_models.py
            model = GCN(in_channels=dataset.num_features,
                        out_channels=dataset.num_classes,
                        number_layers=n_layers,
                        args=args).to(device)
        if args.GNN == 'SGC':
            model = SGC(in_channels=dataset.num_features,
                        out_channels=dataset.num_classes,
                        number_layers=n_layers,
                        args=args).to(device)
        if args.GraphDifussion and args.GNN == 'GCN':
            optimizer = torch.optim.Adam([
                dict(params=model.convs[0].parameters(), weight_decay=args.weight_decay),
                {'params': list([p for l in model.convs[1:] for p in l.parameters()]), 'weight_decay': 0}
            ], lr=args.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_acc = test_acc = 0
        for epoch in range(0, args.epochs):
            start_time = time.time()
            loss_train_vec[cont_repetition, epoch] = train(model, dataset.data, optimizer)
            train_acc, loss_train, val_acc, loss_val, tmp_test_acc, loss_test = test(model, dataset.data)
            loss_val_vec[cont_repetition, epoch] = loss_val
            loss_test_vec[cont_repetition, epoch] = loss_test
            err_test_vec[cont_repetition, epoch] = 1 - tmp_test_acc
            err_val_vec[cont_repetition, epoch] = 1 - val_acc
            err_train_vec[cont_repetition, epoch] = 1 - train_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            best_acc_test_vec[cont_repetition, epoch] = test_acc
            best_acc_val_vec[cont_repetition, epoch] = best_val_acc
            end_time = time.time()
            sys.stdout.flush()
        file_name_base = dataset_name + '_nL_' + str(n_layers) + '_hU_' + str(args.hidden_units) + '_lr_' + \
                         str(args.lr) + '_wD_' + str(args.weight_decay) + '_dr_' + str(args.dropout) + '_mE_' + \
                         str(args.epochs) + '_rDC_' + str(args.ResidualDenseConnection) + '_pN_' + \
                         str(args.PairNorm) + '_DE_' + str(args.DropEdge) + '_DGN_' + \
                         str(args.DiffGroupNorm) + '_JLC_' + str(args.JostLiuCurvature) + '_JLCo_' + \
                         str(args.JostLiuCurvature_Online) + '_FA_' + str(args.FALayer) + '_GD_' + \
                         str(args.GraphDifussion) + '_RC_' + str(args.RicciCurvature)
        if args.JostLiuCurvature:
            complement_file_name = '_pA_' + str(args.pA) + '_pD_' + str(args.pD) + '_tau_' + str(args.tau) + \
                                   '_alpha_' + str(args.alpha)
            file_name_base = file_name_base + complement_file_name
        if args.JostLiuCurvature_Online:
            complement_file_name = '_pA_' + str(args.pA) + '_pD_' + str(args.pD) + '_alpha_' + str(args.alpha)
            file_name_base = file_name_base + complement_file_name
        if args.DropEdge:
            complement_file_name = '_pD_' + str(args.pD)
            file_name_base = file_name_base + complement_file_name
        if args.PairNorm:
            complement_file_name = '_s_' + str(args.s)
            file_name_base = file_name_base + complement_file_name
        if args.DiffGroupNorm:
            complement_file_name = '_c_' + str(args.clusters) + '_lam_' + str(args.lambda_p)
            file_name_base = file_name_base + complement_file_name
        if args.GraphDifussion:
            complement_file_name = '_alphaGDC_' + str(args.alphaGDC) + '_k_' + str(args.k)
            file_name_base = file_name_base + complement_file_name
        if args.RicciCurvature:
            complement_file_name = '_tau_' + str(args.tau) + '_iRic_' + str(args.iterRicci) + '_c_' + str(args.c)
            file_name_base = file_name_base + complement_file_name
        if args.hyperparameterTunning_mode and args.is_deep_experiment:
            file_name = 'hyperparameters_tuning/' + args.GNN + '_hyperTuning_Deep/' + file_name_base + '.pkl'
        elif args.hyperparameterTunning_mode:
            file_name = 'hyperparameters_tuning/' + args.GNN + '_hyperTuning/' + file_name_base + '.pkl'
        elif args.is_deep_experiment:
            file_name = 'results/' + args.GNN + '_Deep/' + file_name_base + '.pkl'
        else:
            file_name = 'results/' + args.GNN + '/' + file_name_base + '.pkl'
        if args.hyperparameterTunning_mode:
            with open(file_name, 'wb') as f:
                pickle.dump([best_acc_test_vec[:, -1], best_acc_val_vec[:, -1]], f)
        else:
            with open(file_name, 'wb') as f:
                pickle.dump([loss_train_vec, loss_val_vec, loss_test_vec, best_acc_test_vec, err_test_vec,
                             err_val_vec, err_train_vec], f)
        cont_repetition += 1
    if args.verbose:
        acc_test_vec_test = best_acc_test_vec[:, -1]
        boots_series = sns.algorithms.bootstrap(acc_test_vec_test, func=np.mean, n_boot=1000)
        test_std_test_seeds = np.max(np.abs(sns.utils.ci(boots_series, 95) - np.mean(acc_test_vec_test)))
        results_log = 'The result for rDC_' + str(args.ResidualDenseConnection) + '_pN_' + str(args.PairNorm) + \
                      '_DE_' + str(args.DropEdge) + '_DGN_' + str(args.DiffGroupNorm) + '_JLC_' + \
                      str(args.JostLiuCurvature) + '_JLCo_' + str(args.JostLiuCurvature_Online) + '_FA_' + \
                      str(args.FALayer) + '_GD_' + str(args.GraphDifussion) + '_RC_' + str(args.RicciCurvature) + \
                      ' method in ' + args.dataset + ' dataset is ' + str(np.mean(boots_series)) + '+-' + str(test_std_test_seeds)
        print(results_log)
