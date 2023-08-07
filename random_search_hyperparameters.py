__author__ = "Jhony H. Giraldo"
__license__ = "MIT"

import os
import argparse
from os.path import exists

import numpy as np
import pickle

import sys

parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--gpu_number', type=int, default=0,
                    help='GPU index.')
parser.add_argument('--search_iterations', type=int, default=100,
                    help='Number of search iterations.')
parser.add_argument("--ResidualDenseConnection", action='store_true', default=False,
                    help='Activate residual/dense connections.')
parser.add_argument("--PairNorm", action='store_true', default=False,
                    help='Activate pairNorm.')
parser.add_argument("--DiffGroupNorm", action='store_true', default=False,
                    help='Activate DiffGroupNorm Layers.')
parser.add_argument("--DropEdge", action='store_true', default=False,
                    help='Activate DropEdge Layers.')
parser.add_argument("--JostLiuCurvature", action='store_true', default=False,
                    help='Activate JostLiuCurvature.')
parser.add_argument("--JostLiuCurvature_Online", action='store_true', default=False,
                    help='Activate JostLiuCurvature_Online.')
parser.add_argument("--FALayer", action='store_true', default=False,
                    help='Activate Fully Adjacent Layer.')
parser.add_argument("--GraphDifussion", action='store_true', default=False,
                    help='Activate Graph Difussion preprocessing.')
parser.add_argument("--RicciCurvature", action='store_true', default=False,
                    help='Activate Ricci flow curvature rewiring method.')
parser.add_argument('--GNN', type=str, default='GCN',
                    help='GNN model.')
parser.add_argument('--dataset', type=str, default='Cornell',
                    help='Name of dataset.')
args = parser.parse_args()
print(args)

dataset = args.dataset

# Search space general hyperparameters.
epochs = 1000
lr_space = np.array([0.005, 0.02])
weight_decay_space = np.array([1e-4, 1e-3])
hidden_units_space = np.array([16, 32, 64, 128])
dropout_space = np.array([0.3, 0.6])
n_layers_space = np.array([2, 3, 4])
# Search space hyperparameters of Jost & Liu curvature.
if args.JostLiuCurvature:
    if dataset in ['Pubmed', 'squirrel']:
        pA_space = np.array([0, 0.2])
    else:
        pA_space = np.array([0, 1])
    pD_space = np.array([0, 1])
    tau_space = np.array([1, 500])
    alpha_space = np.array([0, 1])
if args.JostLiuCurvature_Online:
    pA_space = np.array([0, 1])
    pD_space = np.array([0, 1])
    alpha_space = np.array([0, 1])
# Search space hyperparameters of DropEdge.
if args.DropEdge:
    pD_space = np.array([0, 1])
# Search space hyperparameters of PairNorm.
if args.PairNorm:
    s_space = np.array([0.1, 1, 10, 50, 100])
# Search space hyperparameters of Differential Group Normalization.
if args.DiffGroupNorm:
    clusters_space = np.array([3, 4, 5, 6, 7, 8, 9, 10])
    lambda_space = np.array([5e-4, 5e-2])
if args.GraphDifussion:
    alphaGDC_space = np.array([0.01, 0.2])
    k_space = np.array([16, 32, 64, 128])
if args.RicciCurvature:
    tau_space = np.array([1, 500])
    iterRicci_space = np.array([20, 4000])
    c_space = np.array([0.1, 40])
# Number of search iterations
search_iterations = args.search_iterations
# Look if there is previous jobs
file_name_hyperparameters = 'hyperparameters_tuning/' + args.GNN + '_hyperTuning/' + dataset + '_rDC_' + \
                            str(args.ResidualDenseConnection) + '_pN_' + str(args.PairNorm) + '_DE_' + \
                            str(args.DropEdge) + '_DGN_' + str(args.DiffGroupNorm) + '_JLC_' + \
                            str(args.JostLiuCurvature) + '_JLCo_' + str(args.JostLiuCurvature_Online) + '_FA_' + \
                            str(args.FALayer) + '_GD_' + str(args.GraphDifussion) + '_RC_' + str(args.RicciCurvature) + '.pkl'
# Check if there are work in this dataset
if exists(file_name_hyperparameters):
    with open(file_name_hyperparameters, 'rb') as f:
        hyperparameters = pickle.load(f)
    lr_parameters = hyperparameters["lr"]
    weight_decay_parameters = hyperparameters["weight_decay"]
    hidden_units_parameters = hyperparameters["hidden_units"]
    dropout_parameters = hyperparameters["dropout"]
    n_layers_parameters = hyperparameters["n_layers"]
    if args.JostLiuCurvature:
        pA_parameters = hyperparameters["pA"]
        pD_parameters = hyperparameters["pD"]
        tau_parameters = hyperparameters["tau"]
        alpha_parameters = hyperparameters["alpha"]
    if args.JostLiuCurvature_Online:
        pA_parameters = hyperparameters["pA"]
        pD_parameters = hyperparameters["pD"]
        alpha_parameters = hyperparameters["alpha"]
    if args.DropEdge:
        pD_parameters = hyperparameters["pD"]
    if args.PairNorm:
        s_parameters = hyperparameters["s"]
    if args.DiffGroupNorm:
        clusters_parameters = hyperparameters["clusters"]
        lambda_parameters = hyperparameters["lambda_p"]
    if args.GraphDifussion:
        alphaGDC_parameters = hyperparameters["alphaGDC"]
        k_parameters = hyperparameters["k"]
    if args.RicciCurvature:
        tau_parameters = hyperparameters["tau"]
        iterRicci_parameters = hyperparameters["iterRicci"]
        c_parameters = hyperparameters["c"]
    indx_zeros = np.where(lr_parameters == 0)
else:
    lr_parameters = np.zeros((search_iterations,))
    weight_decay_parameters = np.zeros((search_iterations,))
    hidden_units_parameters = np.zeros((search_iterations,))
    dropout_parameters = np.zeros((search_iterations,))
    n_layers_parameters = np.zeros((search_iterations,))
    if args.JostLiuCurvature:
        pA_parameters = np.zeros((search_iterations,))
        pD_parameters = np.zeros((search_iterations,))
        tau_parameters = np.zeros((search_iterations,))
        alpha_parameters = np.zeros((search_iterations,))
    if args.JostLiuCurvature_Online:
        pA_parameters = np.zeros((search_iterations,))
        pD_parameters = np.zeros((search_iterations,))
        alpha_parameters = np.zeros((search_iterations,))
    if args.DropEdge:
        pD_parameters = np.zeros((search_iterations,))
    if args.PairNorm:
        s_parameters = np.zeros((search_iterations,))
    if args.DiffGroupNorm:
        clusters_parameters = np.zeros((search_iterations,))
        lambda_parameters = np.zeros((search_iterations,))
    if args.GraphDifussion:
        alphaGDC_parameters = np.zeros((search_iterations,))
        k_parameters = np.zeros((search_iterations,))
    if args.RicciCurvature:
        tau_parameters = np.zeros((search_iterations,))
        iterRicci_parameters = np.zeros((search_iterations,))
        c_parameters = np.zeros((search_iterations,))
    indx_zeros = np.where(lr_parameters == 0)
for i in range(0, search_iterations):
    print('Search iteration: ' + str(i))
    sys.stdout.flush()
    repeat_flag = False
    if i not in indx_zeros[0]:
        file_name_base = args.dataset + '_nL_' + str(int(n_layers_parameters[i])) + '_hU_' + str(int(hidden_units_parameters[i])) + \
                         '_lr_' + str(lr_parameters[i]) + '_wD_' + str(weight_decay_parameters[i]) + '_dr_' + \
                         str(dropout_parameters[i]) + '_mE_' + str(epochs) + '_rDC_' + \
                         str(args.ResidualDenseConnection) + '_pN_' + str(args.PairNorm) + '_DE_' + \
                         str(args.DropEdge) + '_DGN_' + str(args.DiffGroupNorm) + '_JLC_' + \
                         str(args.JostLiuCurvature) + '_JLCo_' + str(args.JostLiuCurvature_Online) + '_FA_' + \
                         str(args.FALayer) + '_GD_' + str(args.GraphDifussion) + '_RC_' + str(args.RicciCurvature)
        if args.JostLiuCurvature:
            complement_file_name = '_pA_' + str(pA_parameters[i]) + '_pD_' + str(pD_parameters[i]) + \
                                   '_tau_' + str(tau_parameters[i]) + '_alpha_' + str(alpha_parameters[i])
            file_name_base = file_name_base + complement_file_name
        if args.JostLiuCurvature_Online:
            complement_file_name = '_pA_' + str(pA_parameters[i]) + '_pD_' + str(pD_parameters[i]) + \
                                   '_alpha_' + str(alpha_parameters[i])
            file_name_base = file_name_base + complement_file_name
        if args.DropEdge:
            complement_file_name = '_pD_' + str(pD_parameters[i])
            file_name_base = file_name_base + complement_file_name
        if args.PairNorm:
            complement_file_name = '_s_' + str(s_parameters[i])
            file_name_base = file_name_base + complement_file_name
        if args.DiffGroupNorm:
            complement_file_name = '_c_' + str(int(clusters_parameters[i])) + '_lam_' + str(lambda_parameters[i])
            file_name_base = file_name_base + complement_file_name
        if args.GraphDifussion:
            complement_file_name = '_alphaGDC_' + str(alphaGDC_parameters[i]) + '_k_' + str(int(k_parameters[i]))
            file_name_base = file_name_base + complement_file_name
        if args.RicciCurvature:
            complement_file_name = '_tau_' + str(tau_parameters[i]) + '_iRic_' + str(int(iterRicci_parameters[i])) + \
                                   '_c_' + str(c_parameters[i])
            file_name_base = file_name_base + complement_file_name
        file_name = 'hyperparameters_tuning/' + args.GNN + '_hyperTuning/' + file_name_base + '.pkl'
        if exists(file_name):
            with open(file_name, 'rb') as f:
                best_acc_test_vec, best_acc_val_vec = pickle.load(f)
            indx_zero = np.where(best_acc_test_vec == 0)
            if indx_zero[0].size > 0:
                repeat_flag = True
                print('Repeating experiment.')
                sys.stdout.flush()
            else:
                print('This experiment is fine.')
        else:
            repeat_flag = True
            print('Repeating experiment.')
            sys.stdout.flush()
    if (i in indx_zeros[0]) or repeat_flag:
        # Random sampling in the search space.
        lr = np.random.uniform(lr_space[0], lr_space[1], size=1)
        lr = np.round(lr, decimals=4)
        lr_parameters[i] = lr
        weight_decay = np.random.uniform(weight_decay_space[0], weight_decay_space[1], size=1)
        weight_decay = np.round(weight_decay, decimals=4)
        weight_decay_parameters[i] = weight_decay
        hidden_units = np.random.choice(hidden_units_space, size=1)
        hidden_units_parameters[i] = hidden_units
        dropout = np.random.uniform(dropout_space[0], dropout_space[1], size=1)
        dropout = np.round(dropout, decimals=4)
        dropout_parameters[i] = dropout
        n_layers = np.random.choice(n_layers_space, size=1)
        n_layers_parameters[i] = n_layers
        # Jost & Liu curvature random sampling.
        if args.JostLiuCurvature:
            pA = np.random.uniform(pA_space[0], pA_space[1], size=1)
            pA = np.round(pA, decimals=4)
            pA_parameters[i] = pA
            pD = np.random.uniform(pD_space[0], pD_space[1], size=1)
            pD = np.round(pD, decimals=4)
            pD_parameters[i] = pD
            tau = np.random.uniform(tau_space[0], tau_space[1], size=1)
            tau = np.round(tau, decimals=4)
            tau_parameters[i] = tau
            alpha = np.random.uniform(alpha_space[0], alpha_space[1], size=1)
            alpha = np.round(alpha, decimals=4)
            alpha_parameters[i] = alpha
        if args.JostLiuCurvature_Online:
            pA = np.random.uniform(pA_space[0], pA_space[1], size=1)
            pA = np.round(pA, decimals=4)
            pA_parameters[i] = pA
            pD = np.random.uniform(pD_space[0], pD_space[1], size=1)
            pD = np.round(pD, decimals=4)
            pD_parameters[i] = pD
            alpha = np.random.uniform(alpha_space[0], alpha_space[1], size=1)
            alpha = np.round(alpha, decimals=4)
            alpha_parameters[i] = alpha
        if args.DropEdge:
            pD = np.random.uniform(pD_space[0], pD_space[1], size=1)
            pD = np.round(pD, decimals=4)
            pD_parameters[i] = pD
        if args.PairNorm:
            s = np.random.choice(s_space, size=1)
            s_parameters[i] = s
        if args.DiffGroupNorm:
            clusters = np.random.choice(clusters_space, size=1)
            clusters_parameters[i] = clusters
            lambda_p = np.random.uniform(lambda_space[0], lambda_space[1], size=1)
            lambda_p = np.round(lambda_p, decimals=4)
            lambda_parameters[i] = lambda_p
        if args.GraphDifussion:
            alphaGDC = np.random.uniform(alphaGDC_space[0], alphaGDC_space[1], size=1)
            alphaGDC = np.round(alphaGDC, decimals=4)
            alphaGDC_parameters[i] = alphaGDC
            k = np.random.choice(k_space, size=1)
            k_parameters[i] = k
        if args.RicciCurvature:
            tau = np.random.uniform(tau_space[0], tau_space[1], size=1)
            tau = np.round(tau, decimals=4)
            tau_parameters[i] = tau
            iterRicci = np.random.uniform(iterRicci_space[0], iterRicci_space[1], size=1)
            iterRicci = int(iterRicci)
            iterRicci_parameters[i] = iterRicci
            c = np.random.uniform(c_space[0], c_space[1], size=1)
            c = np.round(c, decimals=4)
            c_parameters[i] = c
        base_script_name = 'python main.py --epochs ' + str(epochs) + ' --lr ' + str(lr[0]) + ' --weight_decay ' + \
                           str(weight_decay[0]) + ' --hidden_units ' + str(hidden_units[0]) + ' --n_layers_set ' + \
                           str(n_layers[0]) + ' --dropout ' + str(dropout[0]) + ' --dataset ' + dataset + ' --GNN ' + \
                           args.GNN + ' --hyperparameterTunning_mode'
        if args.JostLiuCurvature:
            complement_script_name = ' --JostLiuCurvature --pA ' + str(pA[0]) + ' --pD ' + str(pD[0]) + ' --tau ' + \
                                     str(tau[0]) + ' --alpha ' + str(alpha[0])
            base_script_name = base_script_name + complement_script_name
        if args.JostLiuCurvature_Online:
            complement_script_name = ' --JostLiuCurvature_Online --pA ' + str(pA[0]) + ' --pD ' + str(pD[0]) + ' --alpha ' + \
                                     str(alpha[0])
            base_script_name = base_script_name + complement_script_name
        if args.DropEdge:
            complement_script_name = ' --DropEdge --pD ' + str(pD[0])
            base_script_name = base_script_name + complement_script_name
        if args.PairNorm:
            complement_script_name = ' --PairNorm --s ' + str(s[0])
            base_script_name = base_script_name + complement_script_name
        if args.DiffGroupNorm:
            complement_script_name = ' --DiffGroupNorm --clusters ' + str(clusters[0]) + ' --lambda_p ' + str(lambda_p[0])
            base_script_name = base_script_name + complement_script_name
        if args.ResidualDenseConnection:
            complement_script_name = ' --ResidualDenseConnection'
            base_script_name = base_script_name + complement_script_name
        if args.FALayer:
            complement_script_name = ' --FALayer'
            base_script_name = base_script_name + complement_script_name
        if args.GraphDifussion:
            complement_script_name = ' --GraphDifussion --k ' + str(k[0]) + ' --alphaGDC ' + str(alphaGDC[0])
            base_script_name = base_script_name + complement_script_name
        if args.RicciCurvature:
            complement_script_name = ' --RicciCurvature --tau ' + str(tau[0]) + ' --iterRicci ' + str(iterRicci) + \
                                     ' --c ' + str(c[0])
            base_script_name = base_script_name + complement_script_name
        if args.no_cuda:
            script_execution_name = base_script_name
        else:
            script_execution_name = 'CUDA_VISIBLE_DEVICES=' + str(args.gpu_number) + ' ' + base_script_name
        # Run the main script with the given hyperparameters.
        os.system(script_execution_name)
        file_name = 'hyperparameters_tuning/' + args.GNN + '_hyperTuning/' + dataset + '_rDC_' + \
                    str(args.ResidualDenseConnection) + '_pN_' + str(args.PairNorm) + '_DE_' + str(args.DropEdge) + \
                    '_DGN_' + str(args.DiffGroupNorm) + '_JLC_' + str(args.JostLiuCurvature) + '_JLCo_' + \
                    str(args.JostLiuCurvature_Online) + '_FA_' + str(args.FALayer) + '_GD_' + str(args.GraphDifussion) + \
                    '_RC_' + str(args.RicciCurvature) + '.pkl'
        # Save the hyperparameters
        hyperparameters = dict()
        hyperparameters["lr"] = lr_parameters
        hyperparameters["weight_decay"] = weight_decay_parameters
        hyperparameters["hidden_units"] = hidden_units_parameters
        hyperparameters["dropout"] = dropout_parameters
        hyperparameters["n_layers"] = n_layers_parameters
        if args.JostLiuCurvature:
            hyperparameters["pA"] = pA_parameters
            hyperparameters["pD"] = pD_parameters
            hyperparameters["tau"] = tau_parameters
            hyperparameters["alpha"] = alpha_parameters
        if args.JostLiuCurvature_Online:
            hyperparameters["pA"] = pA_parameters
            hyperparameters["pD"] = pD_parameters
            hyperparameters["alpha"] = alpha_parameters
        if args.DropEdge:
            hyperparameters["pD"] = pD_parameters
        if args.PairNorm:
            hyperparameters["s"] = s_parameters
        if args.DiffGroupNorm:
            hyperparameters["clusters"] = clusters_parameters
            hyperparameters["lambda_p"] = lambda_parameters
        if args.GraphDifussion:
            hyperparameters["alphaGDC"] = alphaGDC_parameters
            hyperparameters["k"] = k_parameters
        if args.RicciCurvature:
            hyperparameters["tau"] = tau_parameters
            hyperparameters["iterRicci"] = iterRicci_parameters
            hyperparameters["c"] = c_parameters
        with open(file_name, 'wb') as f:
            pickle.dump(hyperparameters, f)
