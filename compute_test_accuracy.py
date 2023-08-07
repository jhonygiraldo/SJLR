__author__ = "Jhony H. Giraldo"
__license__ = "MIT"

import os
import argparse

import seaborn as sns

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
                    help='Activate JostLiuCurvature.')
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

epochs = 1000
# Number of search iterations
search_iterations = args.search_iterations
dataset = args.dataset
file_name_hyperparameters = 'hyperparameters_tuning/' + args.GNN + '_hyperTuning/' + dataset + '_rDC_' + \
                            str(args.ResidualDenseConnection) + '_pN_' + str(args.PairNorm) + '_DE_' + \
                            str(args.DropEdge) + '_DGN_' + str(args.DiffGroupNorm) + '_JLC_' + \
                            str(args.JostLiuCurvature) + '_JLCo_' + str(args.JostLiuCurvature_Online) + '_FA_' + \
                            str(args.FALayer) + '_GD_' + str(args.GraphDifussion) + '_RC_' + \
                            str(args.RicciCurvature) + '.pkl'
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
val_mean = np.zeros((search_iterations,))
test_mean = np.zeros((search_iterations,))
test_std = np.zeros((search_iterations,))
for i in range(0, search_iterations):
    # Load results of each experiment.
    file_name_base = dataset + '_nL_' + str(int(n_layers_parameters[i])) + '_hU_' + \
                     str(int(hidden_units_parameters[i])) + '_lr_' + str(lr_parameters[i]) + '_wD_' + \
                     str(weight_decay_parameters[i]) + '_dr_' + str(dropout_parameters[i]) + '_mE_' + str(epochs) + \
                     '_rDC_' + str(args.ResidualDenseConnection) + '_pN_' + str(args.PairNorm) + '_DE_' + \
                     str(args.DropEdge) + '_DGN_' + str(args.DiffGroupNorm) + '_JLC_' + str(args.JostLiuCurvature) + \
                     '_JLCo_' + str(args.JostLiuCurvature_Online) + '_FA_' + str(args.FALayer) + '_GD_' + str(args.GraphDifussion) + \
                     '_RC_' + str(args.RicciCurvature)
    if args.JostLiuCurvature:
        complement_file_name = '_pA_' + str(pA_parameters[i]) + '_pD_' + str(pD_parameters[i]) + '_tau_' + \
                               str(tau_parameters[i]) + '_alpha_' + str(alpha_parameters[i])
        file_name_base = file_name_base + complement_file_name
    if args.JostLiuCurvature_Online:
        complement_file_name = '_pA_' + str(pA_parameters[i]) + '_pD_' + str(pD_parameters[i]) + '_alpha_' + str(alpha_parameters[i])
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
    file_name_results = 'hyperparameters_tuning/' + args.GNN + '_hyperTuning/' + file_name_base + '.pkl'
    with open(file_name_results, 'rb') as f:
        best_acc_test_vec, best_acc_val_vec = pickle.load(f)
    # Compute performances with bootstrapping.
    boots_series = sns.algorithms.bootstrap(best_acc_val_vec, func=np.mean, n_boot=1000)
    val_mean[i] = np.mean(best_acc_val_vec)
    boots_series = sns.algorithms.bootstrap(best_acc_test_vec, func=np.mean, n_boot=1000)
    test_mean[i] = np.mean(boots_series)
    # 95% confidence interval using the bootstrapping.
    test_std[i] = np.max(np.abs(sns.utils.ci(boots_series, 95) - np.mean(best_acc_test_vec)))
# Compute the best performance in the validation set.
best_val_indx = np.where(val_mean == np.max(val_mean))
if best_val_indx[0].shape[0] > 1:
    best_val_indx = ([best_val_indx[0][0]],)
#best_val_indx = best_val_indx[0][0]
results_log = 'The best result for rDC_' + str(args.ResidualDenseConnection) + '_pN_' + str(args.PairNorm) + \
              '_DE_' + str(args.DropEdge) + '_DGN_' + str(args.DiffGroupNorm) + '_JLC_' + \
              str(args.JostLiuCurvature) + '_JLCo_' + str(args.JostLiuCurvature_Online) + '_FA_' + str(args.FALayer) + \
              '_GD_' + str(args.GraphDifussion) + '_RC_' + str(args.RicciCurvature) + ' method with val seeds in ' + \
              dataset + ' dataset is ' + str(test_mean[best_val_indx]) + '+-' + str(test_std[best_val_indx])
print(results_log)
hyperparameters_log = 'n_layers: ' + str(int(n_layers_parameters[best_val_indx])) + ' hiddenUnits: ' + \
                      str(int(hidden_units_parameters[best_val_indx])) + ' lr: ' + str(lr_parameters[best_val_indx]) + \
                      ' weightDecay: ' + str(weight_decay_parameters[best_val_indx]) + ' dropout: ' + \
                      str(dropout_parameters[best_val_indx])
if args.JostLiuCurvature:
    complement_hyperparameters_log = ' pA: ' + str(pA_parameters[best_val_indx]) + ' pD: ' + \
                                     str(pD_parameters[best_val_indx]) + '_tau_' + str(tau_parameters[best_val_indx]) +\
                                     ' alpha: ' + str(alpha_parameters[best_val_indx])
    hyperparameters_log = hyperparameters_log + complement_hyperparameters_log
if args.JostLiuCurvature_Online:
    complement_hyperparameters_log = ' pA: ' + str(pA_parameters[best_val_indx]) + ' pD: ' + \
                                     str(pD_parameters[best_val_indx]) + ' alpha: ' + str(alpha_parameters[best_val_indx])
    hyperparameters_log = hyperparameters_log + complement_hyperparameters_log
if args.DropEdge:
    complement_hyperparameters_log = ' pD: ' + str(pD_parameters[best_val_indx])
    hyperparameters_log = hyperparameters_log + complement_hyperparameters_log
if args.PairNorm:
    complement_hyperparameters_log = ' s: ' + str(s_parameters[best_val_indx])
    hyperparameters_log = hyperparameters_log + complement_hyperparameters_log
if args.DiffGroupNorm:
    complement_hyperparameters_log = ' c: ' + str(clusters_parameters[best_val_indx]) + ' lam: ' + \
                                     str(lambda_parameters[best_val_indx])
    hyperparameters_log = hyperparameters_log + complement_hyperparameters_log
if args.GraphDifussion:
    complement_hyperparameters_log = ' alphaGDC: ' + str(alphaGDC_parameters[best_val_indx]) + ' k: ' + \
                                     str(k_parameters[best_val_indx])
    hyperparameters_log = hyperparameters_log + complement_hyperparameters_log
if args.RicciCurvature:
    complement_hyperparameters_log = ' tau: ' + str(tau_parameters[best_val_indx]) + ' iRic: ' + \
                                     str(iterRicci_parameters[best_val_indx]) + ' c: ' + \
                                     str(c_parameters[best_val_indx])
    hyperparameters_log = hyperparameters_log + complement_hyperparameters_log
print(hyperparameters_log)
sys.stdout.flush()
# Run experiment with test seeds and the best parameters from the validation set.
base_script_name = 'python main.py --epochs ' + str(epochs) + ' --lr ' + str(lr_parameters[best_val_indx][0]) + \
                   ' --weight_decay ' + str(weight_decay_parameters[best_val_indx][0]) + ' --hidden_units ' + \
                   str(int(hidden_units_parameters[best_val_indx])) + ' --n_layers_set ' + \
                   str(int(n_layers_parameters[best_val_indx])) + ' --dropout ' + \
                   str(dropout_parameters[best_val_indx][0]) + ' --dataset ' + dataset + ' --GNN ' + args.GNN
if args.JostLiuCurvature:
    complement_script_name = ' --JostLiuCurvature --pA ' + str(pA_parameters[best_val_indx][0]) + ' --pD ' + \
                             str(pD_parameters[best_val_indx][0]) + ' --tau ' + str(tau_parameters[best_val_indx][0]) +\
                             ' --alpha ' + str(alpha_parameters[best_val_indx][0])
    base_script_name = base_script_name + complement_script_name
if args.JostLiuCurvature_Online:
    complement_script_name = ' --JostLiuCurvature_Online --pA ' + str(pA_parameters[best_val_indx][0]) + ' --pD ' + \
                             str(pD_parameters[best_val_indx][0]) + ' --alpha ' + str(alpha_parameters[best_val_indx][0])
    base_script_name = base_script_name + complement_script_name
if args.DropEdge:
    complement_script_name = ' --DropEdge --pD ' + str(pD_parameters[best_val_indx][0])
    base_script_name = base_script_name + complement_script_name
if args.PairNorm:
    complement_script_name = ' --PairNorm --s ' + str(s_parameters[best_val_indx][0])
    base_script_name = base_script_name + complement_script_name
if args.DiffGroupNorm:
    complement_script_name = ' --DiffGroupNorm --clusters ' + str(int(clusters_parameters[best_val_indx][0])) + \
                             ' --lambda_p ' + str(lambda_parameters[best_val_indx][0])
    base_script_name = base_script_name + complement_script_name
if args.ResidualDenseConnection:
    complement_script_name = ' --ResidualDenseConnection'
    base_script_name = base_script_name + complement_script_name
if args.FALayer:
    complement_script_name = ' --FALayer'
    base_script_name = base_script_name + complement_script_name
if args.GraphDifussion:
    complement_script_name = ' --GraphDifussion --k ' + str(int(k_parameters[best_val_indx][0])) + ' --alphaGDC ' + \
                             str(alphaGDC_parameters[best_val_indx][0])
    base_script_name = base_script_name + complement_script_name
if args.RicciCurvature:
    complement_script_name = ' --RicciCurvature --tau ' + str(tau_parameters[best_val_indx][0]) + ' --iterRicci ' + \
                             str(int(iterRicci_parameters[best_val_indx][0])) + \
                             ' --c ' + str(c_parameters[best_val_indx][0])
    base_script_name = base_script_name + complement_script_name
if args.no_cuda:
    script_execution_name = base_script_name
else:
    script_execution_name = 'CUDA_VISIBLE_DEVICES=' + str(args.gpu_number) + ' ' + base_script_name
os.system(script_execution_name)
# Compute results with bootstrapping and test seeds
file_name_base = 'results/' + args.GNN + '/' + dataset + '_nL_' + str(int(n_layers_parameters[best_val_indx])) + '_hU_'\
                 + str(int(hidden_units_parameters[best_val_indx])) + '_lr_' + str(lr_parameters[best_val_indx][0]) + \
                 '_wD_' + str(weight_decay_parameters[best_val_indx][0]) + '_dr_' + \
                 str(dropout_parameters[best_val_indx][0]) + '_mE_' + str(epochs) + '_rDC_' + \
                 str(args.ResidualDenseConnection) + '_pN_' + str(args.PairNorm) + '_DE_' + str(args.DropEdge) \
                 + '_DGN_' + str(args.DiffGroupNorm) + '_JLC_' + str(args.JostLiuCurvature) + '_JLCo_' + \
                 str(args.JostLiuCurvature_Online) + '_FA_' + str(args.FALayer) + '_GD_' + str(args.GraphDifussion) + \
                 '_RC_' + str(args.RicciCurvature)
if args.JostLiuCurvature:
    complement_file_name = '_pA_' + str(pA_parameters[best_val_indx][0]) + '_pD_' + str(pD_parameters[best_val_indx][0])\
                           + '_tau_' + str(tau_parameters[best_val_indx][0]) + '_alpha_' + \
                           str(alpha_parameters[best_val_indx][0])
    file_name_base = file_name_base + complement_file_name
if args.JostLiuCurvature_Online:
    complement_file_name = '_pA_' + str(pA_parameters[best_val_indx][0]) + '_pD_' + str(pD_parameters[best_val_indx][0])\
                           + '_alpha_' + str(alpha_parameters[best_val_indx][0])
    file_name_base = file_name_base + complement_file_name
if args.DropEdge:
    complement_file_name = '_pD_' + str(pD_parameters[best_val_indx][0])
    file_name_base = file_name_base + complement_file_name
if args.PairNorm:
    complement_file_name = '_s_' + str(s_parameters[best_val_indx][0])
    file_name_base = file_name_base + complement_file_name
if args.DiffGroupNorm:
    complement_file_name = '_c_' + str(int(clusters_parameters[best_val_indx][0])) + '_lam_' + \
                           str(lambda_parameters[best_val_indx][0])
    file_name_base = file_name_base + complement_file_name
if args.GraphDifussion:
    complement_file_name = '_alphaGDC_' + str(alphaGDC_parameters[best_val_indx][0]) + '_k_' + \
                           str(int(k_parameters[best_val_indx][0]))
    file_name_base = file_name_base + complement_file_name
if args.RicciCurvature:
    complement_file_name = '_tau_' + str(tau_parameters[best_val_indx][0]) + '_iRic_' + \
                           str(int(iterRicci_parameters[best_val_indx][0])) + '_c_' + str(c_parameters[best_val_indx][0])
    file_name_base = file_name_base + complement_file_name
file_name_results = file_name_base + '.pkl'
with open(file_name_results, 'rb') as f:
    loss_train_vec, loss_val_vec, loss_test_vec, best_acc_test_vec, err_test_vec, err_val_vec, \
    err_train_vec = pickle.load(f)
acc_test_vec_test_seeds = best_acc_test_vec[:, -1]
boots_series = sns.algorithms.bootstrap(acc_test_vec_test_seeds, func=np.mean, n_boot=1000)
test_std_test_seeds = np.max(np.abs(sns.utils.ci(boots_series, 95) - np.mean(acc_test_vec_test_seeds)))
results_log = 'The best result for rDC_' + str(args.ResidualDenseConnection) + '_pN_' + str(args.PairNorm) + \
              '_DE_' + str(args.DropEdge) + '_DGN_' + str(args.DiffGroupNorm) + '_JLC_' + \
              str(args.JostLiuCurvature) + '_JLCo_' + str(args.JostLiuCurvature_Online) + '_FA_' + str(args.FALayer) + \
              '_GD_' + str(args.GraphDifussion) + '_RC_' + str(args.RicciCurvature) + ' method with test seeds in ' + dataset + ' dataset is ' + \
              str(np.mean(boots_series)) + '+-' + str(test_std_test_seeds)
print(results_log)
