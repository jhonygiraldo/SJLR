# On the Trade-off between Over-smoothing and Over-squashing in Deep Graph Neural Networks

This is the repository of the paper "On the Trade-off between Over-smoothing and Over-squashing in Deep Graph Neural Networks" published in **ACM CIKM 2023**

Authors: [Jhony H Giraldo](https://sites.google.com/view/jhonygiraldo), [Konstantinos Skianis](http://y3nk0.github.io/), [Thierry Bouwmans](https://sites.google.com/site/thierrybouwmans/), and [Fragkiskos D. Malliaros](https://fragkiskos.me/)

<p align="center">
  <img width=500 src="./misc/oversquashing_oversmoothing.png?sanitize=true" />
</p>

**Abstract**: Graph Neural Networks (GNNs) have shown success in various computer science applications. However, while deep learning architectures have excelled in other domains, deep GNNs still underperform their shallow counterparts. Over-smoothing and over-squashing are two main challenges when stacking multiple graph convolutional layers, where GNNs struggle to learn deep representations and propagate information from distant nodes. Our work reveals that over-smoothing and over-squashing are intrinsically related to the spectral gap of the graph Laplacian. Consequently, there is a trade-off between these two issues as it is impossible to alleviate both simultaneously. We argue that a viable approach is to add and remove edges in order to achieve a suitable compromise. To this end, we propose the Stochastic Jost and Liu Curvature Rewiring (SJLR) algorithm, which offers a less computationally expensive solution compared to previous curvature-based rewiring methods, while preserving fundamental properties. Unlike existing methods, SJLR performs edge addition and removal during the training phase of GNNs, while keeping the graph unchanged during testing. A comprehensive comparison of SJLR with previous techniques for addressing over-smoothing and over-squashing is performed, where the proposed algorithm shows competitive performance.

## Installation

Clone this repository
```bash
git clone https://github.com/jhonygiraldo/SJLR
```

Install the requirements: pytorch, pytorch geometric (PyG), seaborn, and standard libraries of python like numpy and pickle. If you want to execute the script JLC_vs_BFC.py, please also install networkx.

## Replicate Results

If you are gonna run extensive experiments with SJLR, **we advise you to run first the script** precompute_JLC_metrics.py for each dataset you want to use. For example, for Texas you can run:

```bash
python precompute_JLC_metrics.py --JostLiuCurvature_Online --dataset Texas
```

If you want to reproduce the results in Table 2 for the Texas dataset, please run:

```bash
python main.py --verbose --lr 0.0177 --weight_decay 0.0003 --hidden_units 64 --n_layers_set 2 --dropout 0.5864 --JostLiuCurvature_Online --pA 0.4002 --pD 0.9487 --alpha 0.9005 --dataset Texas --GNN SGC
```

## Running Hyperparameter Optimization

If you want to run random hyperparameter search with SJLR for the Wisconsin dataset, please run:

```bash
python random_search_hyperparameters.py --JostLiuCurvature_Online --dataset Wisconsin --GNN SGC
```

To compute the test result with the best results from the validation set:

```bash
python compute_test_accuracy.py --JostLiuCurvature_Online --dataset Wisconsin --GNN SGC
```

## Citation

If you use our code, please cite

        @inproceedings{giraldo2023trade,
          title={On the Trade-off between Over-smoothing and Over-squashing in Deep Graph Neural Networks},
          author={Giraldo, Jhony H and Skianis, Konstantinos and Bouwmans, Thierry and Malliaros, Fragkiskos D},
          booktitle={ACM International Conference on Information and Knowledge Management},
          year={2023}
        }
        
## Acknowledgements

This work was supported by the DATAIA Institute as part of the "Programme d'Investissement d'Avenir", (ANR-17-CONV-0003) operated by CentraleSupelec, and by ANR (French National Research Agency) under the JCJC project GraphIA (ANR-20-CE23-0009-01).

We have used and modified some functions from [GDC](https://github.com/gasteigerjo/gdc)
