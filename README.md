# On the Trade-off between Over-smoothing and Over-squashing in Deep Graph Neural Networks

This is the repository of the paper "On the Trade-off between Over-smoothing and Over-squashing in Deep Graph Neural Networks" published in **ACM CIKM 2023**

- - - -
![Pipeline](https://github.com/jhonygiraldo/SJLR/blob/master/misc/oversquashing_oversmoothing.png)
- - - -
Authors: [Jhony H Giraldo](https://sites.google.com/view/jhonygiraldo), [Konstantinos Skianis](http://y3nk0.github.io/), [Thierry Bouwmans](https://sites.google.com/site/thierrybouwmans/), and [Fragkiskos D. Malliaros](https://fragkiskos.me/)
- - - -
**Abstract**: Graph Neural Networks (GNNs) have shown success in various computer science applications. However, while deep learning architectures have excelled in other domains, deep GNNs still underperform their shallow counterparts. Over-smoothing and over-squashing are two main challenges when stacking multiple graph convolutional layers, where GNNs struggle to learn deep representations and propagate information from distant nodes. Our work reveals that over-smoothing and over-squashing are intrinsically related to the spectral gap of the graph Laplacian. Consequently, there is a trade-off between these two issues as it is impossible to alleviate both simultaneously. We argue that a viable approach is to add and remove edges in order to achieve a suitable compromise. To this end, we propose the Stochastic Jost and Liu Curvature Rewiring (SJLR) algorithm, which offers a less computationally expensive solution compared to previous curvature-based rewiring methods, while preserving fundamental properties. Unlike existing methods, SJLR performs edge addition and removal during the training phase of GNNs, while keeping the graph unchanged during testing. A comprehensive comparison of SJLR with previous techniques for addressing over-smoothing and over-squashing is performed, where the proposed algorithm shows competitive performance.
- - - -

#### Installation

Clone this repository
```bash
git clone https://github.com/jhonygiraldo/SJLR
```
Inside the repository, create the following folders
```bash
mkdir hyperparameters_tuning hyperparameters_tuning/GCN hyperparameters_tuning/GCN_hyperTuning
mkdir results results/GCN results/GCN_Deep
```
Install the requirements.

- - - -
## Citation

If you use our code, please cite

        @inproceedings{giraldo2023trade-off,
          title={On the Trade-off between Over-smoothing and Over-squashing in Deep Graph Neural Networks},
          author={Giraldo, Jhony H and Skianis, Konstantinos and Bouwmans, Thierry and Malliaros, Fragkiskos D},
          booktitle={ACM International Conference on Information and Knowledge Management},
          year={2023}
        }
