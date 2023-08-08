__author__ = "Masked_for_double_blind_review"
__license__ = "MIT"

import torch
import torch.nn.functional as functional

from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv, SGConv
from torch_geometric.nn.norm import PairNorm, DiffGroupNorm

from .JLC_curvature import JLC_adding_dropping


class GCN(torch.nn.Module):
    r"""Parametrized GNN using Graph Convolutions from Kipf and Welling paper `"Semi-supervised
    Classification with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        args (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(GCN, self).__init__()

        self.ResidualDenseConnection = args.ResidualDenseConnection
        self.PairNorm = args.PairNorm
        self.DropEdge = args.DropEdge
        self.DiffGroupNorm = args.DiffGroupNorm
        self.JostLiuCurvature_Online = args.JostLiuCurvature_Online
        self.FALayer = args.FALayer
        self.is_undirected = args.undirected
        self.pD = args.pD
        self.alpha = args.alpha
        self.device = args.device

        if args.JostLiuCurvature_Online:
            self.JLC_values_normalized = args.JLC_values_normalized
            self.JLC_indices = args.JLC_indices
            self.JLCc_values_normalized = args.JLCc_values_normalized
            self.JLCc_indices = args.JLCc_indices
            self.number_edges2drop = args.number_edges2drop
            self.number_edges2add = args.number_edges2add

        self.convs = torch.nn.ModuleList()
        if number_layers > 1:
            self.convs.append(GCNConv(in_channels, args.hidden_units, cached=False))
        if args.PairNorm:
            self.pns = torch.nn.ModuleList()
            self.pns.append(PairNorm(scale=args.s))
        if args.DiffGroupNorm:
            self.dgn = torch.nn.ModuleList()
            self.dgn.append(DiffGroupNorm(args.hidden_units, groups=args.clusters, lamda=args.lambda_p))
        for _ in range(number_layers - 2):
            self.convs.append(GCNConv(args.hidden_units, args.hidden_units, cached=False))
            if args.PairNorm:
                self.pns.append(PairNorm(scale=args.s))
            if args.DiffGroupNorm:
                self.dgn.append(DiffGroupNorm(args.hidden_units, groups=args.clusters, lamda=args.lambda_p))
        if number_layers > 1:
            self.convs.append(GCNConv(args.hidden_units, out_channels, cached=False))
        else:
            self.convs.append(GCNConv(in_channels, out_channels, cached=False))

        self.dropout = args.dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.ResidualDenseConnection:
            h_res_dens = 0
        for i, conv in enumerate(self.convs[:-1]):
            if self.JostLiuCurvature_Online and self.training:
                edge_index = JLC_adding_dropping(x, self.JLC_indices, self.JLC_values_normalized, self.JLCc_indices,
                                                 self.JLCc_values_normalized, self.number_edges2drop,
                                                 self.number_edges2add, self.alpha, self.device,
                                                 force_undirected=self.is_undirected)
            if self.DropEdge and self.training:
                edge_index, edge_attr = data.edge_index, data.edge_attr
                edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.pD, force_undirected=self.is_undirected)
            x = conv(x, edge_index, edge_weight=edge_attr)
            if self.PairNorm:
                x = self.pns[i](x)
            if self.DiffGroupNorm:
                x = self.dgn[i](x)
            x = functional.relu(x)
            if self.ResidualDenseConnection:
                x = x + h_res_dens
                h_res_dens = x
            x = functional.dropout(x, p=self.dropout, training=self.training)
        if self.JostLiuCurvature_Online and self.training:
            edge_index = JLC_adding_dropping(x, self.JLC_indices, self.JLC_values_normalized, self.JLCc_indices,
                                             self.JLCc_values_normalized, self.number_edges2drop,
                                             self.number_edges2add, self.alpha, self.device,
                                             force_undirected=self.is_undirected)
        if self.DropEdge and self.training:
            edge_index, edge_attr = data.edge_index, data.edge_attr
            edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.pD, force_undirected=self.is_undirected)
        if self.FALayer:
            num_nodes = x.shape[0]
            edge_index = torch.cartesian_prod(torch.arange(0,num_nodes), torch.arange(0,num_nodes)).T
            edge_index = edge_index.to(self.device)
        x = self.convs[-1](x, edge_index, edge_weight=edge_attr)
        return x.log_softmax(dim=-1)


class SGC(torch.nn.Module):
    r"""Parametrized GNN using the SGC model

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        number_layers (int): Number of layers of the GCN.
        args (Namespace): Arguments.
    """
    def __init__(self, in_channels, out_channels, number_layers, args):
        super(SGC, self).__init__()

        self.PairNorm = args.PairNorm
        self.DropEdge = args.DropEdge
        self.DiffGroupNorm = args.DiffGroupNorm
        self.JostLiuCurvature_Online = args.JostLiuCurvature_Online
        self.FALayer = args.FALayer
        self.is_undirected = args.undirected
        self.pD = args.pD
        self.alpha = args.alpha
        self.device = args.device

        if args.JostLiuCurvature_Online:
            self.JLC_values_normalized = args.JLC_values_normalized
            self.JLC_indices = args.JLC_indices
            self.JLCc_values_normalized = args.JLCc_values_normalized
            self.JLCc_indices = args.JLCc_indices
            self.number_edges2drop = args.number_edges2drop
            self.number_edges2add = args.number_edges2add

        self.conv = SGConv(in_channels=in_channels, out_channels=out_channels, K=number_layers)
        if args.PairNorm:
            self.pns = PairNorm(scale=args.s)
        if args.DiffGroupNorm:
            self.dgn = DiffGroupNorm(out_channels, groups=args.clusters, lamda=args.lambda_p)

        self.dropout = args.dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.JostLiuCurvature_Online and self.training:
            edge_index = JLC_adding_dropping(x, self.JLC_indices, self.JLC_values_normalized, self.JLCc_indices,
                                             self.JLCc_values_normalized, self.number_edges2drop,
                                             self.number_edges2add, self.alpha, self.device,
                                             force_undirected=self.is_undirected)
        if self.DropEdge and self.training:
            edge_index = data.edge_index
            edge_index = dropout_adj(edge_index, p=self.pD, force_undirected=self.is_undirected)
            edge_index = edge_index[0]
        x = self.conv(x, edge_index, edge_weight=edge_attr)
        if self.PairNorm:
            x = self.pns(x)
        if self.DiffGroupNorm:
            x = self.dgn(x)
        x = functional.relu(x)
        x = functional.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)
