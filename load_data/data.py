__author__ = "Stefan Weißenberger and Johannes Klicpera, modified by Jhony H. Giraldo"
__license__ = "MIT"

import os

import numpy as np
import pickle

import torch
from torch_geometric.data import Data, InMemoryDataset
#from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WebKB, WikipediaNetwork, Actor, LRGBDataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WebKB, WikipediaNetwork, Actor

from .seeds import development_seed


DATA_PATH = 'data'


def get_dataset(name: str, use_lcc: bool = True) -> InMemoryDataset:
    path = os.path.join(DATA_PATH, name)
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(path, name)
    elif name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(path, name)
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(path, name)
    elif name in ['Actor']:
        dataset = Actor(path)
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(path, name)
    elif name == 'CoauthorCS':
        dataset = Coauthor(path, 'CS')
    #elif name in ['PascalVOC-SP', 'COCO-SP']:
    #    dataset = LRGBDataset(path, name)
    else:
        raise Exception('Unknown dataset.')

    if use_lcc:
        lcc = get_largest_connected_component(dataset)

        x_new = dataset.data.x[lcc]
        y_new = dataset.data.y[lcc]

        row, col = dataset.data.edge_index.numpy()
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        edges = remap_edges(edges, get_node_mapper(lcc))
        
        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
        )
        dataset.data = data

    return dataset


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


def set_train_val_test_split(
        seed: int,
        data: Data,
        dataset_name: str = 'Cora',
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
    rnd_state = np.random.RandomState(development_seed)
    num_nodes = data.y.shape[0]
    development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        for c in range(data.y.max() + 1):
            class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
            train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))
    else:
        num_train = int(0.6*num_nodes)
        train_idx.extend(rnd_state.choice(development_idx, num_train, replace=False))

    val_idx = [i for i in development_idx if i not in train_idx]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data


def get_adj_matrix(dataset: InMemoryDataset) -> np.ndarray:
    num_nodes = dataset.data.x.shape[0]
    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
    for i, j in zip(dataset.data.edge_index[0], dataset.data.edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix


def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.1) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)


def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm


def get_clipped_matrix(A: np.ndarray, eps: float = 0.01) -> np.ndarray:
    num_nodes = A.shape[0]
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm


class PPRDataset(InMemoryDataset):
    """
    Dataset preprocessed with GDC using PPR diffusion.
    Note that this implementations is not scalable
    since we directly invert the adjacency matrix.
    """
    def __init__(self,
                 name: str = 'Cora',
                 use_lcc: bool = True,
                 alpha: float = 0.1,
                 k: int = 16,
                 eps: float = None):
        self.name = name
        self.use_lcc = use_lcc
        self.alpha = alpha
        self.k = k
        self.eps = eps

        super(PPRDataset, self).__init__(DATA_PATH)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list:
        return []

    @property
    def processed_file_names(self) -> list:
        return [str(self) + '.pt']

    def download(self):
        pass

    def process(self):
        if self.name in ['chameleon', 'squirrel', 'Actor']:
            file_name = 'data/' + self.name + '.pkl'
            with open(file_name, 'rb') as f:
                base = pickle.load(f)
            base = base[0]
        else:
            base = get_dataset(name=self.name, use_lcc=self.use_lcc)
        # base = get_dataset(name=self.name, use_lcc=self.use_lcc)
        # generate adjacency matrix from sparse representation
        adj_matrix = get_adj_matrix(base)
        # obtain exact PPR matrix
        ppr_matrix = get_ppr_matrix(adj_matrix,
                                        alpha=self.alpha)

        if self.k:
            print(f'Selecting top {self.k} edges per node.')
            ppr_matrix = get_top_k_matrix(ppr_matrix, k=self.k)
        elif self.eps:
            print(f'Selecting edges with weight greater than {self.eps}.')
            ppr_matrix = get_clipped_matrix(ppr_matrix, eps=self.eps)
        else:
            raise ValueError

        # create PyG Data object
        edges_i = []
        edges_j = []
        edge_attr = []
        for i, row in enumerate(ppr_matrix):
            for j in np.where(row > 0)[0]:
                edges_i.append(i)
                edges_j.append(j)
                edge_attr.append(ppr_matrix[i, j])
        edge_index = [edges_i, edges_j]

        data = Data(
            x=base.data.x,
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_attr),
            y=base.data.y,
            train_mask=torch.zeros(base.data.train_mask.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(base.data.test_mask.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(base.data.val_mask.size()[0], dtype=torch.bool)
        )

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __str__(self) -> str:
        return f'{self.name}_ppr_alpha={self.alpha}_k={self.k}_eps={self.eps}_lcc={self.use_lcc}'
