__author__ = "Jhony H. Giraldo"
__license__ = "MIT"

import numpy as np

import torch
import torch.nn.functional as functional

from .JLC_curvature import jost_liu_curvature, compute_jost_liu_curvature

def balanced_forman_curvature(edge_index, n, is_undirected):
    r"""This function computes the balanced Forman curvature for the input graph.
    The curvature metric is defined on "UNDERSTANDING OVER-SQUASHING AND BOTTLENECKS
    ON GRAPHS VIA CURVATURE" <https://arxiv.org/pdf/2111.14522.pdf>`_ paper

    Args:
        edge_index (LongTensor): The edge indices.
        n (int): Number of nodes.
        is_undirected (bool, optional): If set to :obj:`True`, will assume the graph is undirected
    """
    adj = torch.zeros((n, n))
    if is_undirected:
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1
    else:
        adj[edge_index[0], edge_index[1]] = 1
    num_edges = edge_index.shape[1]
    balaced_forman_vector = torch.zeros((num_edges,))
    for i in range(0, num_edges):
        edge = edge_index[:, i]
        node_x = edge[0]
        node_y = edge[1]
        balaced_forman_vector[i] = compute_balaced_forman_curvature(adj, node_x, node_y)
    return balaced_forman_vector


def compute_balaced_forman_curvature(adj, node_x, node_y, is_adj_sparse=False, s_x=None, s_y=None):
    r"""This function computes the balaced Forman curvature based on the nodes node_x and node_y.
    The curvature metric is defined on "UNDERSTANDING OVER-SQUASHING AND BOTTLENECKS
    ON GRAPHS VIA CURVATURE" <https://arxiv.org/pdf/2111.14522.pdf>`_ paper

    Args:
        adj (sparse LongTensor or LongTensor): The adjacency matrix of the graph.
        node_x (int): First node of the base edge.
        node_y (int): Second node of the base edge.
        graph_degrees (LongTensor): Vector with the degrees of each node.
        is_adj_sparse (bool): If set to :obj:`True`, the adj variable is a sparse tensor.
        s_x (LongTensor): Vector of neighbors of node_x
        s_y (LongTensor): Vector of neighbors of node_y
    """
    # Computes neighborhood of the base nodes when the input adj matrix is a in sparse representation
    if is_adj_sparse and (s_x is None) and (s_y is None):
        s_x = adj[node_x].coalesce().indices()[0]
        s_y = adj[node_y].coalesce().indices()[0]
    # Computes neighborhood of the base nodes when the input adj matrix is in dense representation
    elif (s_x is None) and (s_y is None):
        s_x = torch.where(adj[node_x] > 0)[0]
        s_y = torch.where(adj[node_y] > 0)[0]
    set_triangles = np.intersect1d(s_x, s_y)  # The set of triangles is computed as the intersection of the neighbors
    # Set of 4-cycles based on x
    set_k_node_x = np.setdiff1d(np.setdiff1d(s_x, s_y), node_y)
    elements_delete = []
    for node_k in set_k_node_x:
        if is_adj_sparse:
            s_k = adj[node_k].coalesce().indices()[0]
        else:
            s_k = torch.where(adj[node_k] > 0)[0]
        set_square_nodes = np.setdiff1d(np.intersect1d(s_k, s_y), node_x)
        if (set_square_nodes.shape[0] == 0) or np.isin(set_square_nodes, set_triangles).any():
            elements_delete.append(node_k)
    set_k_node_x = np.setdiff1d(set_k_node_x, elements_delete)
    # Set of 4-cycles based on y
    set_k_node_y = np.setdiff1d(np.setdiff1d(s_y, s_x), node_x)
    elements_delete = []
    for node_k in set_k_node_y:
        if is_adj_sparse:
            s_k = adj[node_k].coalesce().indices()[0]
        else:
            s_k = torch.where(adj[node_k] > 0)[0]
        set_square_nodes = np.setdiff1d(np.intersect1d(s_k, s_x), node_y)
        if (set_square_nodes.shape[0] == 0) or np.isin(set_square_nodes, set_triangles).any():
            elements_delete.append(node_k)
    set_k_node_y = np.setdiff1d(set_k_node_y, elements_delete)
    # of each based node
    number_triangles = set_triangles.shape[0]
    number_4_cycles_x = set_k_node_x.shape[0]
    number_4_cycles_y = set_k_node_y.shape[0]
    degree_x = torch.Tensor([s_x.shape[0]])
    degree_y = torch.Tensor([s_y.shape[0]])
    max_degree = torch.max(degree_x, degree_y)
    min_degree = torch.min(degree_x, degree_y)
    # Gamma max
    k_vector = np.zeros((number_4_cycles_x,))
    w_vector = np.zeros((number_4_cycles_y,))
    A_x = adj[node_x, :]
    A_y = adj[node_y, :]
    for k in range(0, number_4_cycles_x):
        node_k = set_k_node_x[k]
        A_k = adj[node_k, :]
        k_vector[k] = np.dot(A_k, A_y - (A_x * A_y)) - 1
    if k_vector.shape[0] > 0:
        max_k = max(k_vector)
    else:
        max_k = 0
    for w in range(0, number_4_cycles_y):
        node_w = set_k_node_y[w]
        A_w = adj[node_w, :]
        w_vector[w] = np.dot(A_w, A_x - (A_y * A_x)) - 1
    if w_vector.shape[0] > 0:
        max_w = max(w_vector)
    else:
        max_w = 0
    gamma_max = max(max_k, max_w)
    # First term of balanced Forman Curvature equation
    first_term = (2/degree_x) + (2/degree_y) - 2
    # Second term of balanced Forman Curvature equation
    second_term = 2*(number_triangles/max_degree) + (number_triangles/min_degree)
    # Third term of balanced Forman Curvature equation
    if gamma_max == 0:
        third_term = 0
    else:
        third_term = (number_4_cycles_x + number_4_cycles_y)/(gamma_max * max_degree)
    return first_term + second_term + third_term


def stochastic_discrete_ricci_flow_rewiring(edge_index, iterRicci, tau, c, n, n_edges, device, dataset_name,
                                            force_undirected=False):
    r"""This function rewires and input graph with the Stochastic Discrete Ricci Flow method as defined in
    "Understanding over-squashing and bottlenecks on graphs via curvature" <https://arxiv.org/pdf/2111.14522.pdf>`_ paper

        Args:
            edge_index (LongTensor): The edge indices.
            pA (float): Percentage of added edges.
            tau (float): Variable controlling how stochastic is the rewiring algorithm.
            n (int): Number of nodes in the graph.
            n_edges (int): Number of edges in the graph.
            device (str): Device we are using 'cpu' or 'cuda'.
            dataset_name (str): Dataset name.
            force_undirected: If set to :obj:`True`, we will force the output graph to be undirected.
        """
    update_curvature_module = 1000
    edge_index_original = edge_index.clone()
    if force_undirected:
        edge_index = edge_index.cpu()
        row, col = edge_index
        index_undirected = torch.where(row > col)[0]
        edge_index = edge_index[:, index_undirected]
    else:
        edge_index = edge_index.cpu()
        row, col = edge_index
    JLC = jost_liu_curvature(edge_index, n, force_undirected).cpu()
    # Construct adjacency matrix in sparse structure.
    Adj_row_col = torch.LongTensor([row.tolist(), col.tolist()])
    Adj = torch.sparse.LongTensor(Adj_row_col, torch.ones((n_edges,)))
    # Indices of edge_index in the adjacency matrix.
    Adj_index = torch.zeros((n, n)).cpu()
    for i in range(0, edge_index.shape[1]):
        Adj_index[edge_index[0, i], edge_index[1, i]] = i + 1
        if force_undirected:
            Adj_index[edge_index[1, i], edge_index[0, i]] = i + 1
    # Loop to add edges.
    for i in range(0, iterRicci):
        # flags for convergence
        flag_first_step = False
        flag_second_step = False
        #
        most_negative_curvature = torch.where(JLC == torch.min(JLC))
        most_negative_edge = edge_index[:, most_negative_curvature[0][0]]
        node_x = most_negative_edge[0].cpu()
        node_y = most_negative_edge[1].cpu()
        # This neighborhood includes 3-cycles (triangles) only
        S_x_original = Adj[node_x].coalesce().indices()[0]
        S_y_original = Adj[node_y].coalesce().indices()[0]
        indx_node_y = torch.where(S_x_original == node_y)
        S_x = np.delete(S_x_original, indx_node_y, axis=0)
        indx_node_x = torch.where(S_y_original == node_x)
        S_y = np.delete(S_y_original, indx_node_x, axis=0)
        S_x = torch.cartesian_prod(node_y.reshape((1,)), S_x)
        S_y = torch.cartesian_prod(S_y, node_x.reshape((1,)))
        car_prod_Sx_Sy = torch.cat((S_x, S_y), dim=0)
        # Remove existing edges, i.e., keep non-existing edges.
        indx_non_existing_edges = torch.where(Adj_index[car_prod_Sx_Sy[:, 0], car_prod_Sx_Sy[:, 1]] == 0)[0]
        car_prod_Sx_Sy = car_prod_Sx_Sy[indx_non_existing_edges]
        JLC_improvement = torch.zeros((car_prod_Sx_Sy.shape[0],))
        Adj_row_col = Adj.coalesce().indices()
        for j in range(0, car_prod_Sx_Sy.shape[0]):
            node_u = car_prod_Sx_Sy[j, 0]
            node_v = car_prod_Sx_Sy[j, 1]
            S_y_temp = S_y_original.clone()
            S_x_temp = S_x_original.clone()
            if node_u == node_y:
                S_y_temp = torch.cat((S_y_original, node_v.reshape((1,))))
            if node_v == node_x:
                S_x_temp = torch.cat((S_x_original, node_u.reshape((1,))))
            Adj_row_col_temp = Adj_row_col.clone()
            Adj_row_col_temp = torch.cat((Adj_row_col_temp, car_prod_Sx_Sy[j].reshape((2, 1))), dim=1)
            if force_undirected:
                Adj_row_col_temp = torch.cat((Adj_row_col_temp, torch.tensor(np.flip(car_prod_Sx_Sy[j].numpy(),
                                                                                     0).copy()).reshape((2, 1))), dim=1)
            Adj_temp = torch.sparse.LongTensor(Adj_row_col_temp, torch.ones((Adj_row_col_temp.shape[1],)))
            JLC_improvement[j] = compute_jost_liu_curvature(Adj_temp, node_x, node_y, is_adj_sparse=True, s_x=S_x_temp,
                                                            s_y=S_y_temp) - JLC[most_negative_curvature[0][0]]
        if np.where(JLC_improvement > 0)[0].shape[0] > 0:
            # We transform the curvatures improvements to probabilities.
            prob = functional.softmax(tau * JLC_improvement, dim=0)
            # We sample one edge according to the probability distribution prob.
            sampled_edge_index = np.random.choice(car_prod_Sx_Sy.shape[0], size=1, replace=False, p=np.array(prob))
            # We add the new edge to the graph.
            sampled_edge = car_prod_Sx_Sy[sampled_edge_index]
            if force_undirected:
                sampled_edge = torch.cat((sampled_edge, torch.IntTensor([[sampled_edge[0][1], sampled_edge[0][0]]])), dim=0)
            else:
                row, col = torch.cat((edge_index_original, sampled_edge.T), dim=1)
                Adj_row_col = torch.LongTensor([row.tolist(), col.tolist()])
            sampled_edge = sampled_edge.to(device)
            edge_index_original = torch.cat((edge_index_original, sampled_edge.T), dim=1)
            Adj_index[sampled_edge[0][0], sampled_edge[0][1]] = torch.max(torch.max(Adj_index)) + 1
            if force_undirected:
                row, col = edge_index_original.cpu()
                Adj_row_col = torch.LongTensor([row.tolist(), col.tolist()])
                Adj_index[sampled_edge[0][1], sampled_edge[0][0]] = torch.max(torch.max(Adj_index))
            edge_index = torch.cat((edge_index, torch.IntTensor([[sampled_edge[0][0], sampled_edge[0][1]]]).T), dim=1)
            Adj = torch.sparse.LongTensor(Adj_row_col, torch.ones((Adj_row_col.shape[1],)))
            new_JLC = compute_jost_liu_curvature(Adj, sampled_edge[0][0], sampled_edge[0][1], is_adj_sparse=True)
            JLC = torch.cat((JLC, new_JLC))
            #
            index_base_edge = int(Adj_index[node_x, node_y] - 1)
            JLC[index_base_edge] = compute_jost_liu_curvature(Adj, node_x, node_y, is_adj_sparse=True)
            # If we are working with the biggest datasets Pubmed or squirrel, we update the whole curvature vector
            # only each added edges module update_curvature_module.
            if (dataset_name in ['Pubmed', 'squirrel']) and ((i + 1) % update_curvature_module == 0):
                JLC = jost_liu_curvature(edge_index, n, force_undirected).cpu()
            elif dataset_name not in ['Pubmed', 'squirrel']:
                # Neighborhood of new edge
                B_u = Adj[sampled_edge[0][0]].coalesce().indices()[0]
                B_u = torch.cat((B_u, sampled_edge[0][0].reshape((1,)).cpu()))
                B_v = Adj[sampled_edge[0][1]].coalesce().indices()[0]
                B_v = torch.cat((B_v, sampled_edge[0][1].reshape((1,)).cpu()))
                car_prod_B_u_B_v = torch.cartesian_prod(B_u, B_v)
                # Remove non-existing edges to update the curvature, i.e., keep only existing edges.
                indx_existing_edges = torch.where(Adj_index[car_prod_B_u_B_v[:, 0], car_prod_B_u_B_v[:, 1]] > 0)[0]
                car_prod_B_u_B_v = car_prod_B_u_B_v[indx_existing_edges]
                for j in range(0, car_prod_B_u_B_v.shape[0]):
                    index_edge_original = int(Adj_index[car_prod_B_u_B_v[j, 0], car_prod_B_u_B_v[j, 1]] - 1)
                    edge_to_update = edge_index[:, index_edge_original]
                    JLC[index_edge_original] = compute_jost_liu_curvature(Adj, edge_to_update[0], edge_to_update[1],
                                                                          is_adj_sparse=True)
        else:
            flag_first_step = True
        most_positive_curvature = torch.where(JLC == torch.max(JLC))
        if JLC[most_positive_curvature[0][0]] > c:
            most_positive_edge = edge_index[:, most_positive_curvature[0][0]]
            edge_index = np.delete(edge_index, most_positive_curvature[0][0], axis=1)
            JLC = np.delete(JLC, most_positive_curvature[0][0], axis=0)
            most_positive_index = Adj_index[most_positive_edge[0], most_positive_edge[1]]
            indx_bigger = np.where(Adj_index >= most_positive_index)
            Adj_index[indx_bigger] = Adj_index[indx_bigger] - 1
        else:
            flag_second_step = True
        if flag_first_step and flag_second_step:
            return edge_index_original
    return edge_index_original
