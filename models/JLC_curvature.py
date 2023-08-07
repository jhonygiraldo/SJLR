__author__ = "Jhony H. Giraldo"
__license__ = "MIT"

import numpy as np

import torch
import torch.nn.functional as functional
from torch_geometric.utils import to_undirected


def jost_liu_curvature(edge_index, n, is_undirected):
    r"""This function computes the Jost & Liu curvature for the input graph.
    The curvature metric is defined on "Ollivier's Ricci curvature, local clustering and
    curvature dimension inequalities on graphs" <https://arxiv.org/pdf/1103.4037.pdf>`_ paper

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
    jost_liu_vector = torch.zeros((num_edges,))
    for i in range(0, num_edges):
        edge = edge_index[:, i]
        node_x = edge[0]
        node_y = edge[1]
        jost_liu_vector[i] = compute_jost_liu_curvature(adj, node_x, node_y)
    return jost_liu_vector


def compute_jost_liu_curvature(adj, node_x, node_y, is_adj_sparse=False, s_x=None, s_y=None):
    r"""This function computes the Jost & Liu curvature based on the nodes node_x and node_y.
    The curvature metric is defined on "Ollivier's Ricci curvature, local clustering and
    curvature dimension inequalities on graphs" <https://arxiv.org/pdf/1103.4037.pdf>`_ paper

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
    # of each based node
    number_triangles = set_triangles.shape[0]
    degree_x = torch.Tensor([s_x.shape[0]])
    degree_y = torch.Tensor([s_y.shape[0]])
    max_degree = torch.max(degree_x, degree_y)
    min_degree = torch.min(degree_x, degree_y)
    # First term of Jost Liu Curvature equation
    first_term = (1 - (1 / degree_x) - (1 / degree_y) - (number_triangles / min_degree))
    # Second term of Jost Liu Curvature equation
    second_term = (1 - (1 / degree_x) - (1 / degree_y) - (number_triangles / max_degree))
    return -1 * torch.max(torch.Tensor([0]), first_term) - 1 * torch.max(torch.Tensor([0]), second_term) + \
           (number_triangles / max_degree)


def compute_JLC_matrices(edge_index, n, n_edges, pA, force_undirected=False):
    edge_index_original = edge_index.clone()
    if force_undirected:
        edge_index = edge_index_original.cpu()
        row, col = edge_index
        index_undirected = torch.where(row > col)[0]
        edge_index = edge_index[:, index_undirected]
        row, col = edge_index
    else:
        edge_index = edge_index.cpu()
        row, col = edge_index
    JLC = jost_liu_curvature(edge_index, n, force_undirected).cpu()
    # Construct matrix_JLC in sparse structure.
    Adj_row_col = torch.LongTensor([row.tolist(), col.tolist()])
    matrix_JLC = torch.sparse.LongTensor(Adj_row_col, JLC)
    #matrix_JLC = torch.sparse.LongTensor(Adj_row_col, JLC)
    #if force_undirected:
        #Adj_row_col = torch.cat((Adj_row_col, torch.LongTensor([col.tolist(), row.tolist()])), dim=1)
        #JLC_temp = torch.cat((JLC, JLC), dim=0)
        #matrix_JLC_temp = torch.sparse.LongTensor(Adj_row_col, JLC_temp)
        #matrix_JLC_dense = matrix_JLC_temp.to_dense()
    #else:
        #matrix_JLC_dense = matrix_JLC.to_dense()
    # Construct adjacency matrix in sparse structure.
    if force_undirected:
        row, col = edge_index_original.cpu()
        Adj_row_col = torch.LongTensor([row.tolist(), col.tolist()])
    Adj = torch.sparse.LongTensor(Adj_row_col, torch.ones((n_edges,)))
    Adj_dense = Adj.to_dense()
    if force_undirected:
        row, col = edge_index.cpu()
        Adj_row_col = torch.cat((edge_index, torch.LongTensor([col.tolist(), row.tolist()])), dim=1)
        JLC_temp = torch.cat((JLC, JLC), dim=0)
        matrix_JLC_temp = torch.sparse.LongTensor(Adj_row_col, JLC_temp)
        matrix_JLC_dense = matrix_JLC_temp.to_dense()
    else:
        matrix_JLC_dense = matrix_JLC.to_dense()
    # Search for the potential good nodes
    JLC_sorted = torch.sort(JLC)
    potential_good_edges_to_add = torch.LongTensor([])
    Adj_complement = torch.zeros((n, n))
    cont = 0
    # TODO: This loop should be refined for directed graphs because here we assume the graph is undirected
    while potential_good_edges_to_add.shape[0] < 2*pA*n_edges:
        curved_edge = edge_index[:, JLC_sorted.indices[cont]]
        node_x = curved_edge[0].cpu()
        node_y = curved_edge[1].cpu()
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
        # See if these edges are in the set of original edges
        existing_edges = Adj_dense[car_prod_Sx_Sy[:,0], car_prod_Sx_Sy[:,1]]
        index_existing_edges = torch.where(existing_edges == 1)
        car_prod_Sx_Sy = np.delete(car_prod_Sx_Sy.cpu(), index_existing_edges[0], axis=0)
        # See if these edges are already added in potential_good_edges_to_add
        existing_edges_in_good_edges = Adj_complement[car_prod_Sx_Sy[:, 0], car_prod_Sx_Sy[:, 1]]
        index_existing_edges = torch.where(existing_edges_in_good_edges == 1)
        car_prod_Sx_Sy = np.delete(car_prod_Sx_Sy.cpu(), index_existing_edges[0], axis=0)
        potential_good_edges_to_add = torch.cat((potential_good_edges_to_add, car_prod_Sx_Sy), dim=0)
        Adj_complement[car_prod_Sx_Sy[:, 0], car_prod_Sx_Sy[:, 1]] = 1
        cont += 1
    # Construct matrix_JLCc in sparse structure.
    matrix_JLCc_temp = torch.zeros((n, n))
    for i, edges_complement in enumerate(potential_good_edges_to_add):
        Adj_row_col_temp = Adj_row_col.clone()
        Adj_row_col_temp = torch.cat((Adj_row_col_temp, edges_complement.reshape((2, 1))), dim=1)
        if force_undirected:
            Adj_row_col_temp = to_undirected(Adj_row_col_temp)
        Adj_temp = torch.sparse.LongTensor(Adj_row_col_temp, torch.ones((Adj_row_col_temp.shape[1],)))
        # Look for edges that form a triangle with the new edge
        S_x_original = Adj[edges_complement[0]].coalesce().indices()[0]
        S_y_original = Adj[edges_complement[1]].coalesce().indices()[0]
        set_triangles = torch.LongTensor(np.intersect1d(S_x_original, S_y_original))  # The set of triangles is computed as the intersection of the neighbors
        S_x = torch.cartesian_prod(edges_complement[0].reshape((1,)), set_triangles)
        S_y = torch.cartesian_prod(edges_complement[1].reshape((1,)), set_triangles)
        edges_with_triangles = torch.cat((S_x, S_y), dim=0)
        JLC_temp_improvement = torch.zeros((edges_with_triangles.shape[0],))
        for j, edge_to_process in enumerate(edges_with_triangles):
            JLC_temp_improvement[j] = compute_jost_liu_curvature(Adj_temp, edge_to_process[0],
                                                                 edge_to_process[1], is_adj_sparse=True)
        '''
        S_x_original = Adj[edges_complement[0]].coalesce().indices()[0]
        S_y_original = Adj[edges_complement[1]].coalesce().indices()[0]
        S_x = torch.cartesian_prod(edges_complement[0].reshape((1,)), S_x_original)
        S_y = torch.cartesian_prod(edges_complement[1].reshape((1,)), S_y_original)
        edges_in_neighborhood = torch.cat((S_x, S_y), dim=0)
        JLC_temp_improvement = torch.zeros((edges_in_neighborhood.shape[0],))
        for j, edge_to_process in enumerate(edges_in_neighborhood):
            JLC_temp_improvement[j] = compute_jost_liu_curvature(Adj_temp, edge_to_process[0],
                                                                 edge_to_process[1], is_adj_sparse=True)
        '''
        JLC_temp_improvement = JLC_temp_improvement - matrix_JLC_dense[edges_with_triangles.T[0], edges_with_triangles.T[1]]
        matrix_JLCc_temp[edges_complement[0], edges_complement[1]] = torch.mean(JLC_temp_improvement)
        if force_undirected:
            matrix_JLCc_temp[edges_complement[1], edges_complement[0]] = matrix_JLCc_temp[
                edges_complement[0], edges_complement[1]]
    # Construct matrix_JLCc in sparse structure.
    matrix_JLCc = torch.sparse.LongTensor(potential_good_edges_to_add.T, matrix_JLCc_temp[potential_good_edges_to_add.T[0], potential_good_edges_to_add.T[1]])
    return matrix_JLC, potential_good_edges_to_add.T, matrix_JLCc_temp[potential_good_edges_to_add.T[0], potential_good_edges_to_add.T[1]]


def jost_liu_curvature_adding(edge_index, pA, tau, n, n_edges, device, dataset_name, force_undirected=False):
    r"""This function adds edges based on the Jost & Liu curvature of the edges.

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
    # Number of added edges after which we update the Jost & Liu curvature for Pubmed and squirrel datasets.
    update_curvature_module = 1000
    edge_index_original = edge_index.clone()
    if force_undirected:
        n_edges2add = int(n_edges/2 * pA)
    else:
        n_edges2add = int(n_edges * pA)
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
    for i in range(0, n_edges2add):
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
        # Compute the improvement in the Jost & Liu curvature by adding the edges in car_prod_Sx_Sy.
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
        # We transform the curvatures improvements to probabilities.
        prob = functional.softmax(tau*JLC_improvement, dim=0)
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
        # We update the Jost & Liu curvature of the edges when required.
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
                index_edge_original = int(Adj_index[car_prod_B_u_B_v[j,0], car_prod_B_u_B_v[j,1]] - 1)
                edge_to_update = edge_index[:, index_edge_original]
                JLC[index_edge_original] = compute_jost_liu_curvature(Adj, edge_to_update[0], edge_to_update[1],
                                                                      is_adj_sparse=True)
    # When we are done adding edges, we normalize the curvature between 0 and 1 to later remove some edges on the fly.
    mask_JLC = JLC - torch.min(JLC)
    mask_JLC = mask_JLC / torch.max(mask_JLC)
    return mask_JLC, edge_index, edge_index_original


def jost_liu_curvature_dropping(edge_index, mask_jost_liu_curvature, x, number_edges2drop, alpha,
                                device, force_undirected=False):
    r"""This function drops edges based on the Jost & Liu curvature of the edges, as well as the
    euclidean distance between the nodes of the respective edge.

    Args:
        edge_index (LongTensor): The edge indices.
        mask_jost_liu_curvature (LongTensor): Vector with the curvatures of the set of edges.
        x (LongTensor): Matrix of features or node embeddings.
        number_edges2drop (int): Number of edges that we will drop.
        alpha (float): Parameter that tradeoff curvature and euclidean distances.
        device (str): Device we are using 'cpu' or 'cuda'
        force_undirected: If set to :obj:`True`, we will force the output graph to be undirected.
    """
    euclidean_distances = torch.norm(x[edge_index[0], :] - x[edge_index[1], :], p=2, dim=1)
    # Normalization of the Euclidean distances to be between 0 and 1
    euclidean_distances = euclidean_distances - torch.min(euclidean_distances)
    euclidean_distances = euclidean_distances / (torch.max(euclidean_distances) + 1e-5)
    euclidean_distances = euclidean_distances.cpu()
    # Tradeoff between the Jost & Liu curvature and the Euclidean distance of the node embeddings
    mask_dropping = alpha * mask_jost_liu_curvature.squeeze() + (1 - alpha) * euclidean_distances
    probabilities = functional.softmax(mask_dropping, dim=0)
    # Drop edges based on the probability distribution given by the curvature and the Euclidean distances
    edges2remove = np.random.choice(edge_index.shape[1], size=number_edges2drop, replace=False,
                                    p=probabilities.detach().numpy())
    new_edges = np.delete(edge_index.cpu(), edges2remove, axis=1).to(device)
    if force_undirected:
        new_edges = to_undirected(new_edges)
    edge_index = new_edges.to(device)
    return edge_index

'''
def JLC_adding_dropping(JLC_indices, dropping_vector_stochastic, JLCc_indices, adding_vector_stochastic,
                        number_edges2drop, number_edges2add, device, force_undirected=False):
    # Drop edges based on the probability distribution given by the curvature and the Euclidean distances
    edge_index = JLC_indices
    edges2remove = np.random.choice(edge_index.shape[1], size=number_edges2drop, replace=False,
                                    p=dropping_vector_stochastic.detach().numpy())
    new_edges = np.delete(edge_index.cpu(), edges2remove, axis=1).to(device)
    # Add edges based on the probability distribution given by the curvature and the Euclidean distances
    edges2add_indices = np.random.choice(JLCc_indices.shape[1], size=number_edges2add, replace=False,
                                         p=adding_vector_stochastic.detach().numpy())
    edges2add = JLCc_indices[:, edges2add_indices]
    new_edges = torch.cat((new_edges, edges2add), dim=1)
    if force_undirected:
        new_edges = to_undirected(new_edges)
    edge_index = new_edges.to(device)
    return edge_index
'''


def JLC_adding(x, JLCc_indices, JLCc_values_normalized, number_edges2add, alpha, device, force_undirected=False):
    euclidean_distances = torch.norm(x[JLCc_indices[0], :] - x[JLCc_indices[1], :], p=2, dim=1)
    # Normalization of the Euclidean distances to be between 0 and 1
    euclidean_distances = euclidean_distances - torch.min(euclidean_distances)
    euclidean_distances = euclidean_distances / (torch.max(euclidean_distances) + 1e-5)
    euclidean_distances = euclidean_distances.cpu()
    mask_adding = alpha * JLCc_values_normalized + (1 - alpha) * euclidean_distances
    adding_vector_stochastic = functional.softmax(mask_adding, dim=0)
    edges2add_indices = np.random.choice(JLCc_indices.shape[1], size=number_edges2add, replace=False,
                                         p=adding_vector_stochastic.detach().numpy())
    edges2add = JLCc_indices[:, edges2add_indices]
    if force_undirected:
        edges2add = to_undirected(edges2add)
    edges2add = edges2add.to(device)
    return edges2add


'''
def JLC_dropping(x, JLC_indices, JLC_values_normalized, number_edges2drop, alpha, device, force_undirected=False):
    # Drop edges based on the probability distribution given by the curvature and the Euclidean distances
    edge_index = JLC_indices
    euclidean_distances = torch.norm(x[JLC_indices[0], :] - x[JLC_indices[1], :], p=2, dim=1)
    # Normalization of the Euclidean distances to be between 0 and 1
    euclidean_distances = euclidean_distances - torch.min(euclidean_distances)
    euclidean_distances = euclidean_distances / (torch.max(euclidean_distances) + 1e-5)
    euclidean_distances = euclidean_distances.cpu()
    mask_dropping = alpha * JLC_values_normalized + (1 - alpha) * euclidean_distances
    dropping_vector_stochastic = functional.softmax(mask_dropping, dim=0)
    edges2remove = np.random.choice(edge_index.shape[1], size=number_edges2drop, replace=False,
                                    p=dropping_vector_stochastic.detach().numpy())
    new_edges = np.delete(edge_index.cpu(), edges2remove, axis=1).to(device)
'''


def JLC_adding_dropping(x, JLC_indices, JLC_values_normalized, JLCc_indices, JLCc_values_normalized,
                        number_edges2drop, number_edges2add, alpha, device, force_undirected=False):
    # Drop edges based on the probability distribution given by the curvature and the Euclidean distances
    edge_index = JLC_indices
    euclidean_distances = torch.norm(x[JLC_indices[0], :] - x[JLC_indices[1], :], p=2, dim=1)
    # Normalization of the Euclidean distances to be between 0 and 1
    euclidean_distances = euclidean_distances - torch.min(euclidean_distances)
    euclidean_distances = euclidean_distances / (torch.max(euclidean_distances) + 1e-5)
    #euclidean_distances = euclidean_distances.cpu()
    mask_dropping = alpha * JLC_values_normalized + (1 - alpha) * euclidean_distances
    dropping_vector_stochastic = functional.softmax(mask_dropping, dim=0)
    edges2remove = np.random.choice(edge_index.shape[1], size=number_edges2drop, replace=False,
                                    p=dropping_vector_stochastic.cpu().detach().numpy())
    new_edges = np.delete(edge_index.cpu(), edges2remove, axis=1).to(device)
    # Add edges based on the probability distribution given by the curvature and the Euclidean distances
    euclidean_distances = torch.norm(x[JLCc_indices[0], :] - x[JLCc_indices[1], :], p=2, dim=1)
    # Normalization of the Euclidean distances to be between 0 and 1
    euclidean_distances = euclidean_distances - torch.min(euclidean_distances)
    euclidean_distances = euclidean_distances / (torch.max(euclidean_distances) + 1e-5)
    #euclidean_distances = euclidean_distances.cpu()
    mask_adding = alpha * JLCc_values_normalized - (1 - alpha) * euclidean_distances
    adding_vector_stochastic = functional.softmax(mask_adding, dim=0)
    edges2add_indices = np.random.choice(JLCc_indices.shape[1], size=number_edges2add, replace=False,
                                         p=adding_vector_stochastic.cpu().detach().numpy())
    edges2add = JLCc_indices[:, edges2add_indices]
    new_edges = torch.cat((new_edges, edges2add), dim=1)
    if force_undirected:
        new_edges = to_undirected(new_edges)
    edge_index = new_edges.to(device)
    return edge_index
