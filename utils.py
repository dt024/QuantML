from collections import defaultdict
import torch
import networkx as nx
import fasthare as m
import random
import json
from torch_scatter import scatter_add
from torch_geometric.data import Data

from itertools import chain, islice, combinations

def generate_random_ising(n, d, emin, emax):
    """Generate random Ising Hamiltonian
    """
    p = d*1.0/(n - 1)
    G = nx.fast_gnp_random_graph(n, p, seed = 0, directed = False)

    # Random generator to choose random integer between emin and emax
    rndw = lambda : random.randrange(emin, emax + 1)
    h = { u: rndw() for u in range(n)}
    J = {(u, v): rndw() for u, v in G.edges()}
    return h, J

def gen_combinations(combs, chunk_size):
    yield from iter(lambda: list(islice(combs, chunk_size)), [])

# Calculate results given bitstring and graph definition, includes check for violations
def postprocess_gnn_mis(best_bitstring, nx_graph):
    """
    helper function to postprocess MIS results

    Input:
        best_bitstring: bitstring as torch tensor
    Output:
        size_mis: Size of MIS (int)
        ind_set: MIS (list of integers)
        number_violations: number of violations of ind.set condition
    """

    # get bitstring as list
    bitstring_list = list(best_bitstring)

    # compute cost
    size_mis = sum(bitstring_list)

    # get independent set
    ind_set = set([node for node, entry in enumerate(bitstring_list) if entry == 1])
    edge_set = set(list(nx_graph.edges))

    print('Calculating violations...')
    # check for violations
    number_violations = 0
    for ind_set_chunk in gen_combinations(combinations(ind_set, 2), 100000):
        number_violations += len(set(ind_set_chunk).intersection(edge_set))

    return size_mis, ind_set, number_violations


def generate_graph(n, d=None, p=None, graph_type='reg', random_seed=0):
    """
    Helper function to generate a NetworkX random graph of specified type,
    given specified parameters (e.g. d-regular, d=3). Must provide one of
    d or p, d with graph_type='reg', and p with graph_type in ['prob', 'erdos'].

    Input:
        n: Problem size
        d: [Optional] Degree of each node in graph
        p: [Optional] Probability of edge between two nodes
        graph_type: Specifies graph type to generate
        random_seed: Seed value for random generator
    Output:
        nx_graph: NetworkX OrderedGraph of specified type and parameters
    """
    if graph_type == 'reg':
        print(f'Generating d-regular graph with n={n}, d={d}, seed={random_seed}')
        nx_temp = nx.random_regular_graph(d=d, n=n, seed=random_seed)
    elif graph_type == 'prob':
        print(f'Generating p-probabilistic graph with n={n}, p={p}, seed={random_seed}')
        nx_temp = nx.fast_gnp_random_graph(n, p, seed=random_seed)
    elif graph_type == 'erdos':
        print(f'Generating erdos-renyi graph with n={n}, p={p}, seed={random_seed}')
        nx_temp = nx.erdos_renyi_graph(n, p, seed=random_seed)
    else:
        raise NotImplementedError(f'!! Graph type {graph_type} not handled !!')

    # Networkx does not enforce node order by default
    nx_temp = nx.relabel.convert_node_labels_to_integers(nx_temp)
    # Need to pull nx graph into OrderedGraph so training will work properly
    nx_graph = nx.OrderedGraph()
    nx_graph.add_nodes_from(sorted(nx_temp.nodes()))
    nx_graph.add_edges_from(nx_temp.edges)
    return nx_graph


# helper function to convert Q dictionary to torch tensor
def qubo_dict_to_torch(nx_G, Q, torch_dtype=None, torch_device=None):
    """
    Output Q matrix as torch tensor for given Q in dictionary format.

    Input:
        Q: QUBO matrix as defaultdict
        nx_G: graph as networkx object (needed for node lables can vary 0,1,... vs 1,2,... vs a,b,...)
    Output:
        Q: QUBO as torch tensor
    """

    # get number of nodes
    n_nodes = len(nx_G.nodes)

    # get QUBO Q as torch tensor
    Q_mat = torch.zeros(n_nodes, n_nodes)
    for (x_coord, y_coord), val in Q.items():
        Q_mat[x_coord][y_coord] = val

    if torch_dtype is not None:
        Q_mat = Q_mat.type(torch_dtype)

    if torch_device is not None:
        Q_mat = Q_mat.to(torch_device)

    return Q_mat
    
def gen_q_dict_mis(nx_G, penalty=2):
    """
    Helper function to generate QUBO matrix for MIS as minimization problem.
    
    Input:
        nx_G: graph as networkx graph object (assumed to be unweigthed)
    Output:
        Q_dic: QUBO as defaultdict
    """

    # Initialize our Q matrix
    Q_dic = defaultdict(int)

    # Update Q matrix for every edge in the graph
    # all off-diagonal terms get penalty
    for (u, v) in nx_G.edges:
        Q_dic[(u, v)] = penalty

    # all diagonal terms get -1
    for u in nx_G.nodes:
        Q_dic[(u, u)] = -1

    return Q_dic

def gen_fastHare_q_dict(data, penalty=2):
    # Initialize our Q matrix
    Q_dic = defaultdict(int)

    # Update Q matrix for every edge in the graph
    # all off-diagonal terms get penalty
    for idx in range(data.edge_index.size()[1]):
        u = data.edge_index[0][idx]
        v = data.edge_index[1][idx]
        Q_dic[(u, v)] = data.edge_attr[idx]

    # all diagonal terms get -1
    # for u in range(data.x.size()[0]):
    #     Q_dic[(u, u)] = -1

    return Q_dic

def fastHare_qubo_dict_to_torch(data, Q, torch_dtype=None, torch_device=None):
    """
    Output Q matrix as torch tensor for given Q in dictionary format.

    Input:
        Q: QUBO matrix as defaultdict
        nx_G: graph as networkx object (needed for node lables can vary 0,1,... vs 1,2,... vs a,b,...)
    Output:
        Q: QUBO as torch tensor
    """

    # get number of nodes
    n_nodes = data.num_nodes

    # get QUBO Q as torch tensor
    Q_mat = torch.zeros(n_nodes, n_nodes)
    for (x_coord, y_coord), val in Q.items():
        Q_mat[x_coord][y_coord] = val

    if torch_dtype is not None:
        Q_mat = Q_mat.type(torch_dtype)

    if torch_device is not None:
        Q_mat = Q_mat.to(torch_device)

    return Q_mat


def Create_adj_label_Cora(data, alpha=0.2, min_edge=-20, max_edge=20):
    sk_ising = []
    for idx in range(data.edge_index.size()[1]):
        u = data.edge_index[0][idx]
        v = data.edge_index[1][idx]
        w = data.edge_attr[idx]
        sk_ising.append((u, v, w))
    rh, map, sign, time1 = m.fasthare_reduction(sk_ising,alpha=alpha)
    # adj_label = []
    # for i in range(len(m)):
    #     for j in range(len(m)):
    #         if m[i]!=m[j] or i==j:
    #             adj_label.append(0)
    #         else:
    #             adj_label.append(1)
    map = torch.tensor(map, dtype=torch.float64)
    adj_label = (map[data.edge_index[0,:]]==map[data.edge_index[1,:]])*1
    adj_label = adj_label.to(torch.float64)
    print(adj_label.size())
    print(data)
    train_mask = torch.tensor(random.choices([0,1,2], [0.2, 0.7, 0.1],k=adj_label.size()[0]))
    data.test_mask = (train_mask==0)
    data.val_mask = (train_mask==2)
    data.train_mask = (train_mask==1)
    
    data.update({'adj_label':adj_label})


def load_graph_json(path):
    f = open(path)
    fastHare_data = json.load(f)
    src = []
    dst = []
    for v in fastHare_data['edges'].values():
        src.append(v[0])
        dst.append(v[1])

    edge_index = torch.tensor([src,dst], dtype=torch.long)
    edge_attr = [v for v in fastHare_data['J'].values()]
    # x = F.normalize(torch.tensor([v for v in data['H'].values()], dtype=torch.float),dim=-1).unsqueeze(dim=-1)
    y = torch.tensor([0 if v==-1 else v for v in fastHare_data['label']], dtype=torch.long)
    
    #create temporary x, train_mask, test_mask, val_mask 
    return Data(x=y.unsqueeze(dim=-1), edge_index=edge_index, y=y, 
                edge_attr=edge_attr, 
                train_mask=y, test_mask=y, val_mask=y)

def fast_score_calculate(data):
    abs_weight = torch.abs(data.edge_attr)
    sum_row = scatter_add(abs_weight, data.edge_index[0], 0, dim_size=data.num_nodes)
    
    u,v = data.edge_index[0], data.edge_index[1]
    fast_score = 2*abs_weight - torch.minimum(u, v)
    fast_score = torch.nn.functional.normalize(fast_score, p=2, dim=-1)

    data.update({'fast_score':fast_score})

def graph_compress(data, adj_pred):
    def _union(parent, x, y):
        parent[x] = y
    def _find_parent(parent, i):
        if parent[i] == i:
            return i
        else:
            return _find_parent(parent,parent[i])

    parent = [i for i in ranged(data.num_nodes)]
    for idx in range(data.edge_index):
        compress_pred = adj_pred[idx]
        u,v = data.edge_index[0][idx], data.edge_index[1][idx]
        par_u = _find_parent(parent,u)
        par_v = _find_parent(parent,v)
        if (par_u != par_v) & (compress_pred == 1):
            _union(parent,u,v)