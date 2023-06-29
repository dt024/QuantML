import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree,get_laplacian
import torch_sparse
from torch_scatter import scatter_add
from torch_geometric.nn import GraphConv, GATConv, TransformerConv, SAGEConv, AntiSymmetricConv,GCNConv
from torch.nn import Linear
import torch.nn.functional as F
import os
import random
import numpy as np
import json
import tqdm
import networkx as nx
from collections import defaultdict
from time import time
from torch_geometric.datasets import Planetoid
from utils import *
from loss_functions import *
from GCN import *
from otherGNNs import *
from train import *

# fix seed to ensure consistent results
seed_value = 1
random.seed(seed_value)        # seed python RNG
np.random.seed(seed_value)     # seed global NumPy RNG
torch.manual_seed(seed_value)  # seed torch RNG

# Set GPU/CPU
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float64
print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')


def main():
    # load cora
    is_CustomGraph = False
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    dataset = Planetoid(root=data_dir, name='Cora')
    data = dataset[0]
    # data = load_graph_json("./small_erdos_renyi/graph_0.json")
    data.x = data.x.to(torch.float64)
    data.update({'edge_attr':torch.tensor([random.randint(-20, 20) for _ in range(data.edge_index.size()[1])], dtype=TORCH_DTYPE)})
    Create_adj_label_Cora(data)
    fast_score_calculate(data)

    # NN learning hypers #
    number_epochs = int(1e5)
    learning_rate = 1e-3
    PROB_THRESHOLD = 0.5

    # Early stopping to allow NN to train to near-completion
    tol = 1e-4          # loss must change by more than tol, or trigger
    patience = 1000    # number early stopping triggers before breaking loop

    # Problem size (e.g. graph size)
    n = data.num_nodes
    # n=100
    # Establish dim_embedding and hidden_dim values
    # dim_embedding = int(np.sqrt(n))    # e.g. 10
    if is_CustomGraph:
        dim_embedding = 100
    else:
        dim_embedding = data.x.size()[1]
    # dim_embedding = 10
    hidden_dim = int(dim_embedding/2)  # e.g. 5

    # weights for loss weight[label] = num_samples / (num_classes*num_label)
    nEdges = data.adj_label.size()[0]
    nPostitive = data.adj_label.sum()
    nNegative = nEdges - nPostitive
    weights = [ # 2 classes
        nEdges / (2*nNegative),
        nEdges / (2*nPostitive)
    ]

    opt = {
        'lr': learning_rate,
        'dim_embedding': dim_embedding,
        'hidden_dim': hidden_dim,
        'dropout': 0.0,
        'number_classes': 1,
        'prob_threshold': PROB_THRESHOLD,
        'number_epochs': number_epochs,
        'tolerance': tol,
        'patience': patience,
        'is_CustomGraph': is_CustomGraph,
        'loss_weights': weights,
        'mse':False
    }
    
    #FASTHARE
    q_torch = fastHare_qubo_dict_to_torch(data, gen_fastHare_q_dict(data), torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE)
    #FASTHARE
    model_name = None
    model_name = 'FastHare_AdjCls'
    opt['cluster_num'] = 3
    model = GCN4(embed_channels=opt['dim_embedding'],
                hidden_channels=opt['hidden_dim'],
                num_nodes = data.num_nodes,
                # num_features=1,
                num_classes=opt['number_classes'],
                dropout=opt['dropout']).to(TORCH_DEVICE).to(TORCH_DTYPE)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'])

    prev_loss = 1.  # initial loss value (arbitrary)
    count = 0       # track number times early stopping is triggered
    best_loss = 0
    t_gnn_start = time()

    for epoch in range(1, opt['number_epochs']+1):
        if model_name == 'FastHare_AdjCls':                
            loss, res, final_embed, adj_cls = train(data, data.adj_label, model, optimizer, q_torch, opt,mse=opt['mse'])
        else:
            loss, res, final_embed = train(data.edge_index, edge_weight=data.edge_attr)
        
        if loss < best_loss:
            best_loss = loss
            best_bitstring = res

        if epoch % 10 == 0:
            if not opt['mse']:
                tmp_acc = test(data, data.val_mask, data.adj_label, model, opt)
                test_acc,f1,p,r = test(data, data.test_mask, data.adj_label, model, opt, metrics=True)

                print(f'Epoch: {epoch}, Loss: {loss}, val_adj_acc:{tmp_acc}, test_adj_acc:{test_acc}')
                print(f'Precision: {p}, Recall: {r}, F1: {f1}')
            else:
                tmp_loss = test(data, data.val_mask, data.adj_label, model, opt, mse=opt['mse'])
                test_loss = test(data, data.test_mask, data.adj_label, model, opt, metrics=True,mse=opt['mse'])
                print(f'Loss: {loss}, val_MSE_loss: {tmp_loss}, test_MSE_loss: {test_loss}')
        # early stopping check
        # If loss increases or change in loss is too small, trigger
        if (abs(loss - prev_loss) <= opt['tolerance']) | ((loss - prev_loss) > 0):
            count += 1
        else:
            count = 0

        if count >= opt['patience']:
            print(f'Stopping early on epoch {epoch}')
            break

        # update loss tracking
        prev_loss = loss

    # print(f'GNN training (n={nx_graph.number_of_nodes()}) took {round(time() - t_gnn_start, 3)}')
    print(f'GNN training (n={data.num_nodes}) took {round(time() - t_gnn_start, 3)}')
    print(f'GNN final continuous loss: {loss}')
    print(f'GNN best continuous loss: {best_loss}')

main()
# test(data, data.test_mask, model, opt)