import torch
from torch_geometric.utils import degree, get_laplacian
from torch_geometric.nn import GraphConv, GATConv, TransformerConv, SAGEConv, AntiSymmetricConv,GCNConv
from torch_geometric.utils import degree,get_laplacian
import torch_sparse
from torch_scatter import scatter_add
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear


class GCN(torch.nn.Module):
    def __init__(self, embed_channels, hidden_channels, num_nodes, num_classes, dropout):
        super().__init__()
        self.conv1 = GraphConv(embed_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, num_classes)
        self.dropout = dropout
        self.embed = nn.Embedding(num_nodes, embed_channels)
        self.lin_Q = Linear(hidden_channels, hidden_channels)
        self.lin_K = Linear(hidden_channels, hidden_channels)

    def forward(self, node_features, edge_index, edge_weight=None):
        if node_features is None:
            x = self.embed.weight
        else:
            x = node_features
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        new_x = x.clone()
        x = F.dropout(x, p=self.dropout)
        x = self.conv2(x, edge_index, edge_weight)
        x = torch.sigmoid(x)

        x_Q = self.lin_Q(torch.relu(new_x))
        x_K = self.lin_K(torch.relu(new_x))
        src = x_Q[edge_index[0, :], :]
        dst_k = x_Q[edge_index[1, :], :]

        edge_classify = torch.sum(src * dst_k, dim=1)

        return x, new_x, torch.flatten(torch.tanh(edge_classify))

class GCN2(torch.nn.Module):
    def __init__(self, embed_channels, hidden_channels, num_nodes, num_classes, dropout):
        super().__init__()
        # self.conv1 = SAGEConv(embed_channels, hidden_channels)
        # self.conv2 = SAGEConv(hidden_channels, num_classes)
        self.lin2 = Linear(embed_channels, num_classes)
        self.dropout = dropout
        self.embed = nn.Embedding(num_nodes, embed_channels)

    def forward(self, edge_index, edge_weight=None):
        x = self.embed.weight

        delta = 0.01
        new_x = x
        orig = new_x.clone()
        K = 1
        lap_edge_index, lap_edge_weight = get_laplacian(edge_index, edge_weight, normalization='rw')
        
        for _ in range(20):
          # ax = torch_sparse.spmm(lap_edge_index, lap_edge_weight, new_x.shape[0], new_x.shape[0], new_x)

          cos_R = torch_sparse.spmm(lap_edge_index, lap_edge_weight, new_x.shape[0], new_x.shape[0], torch.cos(new_x))
          sin_R = torch_sparse.spmm(lap_edge_index, lap_edge_weight, new_x.shape[0], new_x.shape[0], torch.sin(new_x))
          phi = torch_sparse.spmm(lap_edge_index, lap_edge_weight, new_x.shape[0], new_x.shape[0], new_x)
          R = torch.sqrt((torch.cos(new_x)-cos_R)**2 + (torch.sin(new_x)-sin_R)**2) 
          # print(R)
          out_phi = -phi
          out_hat = K*R*torch.sin(out_phi)

          # new_x = new_x + delta*(orig-ax)

          new_x = new_x + delta*(orig + out_hat)

        x = self.lin2(torch.relu(new_x))        
        x = torch.sigmoid(x)
        return x, new_x

class GCN3(torch.nn.Module):
    def __init__(self, embed_channels, hidden_channels, num_nodes, num_classes, dropout, num_cluster):
        super().__init__()
        self.conv1 = GraphConv(embed_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, num_classes)
        self.dropout = dropout
        self.embed = nn.Embedding(num_nodes, embed_channels)
        if num_cluster is not None:
          self.cluster_trans = Linear(embed_channels, num_cluster)

    def forward(self, edge_index, edge_weight=None):
        x = self.embed.weight
        A = F.gumbel_softmax(self.cluster_trans(x))

        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = F.dropout(x, p=self.dropout)
        x = self.conv2(x, edge_index, edge_weight)
        x = torch.sigmoid(x)
        return x, A

class GCN4(torch.nn.Module):
    def __init__(self, embed_channels, hidden_channels, num_nodes, num_classes, dropout):
        super().__init__()
        # self.conv1 = SAGEConv(embed_channels, hidden_channels)
        # self.conv2 = SAGEConv(hidden_channels, num_classes)
        self.lin2 = Linear(embed_channels, num_classes)
        self.lin = Linear(embed_channels, embed_channels)
        self.lin_Q = Linear(embed_channels, hidden_channels)
        self.lin_K = Linear(embed_channels, hidden_channels)
        self.lin3 = Linear(embed_channels, embed_channels)

        self.dropout = dropout
        self.embed = nn.Embedding(num_nodes, embed_channels)

    def forward(self, node_features, edge_index, edge_weight=None):
        if node_features is None:
            x = self.embed.weight
        else:
            x = node_features

        #CREATING NEW EDGE_WEIGHT
        # src = x_Q[edge_index[0, :], :]
        # dst_k = x_K[edge_index[1, :], :]

        # new_edge_weight = torch.sum(src * dst_k, dim=1)
        # edge_weight = new_edge_weight

        lamb = 0.0
        new_x = torch.relu(self.lin(x))
        lap_edge_index, lap_edge_weight = get_laplacian(edge_index, edge_weight, normalization='rw')
        orig = new_x.clone()
        for _ in range(1):
          tmp = self.lin3(new_x)
        #   tmp = new_x
          norm_ax = torch_sparse.spmm(lap_edge_index, lap_edge_weight, new_x.shape[0], new_x.shape[0], tmp)
          ax = torch_sparse.spmm(edge_index, edge_weight, new_x.shape[0], new_x.shape[0], tmp)
        #   new_x = lamb*orig + norm_ax
          new_x = (lamb*orig + (norm_ax-ax))/(1+lamb)

        x = self.lin2(torch.relu(new_x))        
        x = torch.sigmoid(x)
        x_Q = self.lin_Q(torch.relu(new_x))
        x_K = self.lin_K(torch.relu(new_x))
        src = x_Q[edge_index[0, :], :]
        dst_k = x_K[edge_index[1, :], :]

        edge_classify = torch.sum(src * dst_k, dim=1)

        return x, new_x, torch.flatten(torch.sigmoid(edge_classify))