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
from sklearn.metrics import f1_score, precision_score, recall_score

# Set GPU/CPU
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float64

def train(data, label, model, optimizer, q_torch, opt, model_name='FastHare_AdjCls', mse=False):
      model.train()
      optimizer.zero_grad()  
      if opt['is_CustomGraph']:
        out = model(data.x.to(TORCH_DEVICE), 
                    data.edge_index.to(TORCH_DEVICE), 
                    edge_weight=data.edge_attr.to(TORCH_DEVICE))
      else:
        out = model(data.x.to(TORCH_DEVICE), 
                    data.edge_index.to(TORCH_DEVICE), 
                    edge_weight=data.edge_attr.to(TORCH_DEVICE))
      final_embed = out[1]
      if model_name=='FastHare_AdjCls':
        adj_probs = out[-1]
      out = out[0].squeeze()
      bitstring = (out >= opt['prob_threshold']) * 1
      bitstring[-1] = 1

      #FASTHARE
      bitstring = (2.0*bitstring-1.0)
      if model_name == 'FastHare_GumbelMax':
        loss = FastHare_GumbelMax_loss_func(out, q_torch, A=final_embed)
      elif model_name == 'FastHare_AdjCls':
        loss = FastHare_loss_func_2(out, q_torch, 
                                    adj_probs[data.train_mask], 
                                    label[data.train_mask].to(TORCH_DEVICE),
                                    weights=opt['loss_weights'],
                                    embed=final_embed,
                                    mse=mse)
      else:
        loss = FastHare_loss_func(out, q_torch, embed=final_embed)

      loss_ = loss.detach().item()
      loss.backward()  
      optimizer.step() 
      # print("END")
      if model_name=='FastHare_AdjCls':
        return loss_, bitstring, final_embed, (adj_probs >= opt['prob_threshold']) * 1
      return loss_, bitstring, final_embed

def test(data, mask, label, model, opt,metrics=False, mse=False):
      model.eval()
      test_acc = 0

      out = model(data.x.to(TORCH_DEVICE), 
                  data.edge_index.to(TORCH_DEVICE), 
                  edge_weight=data.edge_attr.to(TORCH_DEVICE))
      if not mse:
        adj_pred = (out[-1] >= opt['prob_threshold']) * 1  # Use the class with highest probability.
        test_correct = adj_pred[mask].cpu() == label[mask]  # Check against ground-truth labels.
        test_acc += int(test_correct.sum()) / (label[mask].size()[0])  # Derive ratio of correct predictions.
        # print('----------')
        # print(label[mask].size(), label[mask])
        # print(adj_pred[mask].size(), adj_pred[mask], adj_pred[mask].max(), adj_pred[mask].min())
        # print('----------')
        f1 = f1_score(label[mask], adj_pred[mask].cpu(), average='macro')
        p = precision_score(label[mask], adj_pred[mask].cpu(), average='macro')
        r = recall_score(label[mask], adj_pred[mask].cpu(), average='macro')
        if metrics:
          return test_acc, f1,p,r
        return test_acc
      else:
        return torch.nn.MSELoss()(out[-1][mask],label[mask].to(TORCH_DEVICE))