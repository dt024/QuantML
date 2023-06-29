import torch


class GAT(torch.nn.Module):
    def __init__(self, embed_channels, hidden_channels, num_nodes, num_classes, dropout):
        super().__init__()
        # self.lin1 = Linear(num_features, embed_channels)
        self.conv1 = GATConv(embed_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, num_classes)
        # self.lin2 = Linear(hidden_channels, num_classes)
        self.dropout = dropout
        self.embed = nn.Embedding(num_nodes, embed_channels)

    def forward(self, edge_index, edge_weight=None):
        x = self.embed.weight
        # x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=self.dropout)
        x = self.conv2(x, edge_index)
        # x = self.lin2(x)
        x = torch.sigmoid(x)
        return x, _
    
class AntiSymmetric(torch.nn.Module):
    def __init__(self, embed_channels, hidden_channels, num_nodes, num_classes, dropout):
        super().__init__()
        # self.lin1 = Linear(num_features, embed_channels)
        phi = GraphConv(embed_channels, embed_channels)
        self.conv1 = AntiSymmetricConv(embed_channels, phi=phi)
        self.conv2 = AntiSymmetricConv(embed_channels, phi=phi, act='relu')
        self.lin2 = Linear(embed_channels, num_classes)
        self.dropout = dropout
        self.embed = nn.Embedding(num_nodes, embed_channels)

    def forward(self, edge_index, edge_weight=None):
        x = self.embed.weight
        # x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv1(x, edge_index)
        # x = torch.relu(x)
        x = F.dropout(x, p=self.dropout)
        # x = self.conv2(x, edge_index)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        return x
