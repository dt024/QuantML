import torch


def loss_func(probs, Q_mat):
    """
    Function to compute cost value for given probability of spin [prob(+1)] and predefined Q matrix.

    Input:
        probs: Probability of each node belonging to each class, as a vector
        Q_mat: QUBO as torch tensor
    """

    probs_ = torch.unsqueeze(probs, 1)

    # minimize cost = x.T * Q * x
    cost = (probs_.T @ Q_mat @ probs_).squeeze()

    return cost

def FastHare_loss_func(probs, Q_mat, embed=None):
    """
    Function to compute cost value for given probability of spin [prob(+1)] and predefined Q matrix.

    Input:
        probs: Probability of each node belonging to each class, as a vector
        Q_mat: QUBO as torch tensor
    """

    probs_ = torch.unsqueeze(probs, 1)
    # minimize cost = x.T * Q * x
    cost = (-1)*(probs_.T @ Q_mat @ probs_).squeeze()

    embed_norm = embed
    cos = torch.cos(embed_norm).mean(dim=0)
    sin = torch.sin(embed_norm).mean(dim=0)
    R = (cos**2 + sin**2)**0.5
    R = R.mean()

    return cost - R

def FastHare_GumbelMax_loss_func(probs, Q_mat, A=None):
    """
    Function to compute cost value for given probability of spin [prob(+1)] and predefined Q matrix.

    Input:
        probs: Probability of each node belonging to each class, as a vector
        Q_mat: QUBO as torch tensor
    """

    probs_ = torch.unsqueeze(probs, 1)
    # minimize cost = x.T * Q * x
    cost = (-1)*(probs_.T @ Q_mat @ probs_).squeeze()
    Q_top = A.T @ Q_mat @ A
    x_top = A.T @ probs_
    # clustering cost = x'.T * Q' * x'
    cost_top = (-1)*(x_top.T @ Q_top @ x_top).squeeze()

    return cost + 0.1*torch.abs(cost-cost_top)

def BCELoss_class_weighted(weights):

    def loss(input, target):
        input = torch.clamp(input, min=1e-7, max=1-1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return torch.mean(bce)

    return loss

def FastHare_loss_func_2(probs, Q_mat, adj_probs, adj_label, weights=None, embed=None, mse=False):
    """
    Function to compute cost value for given probability of spin [prob(+1)] and predefined Q matrix.

    Input:
        probs: Probability of each node belonging to each class, as a vector
        Q_mat: QUBO as torch tensor
    """
    #adj_classifier loss
    
    if (weights is None) & (not mse):
        loss = torch.nn.BCELoss()
    elif not mse:
        loss = BCELoss_class_weighted(weights)
    else:
        loss = torch.nn.MSELoss()
    adj_loss = loss(adj_probs, adj_label)
    
    probs_ = torch.unsqueeze(probs, 1)
    # minimize cost = x.T * Q * x
    cost = (probs_.T @ Q_mat @ probs_).squeeze()

    embed_norm = embed
    cos = torch.cos(embed_norm).mean(dim=0)
    sin = torch.sin(embed_norm).mean(dim=0)
    R = (cos**2 + sin**2)**0.5
    R = R.mean()

    return cost + adj_loss
