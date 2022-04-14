import torch

def multiclass_crossentropy(y, y_hat, classes = 10):
    return torch.sum(- y * torch.log(y_hat))

def mse(y_hat,y):
    y_oh = torch.nn.functional.one_hot(y)
    return torch.sum((y_oh - y_hat)**2)/y_hat.shape[0]

def mse_l1_regularization(y_hat,y,model,l = 0.001):
    y_oh = torch.nn.functional.one_hot(y)
    m = torch.sum((y_oh - y_hat)**2)/y_hat.shape[0]
    for p in model.parameters():
        m += l*torch.sum(p.abs())
    return m