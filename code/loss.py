import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def make_one_hot(labels, C=245, device = torch.device('cuda')):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    labels.to(device)
    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2)).zero_().to(device)
    target = one_hot.scatter_(1, labels.data.to(device), 1)
    target = Variable(target)
    return target

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, input, target, reduction="mean"):
        cent_loss = F.cross_entsropy(F.normalize(input), target, reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * cent_loss
        if reduction == "mean":
            focal_loss = torch.mean(focal_loss)
        return focal_loss


class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, xent=.1):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent
        self.y = torch.Tensor([1]).cuda()
    def forward(self, input, target, reduction="mean"):
        cosine_loss = F.cosine_embedding_loss(input, make_one_hot(target, C=245), self.y, reduction=reduction)
        cent_loss = F.cross_entropy(F.normalize(input), target, reduce=False, ignore_index=3)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * cent_loss
        if reduction == "mean":
            focal_loss = torch.mean(focal_loss)
        return self.xent * cosine_loss +  focal_loss