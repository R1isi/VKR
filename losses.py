import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):

    def __init__(self, bce_weight=0.5, smooth=1.0, eps=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)
        bce = F.binary_cross_entropy(pred, target)

        b = pred.shape[0]
        p = pred.reshape(b, -1)
        t = target.reshape(b, -1)
        inter = (p * t).sum(dim=1)
        dice = 1.0 - (2.0 * inter + self.smooth) / (p.sum(dim=1) + t.sum(dim=1) + self.smooth)
        dice = dice.mean()

        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice
