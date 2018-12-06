import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})" .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * ((target > 0.5).float() * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()

class DiceLoss(nn.Module):
    def __init__(self,smooth=1.):
        super().__init__()
        self.smooth = smooth
    def forward(self,predicts,labels):
        predicts = predicts.view( predicts.shape[0] , -1 )
        labels = labels.view( labels.shape[0] , -1 )
        intersection = (predicts * labels ).sum()
        score = ( 2 * intersection + self.smooth ) / ( predicts.sum() + labels.sum() + self.smooth )
        return score

