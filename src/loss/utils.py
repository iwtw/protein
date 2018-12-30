import torch.nn as nn
import torch.nn.functional as F
import torch

class FocalLoss(nn.Module):
    def __init__(self, gamma=2 , weight = None):
        super(type(self),self).__init__()
        self.gamma = gamma
        self.weight = weight
        if self.weight is not None:
            self.normalized_weight = self.weight / weight.sum()
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})" .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * ((target > 0.5).float() * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        if self.weight is None:
            loss = 1.0 / loss.shape[1] * loss
        else:
            loss = loss * self.normalized_weight.view(1,-1)
        return loss.sum(dim=1).mean()

class DiceLoss(nn.Module):
    def __init__(self,smooth=1.):
        super(type(self),self).__init__()
        self.smooth = smooth
    def forward(self,predicts,labels):
        predicts = predicts.view( predicts.shape[0] , -1 )
        labels = labels.view( labels.shape[0] , -1 )
        intersection = (predicts * labels ).sum()
        score = ( 2 * intersection + self.smooth ) / ( predicts.sum() + labels.sum() + self.smooth )
        return score

class F1Loss(nn.Module):
    def __init__( self ):
        super(type(self),self).__init__()

    def forward(self, y_true, y_pred , eps = 1e-8 ):
        y_true = ( y_true > 0.5 ).float()
        tp = (y_true*y_pred).sum(0)
        fp = ((1-y_true)*y_pred).sum(0)
        fn = (y_true*(1-y_pred)).sum(0)

        p = tp / ( (tp + fp ) + eps )
        r = tp / ( (tp + fn ) + eps )


        f1 = 2*p*r / ( (p+r ) + eps  )
        #f1 = ( 2 * tp**2 ) / ( 2*tp**2 + tp * ( fp+fn ) )


        f1 = torch.where((torch.isnan(f1)) , torch.zeros_like(f1), f1)
        #f1 = torch.where( (tp == 0) and (( tp+fn ) == 0)  , torch.ones_like(f1) , f1 )
        return 1 - f1.mean()


