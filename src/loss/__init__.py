from .utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class Loss_v8( nn.Module ):
    '''
    first stage : bce loss 
    scond stage : bce loss + soft F1
    '''
    def __init__( self  , weight = None , config = None ):
        super(type(self),self).__init__()
        self.config = config
        self.bce = nn.BCEWithLogitsLoss(weight)
        self.f1_loss = F1Loss(weight)

    def forward( self , results , batch , epoch ):
        config = self.config
        loss_dict = {}
        loss_dict['bce'] = self.bce( results['fc'] , batch['label'] )
        preds = results['fc'] > 0.0
        loss_dict['f1_score_label'] = batch['label'] > 0.5
        loss_dict['f1_score_pred'] = results['fc'] > 0
        loss_dict['acc'] =  ( ( results['fc'] > 0.0 ) == ( batch['label'] > 0.5 ) ).float().mean()
        if epoch >= self.config.loss['stage_epoch'][1]:
            loss_dict['f1_loss'] = self.f1_loss( batch['label'] , F.sigmoid( results['fc'] ) )
            loss_dict['total'] =  config.loss['weight_bce'] * loss_dict['bce'] +  config.loss['weight_f1'] * loss_dict['f1_loss'] 
        else:
            loss_dict['total'] = config.loss['weight_bce'] * loss_dict['bce'] 
        y_pred = (results['fc'] > 0.0 ).long()
        y_true = (batch['label'] > 0.5 ).long()
        tp = (y_true*y_pred).long().sum(0,True)
        tn = ((1-y_true)*(1-y_pred)).long().sum(0,True)
        fp = ((1-y_true)*y_pred).long().sum(0,True)
        fn = (y_true*(1-y_pred)).long().sum(0,True)
        loss_dict['tp'] = tp
        loss_dict['fp'] = fp
        loss_dict['fn'] = fn
        loss_dict['tn'] = tn
        return loss_dict

class Loss_v7( nn.Module ):
    '''
    first stage : bce loss + mse
    scond stage : bce loss + soft F1 + mse
    '''
    def __init__( self  , config = None ):
        super(type(self),self).__init__()
        self.config = config
        self.bce = nn.BCEWithLogitsLoss()
        self.f1_loss = F1Loss()
        self.mse = nn.MSELoss()

    def forward( self , results , batch , epoch ):
        config = self.config
        loss_dict = {}
        loss_dict['bce'] = self.bce( results['fc'] , batch['label'] )
        loss_dict['mse'] = self.mse( results['unet'] , batch['img'].detach() )
        preds = results['fc'] > 0.0
        loss_dict['f1_score_label'] = batch['label'] > 0.5
        loss_dict['f1_score_pred'] = results['fc'] > 0
        loss_dict['acc'] =  ( ( results['fc'] > 0.0 ) == ( batch['label'] > 0.5 ) ).float().mean()
        if epoch >= self.config.loss['stage_epoch'][1]:
            loss_dict['f1_loss'] = self.f1_loss( batch['label'] , F.sigmoid( results['fc'] ) )
            loss_dict['total'] =  config.loss['weight_bce'] * loss_dict['bce'] +  config.loss['weight_f1'] * loss_dict['f1_loss'] + loss_dict['mse']
        else:
            loss_dict['total'] = config.loss['weight_bce'] * loss_dict['bce'] + config.loss['weight_mse'] * loss_dict['mse']
        return loss_dict

class Loss_v6( nn.Module ):
    '''
    first stage : focal loss
    scond stage : focal loss + soft F1
    '''
    def __init__( self  , weight = None , config = None ):
        super(type(self),self).__init__()
        self.config = config
        self.focal_loss = FocalLoss(weight)
        self.f1_loss = F1Loss(weight)
        self.mse = nn.MSELoss(weight)

    def forward( self , results , batch , epoch ):
        loss_dict = {}
        loss_dict['focal'] = self.focal_loss( results['fc'] , batch['label'] )
        preds = results['fc'] > 0.0
        loss_dict['f1_score_label'] = batch['label'] > 0.5
        loss_dict['f1_score_pred'] = results['fc'] > 0
        loss_dict['acc'] =  ( ( results['fc'] > 0.0 ) == ( batch['label'] > 0.5 ) ).float().mean()
        if epoch >= self.config.loss['stage_epoch'][1]:
            loss_dict['f1_loss'] = self.f1_loss( batch['label'] , F.sigmoid( results['fc'] ) )
            loss_dict['mse'] = self.mse( results['unet'] , batch['img'].detach() )
            loss_dict['total'] = 0.5 * loss_dict['focal'] + 0.5 * loss_dict['f1_loss'] + loss_dict['mse']
        else:
            loss_dict['total'] = loss_dict['focal']
        y_pred = (results['fc'] > 0.0 ).long()
        y_true = (batch['label'] > 0.5 ).long()
        tp = (y_true*y_pred).long().sum(0,True)
        tn = ((1-y_true)*(1-y_pred)).long().sum(0,True)
        fp = ((1-y_true)*y_pred).long().sum(0,True)
        fn = (y_true*(1-y_pred)).long().sum(0,True)
        loss_dict['tp'] = tp
        loss_dict['fp'] = fp
        loss_dict['fn'] = fn
        loss_dict['tn'] = tn
        return loss_dict

class Loss_v5( nn.Module ):
    '''
    first stage : focal loss
    scond stage : focal loss + soft F1
    '''
    def __init__( self  , config = None ):
        super(type(self),self).__init__()
        self.config = config
        self.focal_loss = FocalLoss()
        self.f1_loss = F1Loss()

    def forward( self , results , batch , epoch ):
        loss_dict = {}
        loss_dict['focal'] = self.focal_loss( results['fc'] , batch['label'] )
        preds = results['fc'] > 0.0
        loss_dict['f1_score_label'] = batch['label'] > 0.5
        loss_dict['f1_score_pred'] = results['fc'] > 0
        loss_dict['acc'] =  ( ( results['fc'] > 0.0 ) == ( batch['label'] > 0.5 ) ).float().mean()
        if epoch >= self.config.loss['stage_epoch'][1]:
            loss_dict['f1_loss'] = self.f1_loss( batch['label'] , F.sigmoid( results['fc'] ) )
            loss_dict['total'] = 0.5 * loss_dict['focal'] + 0.5 * loss_dict['f1_loss']
        else:
            loss_dict['total'] = loss_dict['focal']
        return loss_dict

class Loss_v4( nn.Module ):
    def __init__( self  , config = None ):
        super(type(self),self).__init__()
        self.config = config
        self.focal_loss = FocalLoss()
    def forward( self , results , batch , epoch ):
        config = self.config
        loss_dict = {}
        #print(batch['label'])
        s, m = config.loss['s'] , config.loss['m'] 
        cos_m = np.cos( m )
        sin_m = np.sin( m )
        border = np.cos( np.pi - m )
        loss_dict = {}
        #print(results['fc'].shape ,  batch['label'].max() )
        labels = batch['label']

        if self.training:
            if epoch < config.loss['arcloss_start_epoch'] :
                #loss_dict['softmax'] = cross_entropy(  s*results['fc'] , labels )
                score = results['fc']

            else:
                cos_theta = results['fc']
                #cos_theta_yi = cos_theta[( torch.arange(cos_theta.shape[0] , dtype = torch.int64 ) , labels )  ]
                labels_idx = torch.nonzero( labels )
                cos_theta_yi = cos_theta[ labels_idx[:,0] , labels_idx[:,1] ]
                sin_theta_yi = ( 1 - cos_theta_yi**2 ) **0.5
                phai_theta_yi = cos_theta_yi * cos_m - sin_theta_yi * sin_m
                phai_theta_yi = torch.where( cos_theta_yi > border , phai_theta_yi , -2. - phai_theta_yi )#the loss curve correction
                phai_theta = cos_theta
                #phai_theta[ ( torch.arange( cos_theta.shape[0] , dtype = torch.int64 ) , labels ) ] = phai_theta_yi
                phai_theta[ labels_idx[:,0] , labels_idx[:,1] ] = phai_theta_yi
                #loss_dict['aam'] = cross_entropy( s * phai_theta , labels )
                score = s * phai_theta
        else:
            score = results['fc']
        loss_dict['focal'] = self.focal_loss( score , batch['label'] )
        loss_dict['acc'] = ( ( score > 0.0 ) == (batch['label']>0.5) ).float().mean()
        loss_dict['total'] = loss_dict['focal']
        return loss_dict

class Loss_v3( nn.Module ):
    def __init__( self  , config = None ):
        super(type(self),self).__init__()
        self.config = config
        self.focal_loss = FocalLoss()
        self.mse = nn.MSELoss()
    def forward( self , results , batch , epoch ):
        loss_dict = {}
        #print(batch['label'])
        loss_dict['mse'] = self.mse( results['layer1_fc'] , batch['label'].byte().sum( 1 ).float() )
        loss_dict['focal'] = self.focal_loss( results['fc'] , batch['label'] )
        loss_dict['acc'] = ( ( results['fc'] > 0.0 ) == (batch['label'] > 0.5 )).float().mean()
        loss_dict['total'] = loss_dict['focal'] + loss_dict['mse']
        return loss_dict

class Loss_v2( nn.Module ):
    def __init__( self  , config = None ):
        super(type(self),self).__init__()
        self.config = config
        self.dice_loss = DiceLoss()
    def forward( self , results , batch , epoch ):
        loss_dict = {}
        #print(batch['label'])
        fc = torch.sigmoid( results['fc'] )
        loss_dict['dice_loss'] = self.dice_loss( fc , batch['label'] )
        loss_dict['acc'] = ( ( fc > 0.5 ) == (batch['label']>0.5) ).float().mean()
        loss_dict['total'] = loss_dict['dice_loss']
        return loss_dict


class Loss_v1( nn.Module ):
    def __init__( self  , config = None ):
        super(type(self),self).__init__()
        self.config = config
        self.focal_loss = FocalLoss()
    def forward( self , results , batch , epoch ):
        loss_dict = {}
        #print(batch['label'])
        loss_dict['focal'] = self.focal_loss( results['fc'] , batch['label'] )
        preds = results['fc'] > 0.0
        labels = batch['label'] > 0.5
        loss_dict['f1_score_label'] = batch['label'] > 0.5
        loss_dict['f1_score_pred'] = results['fc'] > 0
        loss_dict['acc'] = ( ( results['fc'] > 0.0 ) == ( batch['label'] > 0.5 ) ).float().mean()
        loss_dict['total'] = loss_dict['focal']
        return loss_dict

