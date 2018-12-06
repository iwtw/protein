from .utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

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
        loss_dict['err'] = 1 - ( ( score > 0.0 ) == (batch['label']>0.5) ).float().mean()
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
        loss_dict['err'] = 1 - ( ( results['fc'] > 0.0 ) == (batch['label'] > 0.5 )).float().mean()
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
        loss_dict['err'] = 1 - ( ( fc > 0.5 ) == (batch['label']>0.5) ).float().mean()
        loss_dict['total'] = loss_dict['dice_loss']
        return loss_dict


class Loss_v1( nn.Module ):
    def __init__( self  , config = None ):
        super(type(self),self).__init__()
        self.config = config
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
    def forward( self , results , batch , epoch ):
        loss_dict = {}
        #print(batch['label'])
        loss_dict['focal'] = self.focal_loss( results['fc'] , batch['label'] )
        preds = results['fc'] > 0.0
        labels = batch['label'] > 0.5
        '''
        tp = ( preds & labels ).long().sum(0)
        fp = ( preds & (~labels) ).long().sum(0)
        fn = ( (~preds) & labels ).long().sum(0)
        loss_dict['tp'] = tp
        loss_dict['fp'] = fp
        loss_dict['fn'] = fn
        '''
        loss_dict['f1_score_label'] = batch['label'] > 0.5
        loss_dict['f1_score_pred'] = results['fc'] > 0
        loss_dict['err'] = 1 - ( ( results['fc'] > 0.0 ) == ( batch['label'] > 0.5 ) ).float().mean()
        loss_dict['total'] = loss_dict['focal']
        return loss_dict

'''
def compute_loss1( results , batch , epoch , config ,  class_range , is_training = False , mse_attribute = True ):
    s, m, k = config.loss['s'] , config.loss['m'] , config.test['k']
    cos_m = np.cos( m )
    sin_m = np.sin( m )
    border = np.cos( math.pi - m )
    loss_dict = {}
    #print(results['fc'].shape ,  batch['label'].max() )
    if config.net['type'] == 'coarse':
        labels = batch['super_class_label']
    else:
        labels = batch['label']

    if is_training:
        sum_exp = torch.exp(results['fc']).sum(1)
        loss_dict['err'] = 1 - torch.eq( labels  , torch.max( results['fc'] , 1 )[1] ).float().mean()
        topk , top_idx = torch.topk( torch.exp(results['fc'] ) / sum_exp.view(sum_exp.shape[0],1) , k = k , dim = 1 )
        #tqdm.write( "{}".format(topk[0]), file=sys.stdout )
        if epoch < config.loss['arcloss_start_epoch'] :
            loss_dict['softmax'] = cross_entropy(  s*results['fc'] , labels )

        else:
            cos_theta = results['fc']
            cos_theta_yi = cos_theta[( torch.arange(cos_theta.shape[0] , dtype = torch.int64 ) , labels )  ]
            sin_theta_yi = ( 1 - cos_theta_yi**2 ) **0.5
            phai_theta_yi = cos_theta_yi * cos_m - sin_theta_yi * sin_m
            phai_theta_yi = torch.where( cos_theta_yi > border , phai_theta_yi , -2. - phai_theta_yi )#the loss curve correction
            phai_theta = cos_theta
            phai_theta[ ( torch.arange( cos_theta.shape[0] , dtype = torch.int64 ) , labels ) ] = phai_theta_yi
            loss_dict['aam'] = cross_entropy( s * phai_theta , labels )
    else:
        k = config.test['k']
        fc = results['fc']
        sum_exp = torch.exp(fc).sum(1)
        #topk , top_idx = torch.topk( torch.exp( fc ) / sum_exp.view(sum_exp.shape[0],1) , k = k , dim = 1 )
        #for i in range(k):
        #    loss_dict['top{}'.format(i+1)] = torch.mean( topk[:,i] )

        
        #top_idx = top_idx.cpu().detach().numpy()

        predicts = get_predict( results , config , train_dataset.class_attributes , class_range )
        loss_dict['err'] = 1 - torch.eq( predicts , labels ).float().mean() 

    if mse_attribute:
        loss_dict['mse_attribute'] = mse( results['attribute'] , batch['attribute'] )
        #loss_dict['mse_feature_'] = mse( results['feature_'] , results['feature'].detach() )
    return loss_dict

        mix_predicts = get_predict( {'fc':mix_fc}, config , train_dataset.class_attributes )
'''


 
