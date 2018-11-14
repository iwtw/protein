from resnet import ResNet 
import layers 
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from utils import set_requires_grad
from copy import deepcopy
from numpy import prod
from collections import OrderedDict
from nasnet_layers import *
import math

class ArcLinear(nn.Module):
    def __init__( self , in_features , out_features , dropout = 0.0 ):
        super(type(self),self).__init__()
        self.dropout = nn.Dropout( dropout )
        self.linear = nn.Linear( in_features , out_features , bias = False)
        nn.init.xavier_normal_( self.linear.weight )
    def forward( self , x , use_normalization ):
        x = self.dropout( x )
        if use_normalization:
            return F.linear( F.normalize( x ) , F.normalize( self.linear.weight ) )
        else:
            return F.linear( x , self.linear.weight  )



class ArcResNet(nn.Module):
    def __init__( self ,  block , num_blocks , num_features , num_attributes , **kwargs ):
        super(type(self),self).__init__()
        self.features = ResNet( block , num_blocks , num_features , **kwargs )
        del( self.features.dropout )
        del( self.features.fc2 )
        self.features = nn.DataParallel( self.features )
        feature_dim = num_features[-1] if kwargs.get('feature_layer_dim') is None else kwargs['feature_layer_dim']
        self.classifier = ArcLinear( feature_dim  , kwargs['num_classes']  , dropout = kwargs['dropout'])
        self.attribute = AttributeLayer( feature_dim  , num_attributes)
        #self.inverse_attribute = InverseAttributeLayer( num_attributes , feature_dim )
    def forward( self , x , shit , use_normalization = True):
        feats = self.features( x )
        feats = feats.view( feats.shape[0] , -1 )
        fc = self.classifier( feats , use_normalization )
        attribute =  self.attribute( feats ) 
        return {'fc':fc ,  'feature':feats , 'attribute':attribute }


def tede_nasnet(**kwargs):
    kwargs.pop('input_shape')
    return TEDENasNet( **kwargs )

def arc_resnet10(fm_mult,**kwargs):
    feature_layer_dim = [64,64,128]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [2,2]
    return ArcResNet(layers.BasicBlock, num_blocks , feature_layer_dim ,  **kwargs)

def arc_resnet18(fm_mult,**kwargs):
    feature_layer_dim = [64,64,128,256,512]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [2,2,2,2]
    return ArcResNet(layers.BasicBlock, num_blocks , feature_layer_dim ,  **kwargs)

def arc_resnet34(fm_mult,**kwargs):
    feature_layer_dim = [64,64,128,256,512]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [3,4,6,3]
    return ArcResNet(layers.BasicBlock, num_blocks , feature_layer_dim ,  **kwargs)

def arc_resnet50(fm_mult,**kwargs):
    feature_layer_dim = [64,256,512,1024,2048]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [3,4,6,3]
    return ArcResNet(layers.BottleneckBlock, num_blocks , feature_layer_dim ,  **kwargs)

def rule_resnet18(fm_mult,**kwargs):
    feature_layer_dim = [64,64,128,256,512]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [2,2,2,2]
    return RULEResNet(layers.BasicBlock, num_blocks , feature_layer_dim ,  **kwargs)

def ris_resnet18(fm_mult,**kwargs):
    feature_layer_dim = [64,64,128,256,512]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [2,2,2,2]
    return RISResNet(layers.BasicBlock, num_blocks , feature_layer_dim ,  **kwargs)

def ris_resnet10(fm_mult,**kwargs):
    feature_layer_dim = [64,64,128]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [2,2]
    return RISResNet(layers.BasicBlock, num_blocks , feature_layer_dim ,  **kwargs)

def ris_resnet34(fm_mult,**kwargs):
    feature_layer_dim = [64,64,128,256,512]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [3,4,6,3]
    return RISResNet(layers.BasicBlock, num_blocks , feature_layer_dim ,  **kwargs)

def qfsl_resnet18(fm_mult,visual_semantic_layers_kwargs,**kwargs):
    feature_layer_dim = [64,64,128,256,512]
    feature_layer_dim = [ int(num_feature * fm_mult) for num_feature in feature_layer_dim ]
    num_blocks = [2,2,2,2]
    return QFSLResNet(layers.BasicBlock, num_blocks , feature_layer_dim , visual_semantic_layers_kwargs ,   **kwargs)

def hse_resnet18(resnet_kwargs,**kwargs):
    resnet_fm_mult = resnet_kwargs.pop('fm_mult')
    num_features = [64,64,128,256,512]
    num_features = [ int(num_feature * resnet_fm_mult) for num_feature in num_features ]
    num_blocks = [2,2,2,2]
    resnet_kwargs['num_features'] = num_features
    resnet_kwargs['num_blocks'] = num_blocks
    resnet_kwargs['block'] = layers.BasicBlock
    return HSEResnet(resnet_kwargs,**kwargs)

def tede_resnet18(feature_net_kwargs,**kwargs):
    resnet_fm_mult = feature_net_kwargs.pop('fm_mult')
    num_features = [64,64,128,256,512]
    num_features = [ int(num_feature * resnet_fm_mult) for num_feature in num_features ]
    num_blocks = [2,2,2,2]
    feature_net_kwargs['num_features'] = num_features
    feature_net_kwargs['num_blocks'] = num_blocks
    feature_net_kwargs['block'] = layers.BasicBlock
    kwargs.pop('input_shape')
    return TEDEResNet(feature_net_kwargs = feature_net_kwargs  ,**kwargs)

def tede_resnet50(feature_net_kwargs,**kwargs):
    fm_mult = feature_net_kwargs.pop('fm_mult')
    num_features = [64,256,512,1024,2048]
    num_features = [ int(num_feature * fm_mult) for num_feature in num_features ]
    num_blocks = [3,4,6,3]
    feature_net_kwargs['num_features'] = num_features
    feature_net_kwargs['num_blocks'] = num_blocks
    feature_net_kwargs['block'] = layers.BottleneckBlock
    kwargs.pop('input_shape')
    return TEDEResNet(feature_net_kwargs = feature_net_kwargs  ,**kwargs)

def gde_resnet18(feature_net_kwargs,**kwargs):
    resnet_fm_mult = feature_net_kwargs.pop('fm_mult')
    num_features = [64,64,128,256,512]
    num_features = [ int(num_feature * resnet_fm_mult) for num_feature in num_features ]
    num_blocks = [2,2,2,2]
    feature_net_kwargs['num_features'] = num_features
    feature_net_kwargs['num_blocks'] = num_blocks
    feature_net_kwargs['block'] = layers.BasicBlock
    kwargs.pop('input_shape')
    return GDEResNet(feature_net_kwargs = feature_net_kwargs  ,**kwargs)
