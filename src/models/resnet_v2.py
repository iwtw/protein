import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from numpy import prod
from .layers import *
import torch


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SE(nn.Module):
    def __init__( self , num_channels , pool_fn = partial( nn.AdaptiveAvgPool2d , output_size = (1,1)) , reduction_ratio = 16 ):
        super(type(self),self).__init__()
        self.pool = pool_fn()
        self.flatten = Flatten()
        self.fc1 = linear( num_channels , num_channels // reduction_ratio , activation_fn = nn.ReLU , use_batchnorm = False )
        self.fc2 = linear( num_channels // reduction_ratio , num_channels , activation_fn = nn.Sigmoid , use_batchnorm = False )
        for m in self.modules():
            if isinstance(m , nn.Linear ):
                nn.init.kaiming_normal_( m.weight )
                m.bias.data.zero_()
    def forward( self , x ):
        w = self.pool( x ) 
        w = self.flatten( w )
        w = self.fc1( w )
        w = self.fc2( w )
        w = w.view( w.shape[0] , w.shape[1] , 1 , 1  )
        return x * w


class BasicBlock(nn.Module):
    """pytorch torch.nn.Linear wrapper
    Notes
    ---------------------------------------------------------------------
    use partial() to wrap activation_fn if arguments are needed 
    examples:
        BasicBlock(32,32,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 , inplace = True ))
    """
    def __init__(self, in_channels , out_channels , kernel_size , stride = 1 , use_batchnorm = False , activation_fn = partial( nn.ReLU ,  inplace=True ) , last_activation_fn = partial( nn.ReLU , inplace=True  ) , pre_activation = False , scaling_factor = 1.0 , se_kwargs = None):
        super(BasicBlock, self).__init__()
        bias = False if use_batchnorm else True
        if pre_activation and stride == 2:
            self.shortcut = nn.Sequential( nn.BatchNorm2d(in_channels) , activation_fn() )
            self.conv1 = conv( in_channels , out_channels , kernel_size , stride , kernel_size//2 ,  activation_fn = None , pre_activation = False , use_batchnorm = False , bias = bias )
        else:
            self.shortcut = None
            self.conv1 = conv( in_channels , out_channels , kernel_size , stride , kernel_size//2 ,  activation_fn , pre_activation = pre_activation , use_batchnorm = use_batchnorm , bias = bias)
            

        self.conv2 = conv( out_channels , out_channels , kernel_size , 1 , kernel_size//2 , activation_fn , pre_activation = pre_activation ,  use_batchnorm = use_batchnorm , weight_init_fn = get_weight_init_fn(last_activation_fn) , bias = bias )
        if se_kwargs is not None:
            self.se = SE( out_channels , **se_kwargs  )
        else:
            self.se = SE( out_channels )
        self.downsample = None
        if stride != 1 or in_channels != out_channels :
            self.downsample = conv( in_channels , out_channels , 1 , stride , 0 , None , use_batchnorm if not pre_activation else False , bias = bias  )
        if not pre_activation and last_activation_fn is not None:
            self.last_activation = last_activation_fn()
        else:
            self.last_activation = None
        self.scaling_factor = scaling_factor

    def forward(self , x ):
        x = (x if self.shortcut is None else self.shortcut( x ))
        if self.downsample is not None:
            residual = self.downsample( x )
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se( out )

        #print(out.shape,residual.shape)
        out = out + residual * self.scaling_factor
        if self.last_activation is not None:
            out = self.last_activation( out )

        return out

class BottleneckBlock( nn.Module ):
    def __init__( self , in_channels , out_channels , kernel_size , stride = 1 , use_batchnorm = False , activation_fn = partial( nn.ReLU , inplace = True ) , last_activation_fn = partial( nn.ReLU , inplace = True ) , pre_activation = False , scaling_factor = 1.0 , se_kwargs = None):
        super(BottleneckBlock , self).__init__()
        mid_channels = out_channels//4
        bias = False if use_batchnorm else True
        if pre_activation and stride == 2 :
            self.shortcut = nn.Sequential( nn.BatchNorm2d(in_channels) , activation_fn() )
            self.conv1 = conv( in_channels , mid_channels , 1 , 1 , 0 , activation_fn = None , pre_activation = False , use_batchnorm = False , bias = bias )
        else:
            self.shortcut = None
            self.conv1 = conv( in_channels , mid_channels , 1 , 1 , 0 , activation_fn , pre_activation = pre_activation , use_batchnorm = use_batchnorm , bias = bias)
            

        self.conv2 = conv( mid_channels , mid_channels , kernel_size , stride , kernel_size//2 , activation_fn , pre_activation  = pre_activation , use_batchnorm = use_batchnorm , bias = bias)
        self.conv3 = conv( mid_channels , out_channels , 1 , 1 , 0 , activation_fn , pre_activation = pre_activation , use_batchnorm = use_batchnorm , bias = bias )
        if se_kwargs is not None:
            self.se = SE( out_channels , **se_kwargs  )
        else:
            self.se = SE( out_channels )
        self.downsample = None
        if stride != 1 or in_channels != out_channels :
            self.downsample = conv( in_channels , out_channels , 1 , stride , 0 , None , use_batchnorm if not pre_activation else False , bias = bias  )
        if not pre_activation and last_activation_fn is not None:
            self.last_activation = last_activation_fn()
        else:
            self.last_activation = None
        self.scaling_factor = scaling_factor

    def forward(self , x ):
        x = (x if self.shortcut is None else self.shortcut( x ))
        if self.downsample is not None:
            residual = self.downsample( x )
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.se( out )

        #print(out.shape,residual.shape)
        out = out + residual * self.scaling_factor
        if self.last_activation is not None:
            out = self.last_activation( out )

        return out
class ResNet( nn.Module ):
    def __init__(self,
            block,
            num_blocks,
            num_features,
            strides,
            num_classes,
            input_shape ,
            fm_mult = 1.0 , 
            first_kernel_size = 7 ,
            use_batchnorm=True,
            activation_fn=partial(nn.ReLU,inplace=True),
            pre_activation=False ,
            use_maxpool = True ,
            feature_layer_dim=None,
            dropout = 0.0 ):
        super(ResNet,self).__init__()
        num_features = [ int(i*fm_mult) for i in num_features ]

        self.use_batchnorm = use_batchnorm
        self.activation_fn = activation_fn
        self.pre_activation = pre_activation
        self.use_maxpool = use_maxpool
        #assert len(num_features) == 5 
        #assert len(num_blocks) == 4 
        self.conv1 = conv( 4 , num_features[0] , first_kernel_size , strides[0] , first_kernel_size//2 , activation_fn , use_batchnorm = use_batchnorm , bias = False  )
        if self.use_maxpool:
            self.maxpool = nn.MaxPool2d( 3,2,1 )

        blocks = []
        #blocks.append( self.build_blocks(block,num_features[0],num_features[0], strides[0] ,num_blocks[0]) )
        for i in range( 0,len(num_blocks)):
            blocks.append( self.build_blocks(block,num_features[i],num_features[i+1] , strides[i+1] , num_blocks[i] ) )
        self.blocks = nn.Sequential( *blocks )
        if self.pre_activation:
            self.post_bn = nn.Sequential( nn.BatchNorm2d( num_features[-1] ) , activation_fn() )


        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool2 = nn.AdaptiveMaxPool2d((1,1))

        if feature_layer_dim is not None:
            self.fc1 = nn.Sequential( Flatten() , linear( num_features[-1] * shape , feature_layer_dim , activation_fn = None , pre_activation  = False , use_batchnorm = use_batchnorm) )
        self.dropout = nn.Dropout( dropout )

        self.classifier = nn.Sequential( 
                Flatten(),
                nn.BatchNorm1d(num_features[-1]*2),
                nn.Dropout( dropout ),
                linear( num_features[-1]*2,num_features[-1] ),
                nn.ReLU(),
                nn.BatchNorm1d( num_features[-1] ),
                nn.Dropout( dropout ),
                linear(num_features[-1] , num_classes)
                )

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d ) or isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        #self.fc2 = ArcLinear( feature_layer_dim if feature_layer_dim is not None else num_features[-1] * shape , num_classes )

    def build_blocks(self,block,in_channels,out_channels,stride,length):
        layers = []
        layers.append( block( in_channels , out_channels , 3 ,  stride , self.use_batchnorm , self.activation_fn ,pre_activation=self.pre_activation ) )
        for i in range(1,length):
            layers.append( block(out_channels,out_channels, 3 , 1 , self.use_batchnorm , self.activation_fn , pre_activation = self.pre_activation ) )
        return nn.Sequential( *layers )

    def forward(self,x):

        
        out = self.conv1(x)
        if self.use_maxpool:
            out = self.maxpool(out)
        out = self.blocks(out)
        if self.pre_activation:
            out = self.post_bn( out )

        avg = self.avgpool(out)
        max_ = self.maxpool2(out)
        out = torch.cat( [avg,max_] , 1 )
        fc = self.classifier( out )
        return {'fc':fc}


        
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    num_features = [64,64,128,256,512]
    model = ResNet(BasicBlock, [2, 2, 2, 2] , num_features , **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model



def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    num_features = [64,64,128,256,512]
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_features , **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model



def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    num_features = [64,64*4,128*4,256*4,512*4]
    model = ResNet(Bottleneck, [3, 4, 6, 3],num_features , **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model



def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    num_features = [64,64*4,128*4,256*4,512*4]
    model = ResNet(Bottleneck, [3, 4, 23, 3],num_features ,  **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model



def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    num_features = [64,64*4,128*4,256*4,512*4]
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_features ,  **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
