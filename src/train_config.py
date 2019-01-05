import torch.nn as nn
from functools import partial
import datetime
#train_config
import torch

train = {}

train['random_seed'] = 0

train['MIL'] = False
train['MIL_aggregate_fn'] = partial( torch.mean  , dim = 0 )
train['mix_up'] = False

train['batch_size'] = 32 
train['val_batch_size'] = 32

train['log_step'] = 100
train['save_epoch'] = 1
train['save_metric'] = {'macro_f1_score':True , 'bce':False , 'acc':True }#True : saves the max , False : saves the min
train['optimizer'] = 'Adam'
train['learning_rate'] = 1e-1

#-- for Adam -- #
train['amsgrad'] = False
train['betas'] = [0.9,0.999]

#-- for SGD -- #
train['momentum'] = 0.9
train['nesterov'] = True 

train['clip_grad_norm'] = 1.0
train['mannual_learning_rate'] = True
#settings for mannual tuning
#train['lr_bounds'] = [ 0 , 40 , 60 , 80 , 100 ]
#train['lr_for_parts'] = [1/10,1/3,1]
train['lr_for_parts'] = [1,1,1]
#train['lrs'] = [ 1e-1 , 1e-2 , 1e-3 , 1e-4 ]
    
train['lr_find'] = True

#settings for cosine annealing learning rate
train['lr_curve'] = 'one_cycle'
assert train['lr_curve'] in ['cosine','cyclical','one_cycle','normal'] 
#cyclical

#train['lr_bounds'] = [0,40,60,80,100]
#train['lrs'] = [1e-1,1e-2,1e-3,1e-4]
train['freeze_feature_layer_epochs'] = 2
train['freeze_lr_curve'] = 'one_cycle'
assert train['freeze_lr_curve'] in ['cosine','cyclical','one_cycle','normal'] 
ilr = 2e-2
train['lrs'] = [ ilr , ilr , ilr/4 , ilr/4 ,  ilr/4 , ilr/4 ,ilr/4 ,ilr/4 , ilr/16 ]
#train['lr_bounds'] = [0,5,7,9,11,13,17,21,29]
#train['lr_bounds'] = [0,1,2 , 4,6,8,10,14,18,26]
train['lr_bounds'] = [0,75]
train['cyclical_lr_inc_ratio'] = 0.3
train['cyclical_lr_init_factor'] =  1/20
train['cyclical_mom_min'] = 0.85
train['cyclical_mom_max'] = 0.95
train['restart_optimizer'] = []


#config for save , log and resume
train['resume'] = None
#train['resume_epoch'] = None  #None means the last epoch
train['resume_optimizer'] = False





global net
net = {}
net['name'] = 'gluoncv_resnet_v15.resnet34'
net['input_shape'] = (512,512)
net['pretrained'] = False



loss = {}
#arc loss
loss['name'] = 'Loss_v8'
loss['stage_epoch'] = [0,1000]

#for Arc loss
#loss['arcloss_start_epoch'] = 10
#loss['m'] = 0.2
#loss['s'] = 16


loss['weight_mse']  = 0.5
loss['weight_bce'] = 1
loss['weight_f1'] = 1
#loss['weight_l2_reg'] = 5e-4
loss['weight_l2_reg'] = 5e-6
loss['class_weight_dampening'] = 'log'


test = {}
test['model'] = '../save/gluoncv_resnet_v15.resnet34_shape512,512_seed1_Adam/20190104_091713/models/last.pth'
test['batch_size'] = 8
test['tta'] = 20

data = {}
data['train_csv_file'] = '../data/train_mix1.csv'
data['test_size'] = 0.2
data['train_dir'] = ''
data['test_dir'] = '../data/test'
data['smooth_label_epsilon'] = 0.0
data['image_format'] = 'png'


def parse_config():
    train['num_epochs'] = train['lr_bounds'][-1]
    #split_args = [train['split_random_seed']]


    global net
    net_name = net['name']
    print('parsing----')
    print(net_name)
    net['name'] = net_name
    #net['input_shape'] = (512,512)
    net['num_classes'] = 28
    #net['dilated'] = False
    net['dropout'] = 0.5
    if 'glu' in  net_name:
        net['dilated'] = False
    if 'v6' in net_name or 'v7' in net_name or 'v10' in net_name:
        net['se_kwargs'] = {}
        net['se_kwargs']['pool_fn'] = partial( nn.AdaptiveAvgPool2d , output_size = (1,1) )
        net['se_kwargs']['reduction_ratio'] = 24
    

    train['sub_dir'] = '{}_shape{},{}_seed{}_{}'.format( net['name'] , net['input_shape'][0],net['input_shape'][1]  , train['random_seed'], train['optimizer'] )

    time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    train['log_dir'] = '../save/{}/{}'.format( train['sub_dir'] , time )

parse_config()
