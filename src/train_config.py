import torch.nn as nn
from functools import partial
import datetime
#train_config

train = {}

train['random_seed'] = 42


train['batch_size'] = 32 
train['val_batch_size'] = 32
train['log_step'] = 10
train['save_epoch'] = 1
train['save_metric'] = 'err'
train['optimizer'] = 'SGD'
train['learning_rate'] = 1e1

#-- for SGD -- #
train['momentum'] = 0.9
train['nesterov'] = True 

train['mannual_learning_rate'] = True
#settings for mannual tuning
train['lr_bounds'] = [ 0 , 40 , 60 , 80 , 100 ]
train['lrs'] = [ 1e-1 , 1e-2 , 1e-3 , 1e-4 ]
    

#settings for cosine annealing
train['use_cycle_lr'] = False
train['cycle_len'] = 1
train['num_cycles'] = 4
#train['num_restart'] = 5
train['cycle_mult'] = 1.6


#config for save , log and resume
train['resume'] = None
#train['resume_epoch'] = None  #None means the last epoch
train['resume_optimizer'] = False





global net
net = {}
net['name'] = 'resnet34'
net['input_shape'] = (512,512)



loss = {}
#arc loss
loss['arcloss_start_epoch'] = 10
loss['m'] = 0.2
loss['s'] = 16

loss['weight_l2_reg'] = 5e-4

test = {}

data = {}
data['train_dir'] = '../data/train'
data['test_dir'] = '../data/test'


def parse_config():
    train['num_epochs'] = train['lr_bounds'][-1]
    data['csv_file'] = '../data/train.csv'
    #split_args = [train['split_random_seed']]
    #train['train_img_list'] = '../data/split_lists/{}_train_splitargs_{}_{}_{}.list'.format( train['dataset'] , *split_args )
    #train['val_img_list'] = {'zero':'../data/split_lists/{}_zero_val_splitargs_{}_{}_{}.list'.format( train['dataset'] ,*split_args )  , 'non_zero':'../data/split_lists/{}_nonzero_val_splitargs_{}_{}_{}.list'.format( train['dataset'] ,*split_args ), 'all':'../data/split_lists/{}_all_val_splitargs_{}_{}_{}.list'.format( train['dataset'] ,*split_args ) }


    global net
    net_name = net['name']
    print('parsing----')
    print(net_name)
    net = {}
    net['name'] = net_name
    net['input_shape'] = (512,512)
    net['num_classes'] = 28
    
    '''
    if 'arc_resnet' in net['name']:
        net['strides'] = [1, 1, 2, 2, 1]#including first conv
        net['first_kernel_size'] = 3  
        net['fm_mult'] =  1.0
        net['use_batchnorm'] = True
        net['activation_fn'] = partial( nn.ReLU , inplace = True)
        net['pre_activation'] = True
        net['use_maxpool'] = False
        net['use_avgpool'] = True
        net['feature_layer_dim'] = 512
        net['dropout'] = 0.5
        #net['type'] = 'all'
        if net['type'] == 'coarse':
            net['num_classes'] = 10
            train['lr_bounds'] = [ 0 , 40 , 60 , 80 , 100 ]
            train['lrs'] = [ 1e-1 , 1e-2 , 1e-3 , 1e-4 ]
            net['feature_layer_dim'] = 128
    '''
            

    train['sub_dir'] = '{}_{}_shape{},{}_seed{}'.format( net['name'] , net['input_shape'][0],net['input_shape'][1] , train['optimizer'] , train['random_seed'] )

    time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    train['log_dir'] = '../save/{}/{}'.format( train['sub_dir'] , time )

parse_config()
