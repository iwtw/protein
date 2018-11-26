import torch.nn as nn
from functools import partial
import datetime
#train_config

train = {}

train['random_seed'] = 0


train['batch_size'] = 64 
train['val_batch_size'] = 64

train['log_step'] = 100
train['save_epoch'] = 1
train['save_metric'] = 'err'
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
train['lr_for_parts'] = [1/10,1/3,1]
#train['lrs'] = [ 1e-1 , 1e-2 , 1e-3 , 1e-4 ]
    

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
train['lr_bounds'] = [0,1,2 , 4,6,8,10,14,18,26]
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
net['name'] = 'resnet34'
net['input_shape'] = (256,256)



loss = {}
#arc loss
loss['name'] = 'Loss_v1'
loss['arcloss_start_epoch'] = 10
loss['m'] = 0.2
loss['s'] = 16

#loss['weight_l2_reg'] = 5e-4
loss['weight_l2_reg'] = 5e-6

test = {}
test['model'] = '../save/resnet34_shape256,256_seed0_Adam/20181125_225542/models/last.pth'
test['batch_size'] = 16
test['tta'] = 16

data = {}
data['train_dir'] = '../data/train'
data['test_dir'] = '../data/test'


def parse_config():
    train['num_epochs'] = train['lr_bounds'][-1]
    data['train_csv_file'] = '../data/train.csv'
    data['test_csv_file'] = '../data/test.csv'
    #split_args = [train['split_random_seed']]
    #train['train_img_list'] = '../data/split_lists/{}_train_splitargs_{}_{}_{}.list'.format( train['dataset'] , *split_args )
    #train['val_img_list'] = {'zero':'../data/split_lists/{}_zero_val_splitargs_{}_{}_{}.list'.format( train['dataset'] ,*split_args )  , 'non_zero':'../data/split_lists/{}_nonzero_val_splitargs_{}_{}_{}.list'.format( train['dataset'] ,*split_args ), 'all':'../data/split_lists/{}_all_val_splitargs_{}_{}_{}.list'.format( train['dataset'] ,*split_args ) }


    global net
    net_name = net['name']
    print('parsing----')
    print(net_name)
    net['name'] = net_name
    #net['input_shape'] = (512,512)
    net['num_classes'] = 28
    net['dilated'] = False
    net['dropout'] = 0.5
    
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
            

    train['sub_dir'] = '{}_shape{},{}_seed{}_{}'.format( net['name'] , net['input_shape'][0],net['input_shape'][1]  , train['random_seed'], train['optimizer'] )

    time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    train['log_dir'] = '../save/{}/{}'.format( train['sub_dir'] , time )

parse_config()
