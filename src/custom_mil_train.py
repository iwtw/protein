import train_config as config
import os
from train import main
from time import sleep
from functools import partial
from torch import nn
import torch


    
if __name__ == "__main__":

    best_zero = -1e9
    best_result = None


    config.train['MIL'] = True
    config.train['MIL_aggregate_fn'] = partial( torch.mean  , dim = 0 )
    config.train['batch_size'] = 32
    config.train['val_batch_size'] = 32

    config.train['save_metric'] = {'macro_f1_score':True , 'bce':False , 'acc':True }#True : saves the max , False : saves the min
    config.net['pretrained'] = True
    config.train['freeze_feature_layer_epochs'] = 0
    config.train['lr_for_parts'] = [1,1,1]
    config.train['lr_curve'] = 'one_cycle'
    config.train['lr_find'] = False
    config.loss['name'] = 'Loss_v8'
    config.loss['stage_epoch'] = [0,75]
    config.net['name'] =  'gluoncv_resnet_v9.resnet34' 

    config.net['input_shape'] = (128,128)
    #config.data['train_dir'] = '../data/external/HPAv18_images_single_cell_crop'
    #config.data['train_csv_file'] = '../data/external/HPAv18RBGY_wodpl.csv'
    config.data['train_dir'] = '/mnt/ssd_raid0/wtw/datasets/human_protein/train_single_cell_crop'
    config.data['train_csv_file'] = '../data/train_single_cell_crop.csv'

    for config.train['lrs'] in [  [8e-3,8e-3]  ]:
        for config.train['lr_bounds'] in [  [0,75,76]   ] :
            config.parse_config()
            #for config.net['se_kwargs']['pool_fn'] in [ partial( nn.AdaptiveAvgPool2d , output_size = (1,1) )  ]:
            print('================================================================================================================================================================================================================================')
            sleep(5)
            try:
                result = main(config)
                if best_zero < result['macro_f1_score']:
                    best_zero = result['macro_f1_score']
                    best_result = result
            except KeyboardInterrupt:
                pass
    print('best result :')
    print( best_result )
