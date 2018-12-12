import train_config as config
import os
from train import main
from time import sleep
from functools import partial
from torch import nn


    
if __name__ == "__main__":

    best_zero = -1e9
    best_result = None

    #for  config.loss['weight_l2_reg'] in [0,1e-4,1e-2]:
    #    for config.train['restart_optimizer'] in [ [2,10,18] , [] ] : for config.train['restart_optimizer'] in [ [2,10,18] , [] ] :
    #        config.parse_config()
    config.train['save_metric'] = {'macro_f1_score':True , 'bce':False , 'acc':True }#True : saves the max , False : saves the min
    config.net['pretrained'] = False
    config.train['freeze_feature_layer_epochs'] = 0
    config.train['lr_for_parts'] = [1,1,1]
    config.train['lr_curve'] = 'one_cycle'
    config.train['lr_find'] = True
    config.loss['name'] = 'Loss_v7'
    config.loss['stage_epoch'] = [0,75]
    for config.net['name'] in [ 'gluoncv_resnet_v11.resnet34' ]:
        for config.train['lrs'] in [  [1e-2 , 5e-3 , 1e-3] ]:
            for config.train['lr_bounds'] in [  [0,75,100]   ] :
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
