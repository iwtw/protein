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
    config.net['pretrained'] = False
    config.train['freeze_feature_layer_epochs'] = 0
    config.train['lr_for_parts'] = [1,1,1]
    config.train['lr_curve'] = 'one_cycle'
    config.loss['name'] = 'Loss_v5'
    config.data['smooth_label_epsilon'] = 0.1
    config.loss['stage_epoch'] = [0,5]
    for config.net['name'] in [ 'gluoncv_resnet_v10.resnet34' ]:
        for config.train['lrs'] in [  [2e-2 , 2e-2] ]:
            for config.train['lr_bounds'] in [ [0,5,30]  ]:
                config.parse_config()
                config.net['se_kwargs']['reduction_ratio'] = 32
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
