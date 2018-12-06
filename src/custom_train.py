import train_config as config
import os
from train import main
from time import sleep
from functools import partial
from torch import nn


    
if __name__ == "__main__":

    best_zero = 1e9
    best_result = None

    #for  config.loss['weight_l2_reg'] in [0,1e-4,1e-2]:
    #    for config.train['restart_optimizer'] in [ [2,10,18] , [] ] : for config.train['restart_optimizer'] in [ [2,10,18] , [] ] :
    #        config.parse_config()
    config.net['pretrained'] = False
    config.train['lr_for_parts'] = [1,1,1]
    config.train['lr_curve'] = 'one_cycle'
    config.train['freeze_feature_layer_epochs'] = 0
    config.loss['arcloss_start_epoch'] = 1
    config.loss['s'] = 32
    for config.train['lrs'] in [  [2e-2 , 2e-2] ]:
        for config.train['lr_bounds'] in [ [0,1,70]  , [0,1,100]] :
            config.parse_config()
            print('================================================================================================================================================================================================================================')
            sleep(5)
            try:
                result = main(config)
                if best_zero > result['err']:
                    best_zero = result['err']
                    best_result = result
                
            except KeyboardInterrupt:
                pass
    print('best result :')
    print( best_result )
