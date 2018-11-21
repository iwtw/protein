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
    config.train['lr_curve'] = 'one_cycle'
    for config.train['lrs'] in [  [2e-2 , 2e-2/4] ]:
        for config.train['lr_bounds'] in [ [0,2,20] , [0,2,30] , [0,2,40] ] :
            for config.loss['weight_l2_reg']  in [ 3e-6 ]:
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
