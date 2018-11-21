import train_config as config
import os
from train import main
from time import sleep
from functools import partial
from torch import nn


    
if __name__ == "__main__":

    best_zero = 1e9
    best_result = None

    for  config.train['random_seed'] in [0,1,2]:
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
