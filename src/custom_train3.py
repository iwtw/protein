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
    config.train['optimizer'] = 'Adam'
    config.train['MIL'] = False
    config.train['save_metric'] = {'macro_f1_score':True , 'bce':False , 'acc':True }#True : saves the max , False : saves the min
    config.net['pretrained'] = False
    config.train['batch_size'] = 32
    config.train['val_batch_size'] = 32
    config.train['freeze_feature_layer_epochs'] = 0
    config.train['lr_for_parts'] = [1,1,1]
    config.train['lr_curve'] = 'one_cycle'
    config.train['lr_find'] = True
    config.loss['name'] = 'Loss_v8'
    config.loss['stage_epoch'] = [0,1000]
    config.data['train_dir'] = ''
    config.data['train_csv_file'] = '../data/train_mix1.csv'
    config.data['image_format'] = 'jpg'

    config.net['input_shape'] = (512,512)
    #config.net['fm_mult'] = 1.0
    config.loss['class_weight'] = True
    
    ilr = 2e-2
    for config.net['name'] in [  'gluoncv_resnet_v10.resnet34' ] :
        for config.train['lrs'] in [  [ ilr, ilr/4 , ilr/16 , ilr/64 ] ]:
            for config.train['lr_bounds'] in [  [0,75]   ] :
                for config.loss['class_weight_dampening'] in ['log' , None]:
                    config.parse_config()
                    for config.net['se_kwargs']['reduction_ratio'] in [ 16 , 20 ,24 ]:
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
