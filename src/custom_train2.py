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
    config.train['random_seed'] = 0
    config.train['optimizer'] = 'Adam'
    config.train['MIL'] = False
    config.train['save_metric'] = {'macro_f1_score':True , 'bce':False , 'acc':True }#True : saves the max , False : saves the min
    config.net['pretrained'] = True
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
    config.net['fm_mult'] = 1.0
    config.loss['class_weight_dampening'] = 'log' 
    
    config.train['batch_size'] = 32

    ilr = 1.6e-2
    for config.loss['name'] in [ 'Loss_v8'  ] :
        for config.net['name'] in [  'gluoncv_resnet_v15.resnet34' ] :
                for config.train['lrs'] in [  [ ilr ] ]:
                    for config.train['lr_bounds'] in [  [0,60],[0,70]   ] :
                            config.parse_config()
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
