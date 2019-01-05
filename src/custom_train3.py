import train_config as config
import os
from train import main
from time import sleep
from functools import partial
from torch import nn


    
if __name__ == "__main__":

    best_zero = -1e9
    best_result = None
    batch_size = 32
    config.train['val_batch_size'] = batch_size
    config.train['batch_size'] = batch_size

    #for  config.loss['weight_l2_reg'] in [0,1e-4,1e-2]:
    #    for config.train['restart_optimizer'] in [ [2,10,18] , [] ] : for config.train['restart_optimizer'] in [ [2,10,18] , [] ] :
    #        config.parse_config()
    config.train['optimizer'] = 'Adam'
    config.train['MIL'] = False
    config.train['save_metric'] = {'macro_f1_score':True , 'bce':False , 'acc':True }#True : saves the max , False : saves the min
    config.net['pretrained'] = False
    config.train['freeze_feature_layer_epochs'] = 0
    config.train['lr_for_parts'] = [1,1,1]
    config.train['lr_curve'] = 'one_cycle'
    config.train['lr_find'] = False
    config.loss['name'] = 'Loss_v8'
    config.loss['stage_epoch'] = [0,1000]
    config.data['train_dir'] = ''
    config.data['train_csv_file'] = '../data/train_mix1.csv'
    config.data['image_format'] = 'jpg'

    config.net['input_shape'] = (512,512)
    config.net['fm_mult'] = 1.0
    config.loss['class_weight_dampening'] = 'log' 
    
    ilr = 2e-2
    config.train['lr_bounds'] = [0,75] 
    config.train['lrs'] =  [ ilr ] 

    config.loss['name'] =  'Loss_v8'  
    config.net['name'] =  'gluoncv_resnet_v13.resnet34' 
    for config.train['mix_up'] in [True,False]:
        for config.data['smooth_label_epsilon'] in [0.1 , 0.0 ]:
            if config.train['mix_up']:
                config.train['batch_size'] = batch_size * 2
                config.train['lr_bounds'] = [0,125]
            else:
                config.train['lr_bounds'] = [0,75] 
                config.train['batch_size'] = batch_size


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
