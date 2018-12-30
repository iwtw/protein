import train_config as config
import os
from train import main
from time import sleep
from functools import partial
from torch import nn
from tqdm import tqdm
import argparse
import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name',type=str,choices=['gcn','tede'],default='tede')
    parser.add_argument('-num_models',type=int,default=10)
    time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_info = '../data/bagging_results/' + time + '.txt'
    parser.add_argument('-output_list',default = model_info )
    parser.add_argument('--start',type=int,default=0)
    return parser.parse_args()

    
if __name__ == "__main__":
    args = parse_args()
    cmd = 'mkdir -p {}'.format( '/'.join( args.output_list.split('/')[:-1] ))
    print(cmd)
    os.system(cmd )

    log = []
    for config.train['split_random_seed'] in tqdm(range(args.start,args.num_models) , desc='bagging' , leave=False):
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
        config.net['fm_mult'] = 1.0
        config.loss['class_weight'] = True

        results = main(config)
        
        log.append( str(results) )
        with open(args.output_list,'w') as fp:
            write_msg = '\n'.join( [ " ".join(v) for v in log ] ) 
            fp.write(write_msg+'\n')
            fp.flush()
    print( log )



        
