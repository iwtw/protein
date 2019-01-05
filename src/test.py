import torch
import argparse
from dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import load_model,aggregate_results,set_requires_grad
from time import time
import os
import train_config as config
import models.gluoncv_resnet
from sklearn.model_selection import train_test_split
from copy import deepcopy
import scipy.optimize as opt
import torch.nn.functional as F
from sklearn.metrics import f1_score
import torch.nn as nn
from functools import partial

th_magic = np.array([0.565,0.39,0.55,0.345,0.33,0.39,0.33,0.45,0.38,0.39,
               0.34,0.42,0.31,0.38,0.49,0.50,0.38,0.43,0.46,0.40,
               0.39,0.505,0.37,0.47,0.41,0.545,0.32,0.1])

lb_prob = np.array( [
 0.362397820,0.043841336,0.075268817,0.059322034,0.075268817,
 0.075268817,0.043841336,0.075268817,0.010000000,0.010000000,
 0.010000000,0.043841336,0.043841336,0.014198783,0.043841336,
 0.010000000,0.028806584,0.014198783,0.028806584,0.059322034,
 0.010000000,0.126126126,0.028806584,0.075268817,0.010000000,
 0.222493880,0.028806584,0.010000000] )

def sigmoid_np(x):
    return 1.0/(1.0 + np.exp(-x))

def F1_soft(preds,targs,th=0.5,d=50.0):
    preds = sigmoid_np(d*(preds - th))
    targs = targs.astype(np.float)
    score = 2.0*(preds*targs).sum(axis=0)/((preds+targs).sum(axis=0) + 1e-6)
    return score

def fit_val(x,y,num_classes):
    params = 0.5*np.ones(num_classes)
    wd = 1e-5
    error = lambda p: np.concatenate((F1_soft(x,y,p) - 1.0,
                                      wd*(p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)
    return p

def Count_soft(preds,th=0.5,d=50.0):
    preds = sigmoid_np(d*(preds - th))
    return preds.mean(axis=0)

def fit_test(x,y,num_classes):
    params = 0.5*np.ones(num_classes)
    wd = 1e-5
    error = lambda p: np.concatenate((Count_soft(x,p) - y,
                                      wd*(p - 0.5)), axis=None)
    p, success = opt.leastsq(error, params)
    return p

def save_pred(pred,th,out_name):
    out_fp = open( out_name , 'w')
    out_fp.write('Id,Predicted\n')

    filenames = pd.read_csv( '../data/test.csv' , index_col = 0 ).index.values
    Id = test_df.index.values

    pred_dict = { k : v for k,v in zip(Id,pred)}

    for filename  in filenames: 
        if filename in pred_dict:
            fc = pred_dict[filename]
            predict = np.arange( config.net['num_classes'] )[ fc > th ]
            p = predict.tolist()
            if isinstance(p,int) :
                p = [p]
            if len(p) == 0 :
                p = [ fc.argmax().tolist() ] 

            p = [ x for x in map( lambda x : str(x) , p ) ] 
        else:
            p = []
        out_fp.write( filename+',' + ' '.join(p)+'\n' )


    out_fp.close()

def main(config):
    #args = parse_args()


    global test_df

    if config.train['MIL']:
        #df = pd.read_csv( '../data/train_single_cell_crop.csv' , index_col = 0  )
        test_df = pd.read_csv( '../data/test_single_cell_crop.csv' , index_col = 0  )
        train_data_dir =  '/mnt/ssd_raid0/wtw/datasets/human_protein/train_single_cell_crop'
        test_data_dir = '/mnt/ssd_raid0/wtw/datasets/human_protein/test_single_cell_crop'
        dataset_fn = MILProteinDataset
        dataloader_fn = partial( torch.utils.data.DataLoader , collate_fn = mil_collate_fn )
    else:
        #df = pd.read_csv( '../data/train.csv' , index_col = 0  )
        test_df = pd.read_csv( '../data/test.csv' , index_col = 0  )
        train_data_dir =  '../data/train'
        test_data_dir = '../data/test'
        dataset_fn = ProteinDataset
        dataloader_fn = torch.utils.data.DataLoader

    df = pd.read_csv( config.data['train_csv_file'] , index_col = 0  )
    
    train_df , val_df =  train_test_split( df , test_size = config.data['test_size'] ,random_state = config.train['random_seed'] , stratify = df['Target'].map(lambda x: x[:3] if '27' not in x else '0' ) )
    print(len(val_df))
    original_train_df = pd.read_csv( '../data/train.csv', index_col = 0 )
    val_df = val_df[val_df.index.map( lambda x : x in original_train_df.index )]
    print(len(val_df))

    val_dataset = dataset_fn( config , val_df ,  is_training = False , tta = config.test['tta'] , data_dir = train_data_dir )
    val_dataloader = dataloader_fn(  val_dataset , batch_size = config.test['batch_size']  , shuffle = False , drop_last = False , num_workers = 8 , pin_memory = False) 

    test_dataset = dataset_fn( config , test_df ,  is_training = False , tta = config.test['tta'] , data_dir = test_data_dir , has_label = False )
    test_dataloader = dataloader_fn(  test_dataset , batch_size = config.test['batch_size']  , shuffle = False , drop_last = False , num_workers = 8 , pin_memory = False) 

            

    net_kwargs = deepcopy( config.net )
    net_name = net_kwargs.pop('name')

    net = eval("models.{}".format(net_name))(**net_kwargs)
    net = nn.DataParallel( net )
    net.cuda()
    set_requires_grad( net , False )
    


    load_dict = torch.load(config.test['model']) 
    #for k in load_dict['model']:
    #    print(k)
    net.load_state_dict( load_dict['model'] , strict = True )
    print( 'Sucessfully load {} , epoch {}'.format(config.test['model'],load_dict['epoch']) )


    net.eval()
    val_pred = []
    val_label = []
    acc_list = []
    with torch.no_grad():
        for step , batch in tqdm(enumerate( val_dataloader ) , total = len(val_dataloader) ):

            #print( type( batch['img'][0] ) )
            #TTA
            if config.train['MIL']:
                bag_sizes = [ len( v ) for v in batch['img'] ]
                batch['img'] = torch.cat( batch['img'] , 0 )

            for k in batch:
                if k in ['img']:
                    batch[k] = batch[k].cuda(async=True)
                    batch[k].requires_grad = False

            
            results_list = []
            for i in range( config.test['tta'] ):
                results = net( batch['img'][:,i] )
                for k in results:
                    results[k] = results[k].detach().cpu()
                if config.train['MIL']:
                    results = aggregate_results( results , bag_sizes , config.train['MIL_aggregate_fn'])
                results_list.append( results )
            
            batch_fc = [ x['fc'] for x in results_list ]
            batch_fc = torch.stack( batch_fc , dim = -1  )
            #print( batch_fc.shape )
            #batch_fc = batch_fc.max( dim = - 1 )[0]
            batch_fc = F.sigmoid( batch_fc )
            batch_fc = batch_fc.mean( dim = - 1 )
            val_pred.append( batch_fc.detach().cpu().numpy() ) 
            val_label.append( batch['label'].numpy() )
            acc = ( ( batch_fc.cpu().detach() > 0.5 ) == (batch['label'] > 0.5 ) ).float().mean()
            acc_list.append( acc.numpy() )
            #tqdm.write( str( acc ) )

    tqdm.write( "val acc : " + str(np.mean( acc_list )) )
    val_pred = np.concatenate( val_pred , axis = 0 ) 
    val_label = np.concatenate( val_label , axis = 0 ) > 0.5
    
    th = fit_val(val_pred,val_label,config.net['num_classes'])
    th[th<0.1] = 0.1
    print('Thresholds: ',th)
    print('F1 macro: ',f1_score(val_label, val_pred>th, average='macro'))
    print('F1 macro: (th = th_magic)',f1_score(val_label, val_pred>th_magic, average='macro'))
    print('F1 macro (th = 0.5): ',f1_score(val_label, val_pred>0.5, average='macro'))
    print('F1 micro: ',f1_score(val_label, val_pred>th, average='micro'))
    print('Fractions: \n',(val_pred > th).mean(axis=0))
    print('Fractions (true): \n',(val_label > th).mean(axis=0))

    labels = pd.read_csv('../data/train.csv')
    label_count = np.zeros(config.net['num_classes'])
    for label in labels['Target']:
        l = [int(i) for i in label.split()]
        label_count += np.eye(config.net['num_classes'])[l].sum(axis=0)
    label_fraction = label_count.astype(np.float)/len(labels)
    print('Fractions (train): \n',label_fraction)
    print('Fractions (lb_prob): \n',lb_prob)

    tt = time()
    test_pred = []
    with torch.no_grad():
        for step , batch in tqdm(enumerate( test_dataloader ) , total = len(test_dataloader) ):

            if config.train['MIL']:
                bag_sizes = [ len( v ) for v in batch['img'] ]
                batch['img'] = torch.cat( batch['img'] , 0 )

            for k in batch:
                if k in ['img']:
                    batch[k] = batch[k].cuda(async=True)
                    batch[k].requires_grad = False

            
            results_list = []
            for i in range( config.test['tta'] ):
                results = net( batch['img'][:,i] )
                for k in results:
                    results[k] = results[k].detach().cpu()
                if config.train['MIL']:
                    results = aggregate_results( results , bag_sizes , config.train['MIL_aggregate_fn'])
                results_list.append( results )
            
            batch_fc = [ x['fc'] for x in results_list ]
            batch_fc = torch.stack( batch_fc , dim = -1  )
            #batch_fc = batch_fc.max( dim = -1 )[0]
            batch_fc = F.sigmoid( batch_fc )
            batch_fc = batch_fc.mean( dim = - 1 )
            for x in batch_fc.numpy() :
                test_pred.append( x )

    #label_count, label_fraction

    th_train = fit_test( test_pred , label_fraction , config.net['num_classes'] )
    print( 'threshold train :\n' , th_train )
    th_lb = fit_test( test_pred , lb_prob , config.net['num_classes'])
    save_pred( test_pred , th_train , '../submit/{}'.format( config.test['model'].split('/')[-3] + '_train.csv' ) )
    save_pred( test_pred , th, '../submit/{}'.format( config.test['model'].split('/')[-3] + '_val.csv' ) )
    save_pred( test_pred , th_lb , '../submit/{}'.format( config.test['model'].split('/')[-3] + '_lb_prob.csv' ) )
    save_pred( test_pred , 0.5 , '../submit/{}'.format( config.test['model'].split('/')[-3] + '_05.csv' ) )
    return test_pred

if __name__ == '__main__' :
    main(config)

