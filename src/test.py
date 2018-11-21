import torch
import argparse
from dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import load_model
from time import time
import os
import train_config as config
import models.gluoncv_resnet
from sklearn.model_selection import train_test_split
from copy import deepcopy
import scipy.optimize as opt
import torch.nn.functional as F
from sklearn.metrics import f1_score

th_t = np.array([0.565,0.39,0.55,0.345,0.33,0.39,0.33,0.45,0.38,0.39,
               0.34,0.42,0.31,0.38,0.49,0.50,0.38,0.43,0.46,0.40,
               0.39,0.505,0.37,0.47,0.41,0.545,0.32,0.1])

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

if __name__ == '__main__' :
    #args = parse_args()


    df = pd.read_csv( config.data['train_csv_file'] , index_col = 0  )
    train_df , val_df =  train_test_split( df , test_size = 0.1 , random_state = config.train['random_seed'] )
    test_df = pd.read_csv( config.data['test_csv_file'] , index_col = 0  )


    val_dataset = ProteinDataset( config , val_df ,  is_training = False , data_dir = config.data['train_dir'] )
    val_dataloader = torch.utils.data.DataLoader(  val_dataset , batch_size = config.train['val_batch_size']  , shuffle = False , drop_last = False , num_workers = 16 , pin_memory = False) 

    test_dataset = ProteinDataset( config , test_df ,  is_training = False , data_dir = config.data['test_dir'] , has_label = False )
    test_dataloader = torch.utils.data.DataLoader(  test_dataset , batch_size = config.test['batch_size']  , shuffle = False , drop_last = False , num_workers = 16 , pin_memory = False) 
    
            
    net_kwargs = deepcopy( config.net )
    net_name = net_kwargs.pop('name')

    net = eval("models.gluoncv_resnet.{}".format(net_name))(pretrained=True , **net_kwargs)
    net = nn.DataParallel( net )
    net.cuda()
    


    #last_epoch = load_model( net , args.resume , epoch = args.resume_epoch  , strict = False) 
    load_dict = torch.load(config.test['model']) 
    net.load_state_dict( load_dict['model'] , strict = True )
    print( 'Sucessfully load {} , epoch {}'.format(config.test['model'],load_dict['epoch']) )

    #log_dir = '{}/test/{}'.format(args.resume,args.dataset)
    #os.system('mkdir -p {}'.format(log_dir) )

    net.eval()
    val_pred = []
    val_label = []
    acc_list = []
    with torch.no_grad():
        for step , batch in tqdm(enumerate( val_dataloader ) , total = len(val_dataloader) ):
            for k in batch:
                if k == 'img':
                    batch[k] = batch[k].cuda(async = True)
                    batch[k].requires_grad = False

            results = net( batch['img'] ) 
            val_pred.append( F.sigmoid( results['fc']).detach().cpu().numpy() ) 
            val_label.append( batch['label'].numpy() )
            acc = ( ( results['fc'].cpu() > 0.0 ) == batch['label'].byte() ).float().mean()
            acc_list.append( acc.numpy() )
            #tqdm.write( str( acc ) )

    tqdm.write( "val acc : " + str(np.mean( acc_list )) )
    val_pred = np.concatenate( val_pred , axis = 0 ) 
    val_label = np.concatenate( val_label , axis = 0 )
    th = fit_val(val_pred,val_label,config.net['num_classes'])
    th[th<0.1] = 0.1
    print('Thresholds: ',th)
    print('F1 macro: ',f1_score(val_label, val_pred>th, average='macro'))
    print('F1 macro: (th = th_t)',f1_score(val_label, val_pred>th_t, average='macro'))
    print('F1 macro (th = 0.5): ',f1_score(val_label, val_pred>0.5, average='macro'))
    print('F1 micro: ',f1_score(val_label, val_pred>th, average='micro'))

    tt = time()
    out_fp = open('../submit/{}.txt'.format( config.test['model'].split('/')[-3] ) , 'w')
    out_fp.write('Id,Predicted\n')
    with torch.no_grad():
        for step , batch in tqdm(enumerate( test_dataloader ) , total = len(test_dataloader) ):
            for k in batch:
                if k == 'img':
                    batch[k] = batch[k].cuda(async = True)
                    batch[k].requires_grad = False

            results = net( batch['img'] ) 
            #tqdm.write( str((results['fc'] > 0).float().float().mean()  ) ) 
            assert len(batch['filename']) == len(results['fc'])
            for idx , v in enumerate( zip( batch['filename'] , results['fc'] )): 
                filename , fc = v
                #print(fc.shape)
                #predict = np.arange( config.net['num_classes'] )[ fc.cpu().numpy()>0.0 ]
                predict = np.arange( config.net['num_classes'] )[ F.sigmoid(fc).cpu().numpy() > th ]
                p = predict.tolist()
                if isinstance(p,int) :
                    p = [p]
                if len(p) == 0 :
                    p = [ fc.argmax().cpu().numpy().tolist() ] 

                p = [ x for x in map( lambda x : str(x) , p ) ] 

                #elif len(p) == 0 :
                #    p = [ results['fc'][idx].max()[1].detach().cpu().numpy().tolist() ] 
                out_fp.write( filename+',' + ' '.join(p)+'\n' )

    out_fp.close()
            #feats_list.append( results['feature'].detach().cpu().numpy() )
    #feats = np.vstack( feats_list )
    #np.savetxt(args.output_list, feats, fmt='%.18e', delimiter=',')
    #os.system('python /home/wtw/scripts/test_lfw.py {} {} {}'.format(args.output_list,'/home/hzl/dataset/lfw-list.txt','/home/hzl/dataset/lfw-pairs.txt'))


    #with open('{}/psnr.txt'.format(log_dir),'a') as log_fp:
    #    log_fp.write( 'epoch {} : psnr {}'.format( last_epoch , psnr ) )
