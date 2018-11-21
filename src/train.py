import torch
import torch.nn as nn
import numpy as np
from log import *
from utils import *
from dataset import *
from tqdm import tqdm
from time import time
#from network import *
from custom_utils import *
import sys
from functools import partial
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sklearn
from copy import deepcopy
import torch.nn.functional  as F
import models.gluoncv_resnet
import models.utils
from sklearn.model_selection import train_test_split


def init(config):
    m = config.loss['m']
    global cos_m , sin_m , border  , train_num_classes 
    cos_m = np.cos( m )
    sin_m = np.sin( m )
    border = np.cos( math.pi - m )

        

def compute_loss( results , batch , epoch , config , is_training ):
    loss_dict = {}
    #print(batch['label'])
    loss_dict['focal'] = focal_loss( results['fc'] , batch['label'] )
    loss_dict['err'] = 1 - ( ( results['fc'] > 0.0 ) == batch['label'].byte() ).float().mean()
    loss_dict['total'] = loss_dict['focal']
    return loss_dict




'''
def compute_loss1( results , batch , epoch , config ,  class_range , is_training = False , mse_attribute = True ):
    s, m, k = config.loss['s'] , config.loss['m'] , config.test['k']
    loss_dict = {}
    #print(results['fc'].shape ,  batch['label'].max() )
    if config.net['type'] == 'coarse':
        labels = batch['super_class_label']
    else:
        labels = batch['label']

    if is_training:
        sum_exp = torch.exp(results['fc']).sum(1)
        loss_dict['err'] = 1 - torch.eq( labels  , torch.max( results['fc'] , 1 )[1] ).float().mean()
        topk , top_idx = torch.topk( torch.exp(results['fc'] ) / sum_exp.view(sum_exp.shape[0],1) , k = k , dim = 1 )
        #tqdm.write( "{}".format(topk[0]), file=sys.stdout )
        if epoch < config.loss['arcloss_start_epoch'] :
            loss_dict['softmax'] = cross_entropy(  s*results['fc'] , labels )

        else:
            cos_theta = results['fc']
            cos_theta_yi = cos_theta[( torch.arange(cos_theta.shape[0] , dtype = torch.int64 ) , labels )  ]
            sin_theta_yi = ( 1 - cos_theta_yi**2 ) **0.5
            phai_theta_yi = cos_theta_yi * cos_m - sin_theta_yi * sin_m
            phai_theta_yi = torch.where( cos_theta_yi > border , phai_theta_yi , -2. - phai_theta_yi )#the loss curve correction
            phai_theta = cos_theta
            phai_theta[ ( torch.arange( cos_theta.shape[0] , dtype = torch.int64 ) , labels ) ] = phai_theta_yi
            loss_dict['aam'] = cross_entropy( s * phai_theta , labels )
    else:
        k = config.test['k']
        fc = results['fc']
        sum_exp = torch.exp(fc).sum(1)
        #topk , top_idx = torch.topk( torch.exp( fc ) / sum_exp.view(sum_exp.shape[0],1) , k = k , dim = 1 )
        #for i in range(k):
        #    loss_dict['top{}'.format(i+1)] = torch.mean( topk[:,i] )

        
        #top_idx = top_idx.cpu().detach().numpy()

        predicts = get_predict( results , config , train_dataset.class_attributes , class_range )
        loss_dict['err'] = 1 - torch.eq( predicts , labels ).float().mean() 

    if mse_attribute:
        loss_dict['mse_attribute'] = mse( results['attribute'] , batch['attribute'] )
        #loss_dict['mse_feature_'] = mse( results['feature_'] , results['feature'].detach() )
    return loss_dict

        mix_predicts = get_predict( {'fc':mix_fc}, config , train_dataset.class_attributes )
'''


    

def main(config):
    init(config)
    
    df = pd.read_csv( config.data['train_csv_file'] , index_col = 0  )
    train_df , val_df =  train_test_split( df , test_size = 0.1 , random_state = config.train['random_seed'] )

    train_dataset = ProteinDataset( config , train_df ,  is_training = True , data_dir = config.data['train_dir'] )
    train_dataloader = torch.utils.data.DataLoader(  train_dataset , batch_size = config.train['batch_size']  , shuffle = True , drop_last = True , num_workers = 12 , pin_memory = False) 
    
    val_dataset = ProteinDataset( config , val_df ,  is_training = False , data_dir = config.data['train_dir'] )
    val_dataloader = torch.utils.data.DataLoader(  val_dataset , batch_size = config.train['val_batch_size']  , shuffle = True , drop_last = True , num_workers = 8 , pin_memory = False) 

    '''
    for k in val_dataset_name:
        val_dataset = ZeroDataset(config.train['val_img_list'][k], config, is_training= False , has_filename = True)
        val_dataloaders[k] = torch.utils.data.DataLoader(  val_dataset , batch_size = config.train['val_batch_size']  , shuffle = False , drop_last = True , num_workers = 0 , pin_memory = False) 
    '''


    net_kwargs = deepcopy( config.net )
    net_name = net_kwargs.pop('name')

    #net = eval(net_name)( **net_kwargs )
    #config.net['name'] = net_name
    #net = gluoncvth.models.get_deeplab_resnet34_ade(pretrained=True)
    net = eval('models.gluoncv_resnet.{}'.format(net_name))(pretrained=True , **net_kwargs)
    #net = eval('models.torchvision_resnet.{}'.format( net_name))( pretrained=True , **net_kwargs )
    #net = eval('models.torchvision_resnet.{}'.format(net_name))( pretrained=True , **net_kwargs )
    net = nn.DataParallel( net )
    net.cuda()
    
    tb = TensorBoardX(config = config , log_dir = config.train['log_dir'] , log_type = ['train' , 'val' , 'net'] )
    tb.write_net(str(net),silent=False)

    optim_config = models.utils.get_optim_config(net,config.train['lr_for_parts'])

    for group in optim_config:
        tb.write_log(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format( group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    assert config.train['optimizer'] in ['Adam' , 'SGD']
    if config.train['optimizer'] == 'Adam':
        optimizer_fn = partial( torch.optim.Adam  ,  optim_config , lr = config.train['learning_rate']  , betas = config.train['betas'] ,  weight_decay = 0 , amsgrad = config.train['amsgrad']  )
    if config.train['optimizer'] == 'SGD':
        optimizer_fn = partial( torch.optim.SGD , optim_config , lr = config.train['learning_rate'] , weight_decay = 0 , momentum = config.train['momentum'] , nesterov = config.train['nesterov'] )
    optimizer = optimizer_fn()

    
    #print( optimizer.param_groups )

    last_epoch = -1 
    if config.train['resume'] is not None:
        load_dict = torch.load( config.train['resume'] )
        last_epoch = load_dict['epoch']
        net.load_state_dict( load_dict['model'] )
        if config.train['resume_optimizer'] :
            optimizer.load_state_dict( load_dict['optimizer'] )
        print('Sucessfully load {} , epoch {}'.format(config.train['resume'],last_epoch))



    
    global focal_loss
    focal_loss = FocalLoss().cuda()

    
    #train_loss_epoch_list = []

    t_list = [time()]
    #convlstm_params = net.module.convlstm.parameters()
    #net_params = net.module.parameters()
    t = time()


   
    #best_metric = {}
    #for k in val_dataloaders:
    #    best_metric[k] = 1e9
    best_metric = 1e9
    log_parser = LogParser()

    origin_curve = config.train['lr_curve']
    for epoch in tqdm(range( last_epoch + 1  , config.train['num_epochs'] ) , file = sys.stdout , desc = 'epoch' , leave=False ):



        #adjust learning rate
        if epoch < config.train['freeze_feature_layer_epochs'] :
            config.train['lr_curve'] = config.train['freeze_lr_curve']
        elif epoch == config.train['freeze_feature_layer_epochs']:
            config.train['lr_curve'] = origin_curve 
            

        if epoch < config.train['freeze_feature_layer_epochs']:
            set_requires_grad( net , False )
            set_requires_grad( net.module.classifier , True )
        else:
            set_requires_grad( net , True )
            
        if epoch in config.train['restart_optimizer'] :
            optimizer =  optimizer_fn()

        log_dicts = {}

        #train
        def train():
            log_t = time()
            train_loss_log_list = [] 
            net.train()
            data_loader = train_dataloader
            length = len(train_dataloader)
            for step , batch in tqdm(enumerate( data_loader) , total = length , file = sys.stdout , desc = 'training' , leave=False):
                #adjust learning rate
                mannual_learning_rate(optimizer,epoch,step,length,config)

                tb.add_scalar( 'lr' , optimizer.param_groups[-1]['lr'] , epoch*len(train_dataloader) + step , 'train')
                if 'momentum' in optimizer.param_groups[-1]:
                    tb.add_scalar( 'momentum' , optimizer.param_groups[-1]['momentum'] , epoch*len(train_dataloader) + step , 'train')
                if 'betas' in optimizer.param_groups[-1]:
                    tb.add_scalar( 'beta1' , optimizer.param_groups[-1]['betas'][0] , epoch*len(train_dataloader) + step , 'train')

                for k in batch:
                    if not k in ['filename']:
                        batch[k] = batch[k].cuda(async =  True) 
                        #batch[k].requires_grad = False

                #results = net( batch['img'] , torch.Tensor( train_dataset.class_attributes[:train_num_classes] ).cuda(),  use_normalization = True )
                #print( batch['img'].shape )
                results = net( batch['img'] )

                loss_dict = compute_loss( results , batch  , epoch , config ,  is_training= True )
                #backward( loss_dict , net , optimizer , config )
                optimizer.zero_grad()
                loss_dict['total'].backward()
                #print(next(net.named_parameters()))
                #nn.utils.clip_grad_norm_( map( lambda x : x[1] , filter(lambda x: 'classifier' not in x[0] , net.named_parameters()) ) , config.train['clip_grad_norm'])
                #nn.utils.clip_grad_norm_( filter( lambda m:m.requires_grad , net.parameters() ) , config.train['clip_grad_norm'])
                nn.utils.clip_grad_norm_( net.parameters()  , config.train['clip_grad_norm'])
                for group in optimizer.param_groups:
                    for param in group['params']:
                        param.data.mul_( 1 - group['true_weight_decay'] *  group['lr'])
                optimizer.step()
                loss_dict.pop('total')

                for k in loss_dict:
                    if len(loss_dict[k].shape) == 0 :
                        loss_dict[k] = float(loss_dict[k].cpu().detach().numpy())
                        tb.add_scalar( k , loss_dict[k] , epoch*len(train_dataloader) + step , 'train' )
                    else:
                        loss_dict[k] = loss_dict[k].cpu().detach().numpy()
                train_loss_log_list.append( { k:loss_dict[k] for k in loss_dict} )

                if step % config.train['log_step'] == 0 and epoch == last_epoch + 1 :
                    log_msg = 'step {} lr {} : '.format(step,optimizer.param_groups[0]['lr'])
                    for k in filter( lambda x:isinstance(loss_dict[x],float) and x not in ['err'], loss_dict):
                        log_msg += "{} : {} ".format(k,loss_dict[k] )
                    tqdm.write( log_msg  , file=sys.stdout )
                            
                    for k,v in net.named_parameters():
                        if v.requires_grad and v.grad is not None:
                            try:
                                tb.add_histogram( k , v , (epoch)*len(train_dataloader) + step , 'net' )
                            except Exception as e:
                                print( "{} is not finite".format(k)   )
                                raise e
                            try:
                                tb.add_histogram( k+'_grad' , v.grad , (epoch)*len(train_dataloader) + step , 'net' )
                            except Exception as e:
                                print( "{}.grad is not finite".format(k)   )
                                raise e


            log_dict = {}
            for k in train_loss_log_list[0]:
                if isinstance(train_loss_log_list[0][k] , float ):
                    log_dict[k] = float( np.mean( [dic[k] for dic in train_loss_log_list ]  ) )
                else:
                    log_dict[k] = np.concatenate(  [dic[k] for dic in train_loss_log_list ] , axis = 0  )
            #log_dict = { k: float( np.mean( [ dic[k] for dic in train_loss_log_list ] )) for k in train_loss_log_list[0] }
            return log_dict

        log_dicts['train'] = train() 

        #validate
        net.eval()
        def validate( val_dataloader):

            val_loss_log_list= [ ]
            with torch.no_grad():
                first_val = False
                for step , batch in tqdm( enumerate( val_dataloader ) , total = len( val_dataloader ) , desc = 'validating' , leave = False  ):
                    for k in batch:
                        if not k in ['filename']:
                            batch[k] = batch[k].cuda(async =  True) 
                            batch[k].requires_grad = False

                    results = net( batch['img'] )

                    loss_dict = compute_loss( results , batch , epoch , config , is_training = False )
                    loss_dict.pop('total')


                    for k in loss_dict:
                        if len(loss_dict[k].shape) == 0 :
                            loss_dict[k] = float(loss_dict[k].cpu().detach().numpy())
                        else:
                            loss_dict[k] = loss_dict[k].cpu().detach().numpy()
                    val_loss_log_list.append( { k:loss_dict[k] for k in loss_dict} )
                #log_dict = { k: float( np.mean( [ dic[k] for dic in val_loss_log_list ] )) for k in val_loss_log_list[0] }
                log_dict = {}
                for k in val_loss_log_list[0]:
                    if isinstance(val_loss_log_list[0][k] , float ) :
                        log_dict[k] = float( np.mean( [dic[k] for dic in val_loss_log_list ]  ) )
                    else:
                        log_dict[k] = np.concatenate(  [dic[k] for dic in val_loss_log_list ] , axis = 0  )
                return log_dict

        log_dicts['val'] =  validate(  val_dataloader  ) 

        #save
        if best_metric > log_dicts['val'][config.train['save_metric']]:
            best_metric = log_dicts['val'][config.train['save_metric']]
            torch.save( { 
                config.train['save_metric']:best_metric,
                'epoch':epoch,
                'model':net.state_dict(),
                'optimizer':optimizer.state_dict()
            } , '{}/models/{}'.format(tb.path,'best.pth'))
        torch.save( {
            config.train['save_metric']:log_dicts['val'][config.train['save_metric']],
            'epoch':epoch,
            'model':net.state_dict(),
            'optimizer':optimizer.state_dict()
        }, '{}/models/{}'.format(tb.path,'last.pth') )

        #print to stdout
        num_imgs = config.train['batch_size'] * len(train_dataloader) + config.train['val_batch_size'] * len(val_dataloader) 
        log_msg = log_parser.parse_log_dict( log_dicts , epoch , optimizer.param_groups[-1]['lr'] , num_imgs , config = config )
        tb.write_log(  log_msg  , use_tqdm = True )

        #log to tensorboard
        log_net_params(tb,net,epoch,len(train_dataloader))

        for tag in log_dicts:
            if 'val' in tag:
                for k,v in log_dicts[tag].items():
                    if isinstance( v  , float ) :
                        tb.add_scalar( k , v , (epoch+1)*len(train_dataloader) , tag ) 
                    else:
                        tb.add_histogram( k , v , (epoch+1)*len(train_dataloader) , tag )


    #tb.write_log("best : {}".format( k ,best_metric[k]) )
    tb.write_log("best : {}".format( best_metric ) )
    return { 'log_path':tb.path , config.train['save_metric']:best_metric }


if __name__ == '__main__':
    import train_config as config
    main(config)
