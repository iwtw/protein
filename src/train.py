import torch
import torch.nn as nn
import numpy as np
from log import *
from utils import *
from dataset import *
from tqdm import tqdm
from time import time
#from network import *
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
from loss import *

def distribution(df):
    count = np.zeros(28)
    temp = df.Target.apply( lambda x : np.array( x.split(' ') , np.uint8 )  )
    for tg in temp:
        for v in tg:
            count[v] += 1 
    return count

def get_class_weight(train_distribution,dampening='log'):
    total_labels = np.sum( train_distribution )

    if dampening == None:
        return None
    elif dampening == 'log':
        mu = 0.5
        weight = np.log( mu * total_labels / train_distribution ) 
        weight[weight<1.0] = 1.0
        weight = torch.Tensor( weight )
    else :
        raise ValueError("dampening type '{}' not supported".format( dampening ))
    return weight


def main(config):

    df = pd.read_csv( config.data['train_csv_file'] , index_col = 0  )
    #df.Target = df.Target.apply( lambda x : np.array( x.split(' ') , np.uint8 )  )
    train_df , val_df =  train_test_split( df , test_size = config.data['test_size'] ,random_state = config.train['random_seed'] , stratify = df['Target'].map(lambda x: x[:3] if '27' not in x else '0' ) )
    train_distribution = distribution( train_df )
    print( "train dsitribution : " , train_distribution )
    print( "val dsitribution : " , distribution( val_df ) ) 
    if config.train['MIL']:
        train_dataset = MILProteinDataset( config , train_df ,  is_training = True , data_dir = config.data['train_dir'] , image_format = config.data['image_format'] )
        val_dataset = MILProteinDataset( config , val_df ,  is_training = False , data_dir = config.data['train_dir'] , image_format = config.data['image_format'] )
    else:

        train_dataset = ProteinDataset( config , train_df ,  is_training = True , data_dir = config.data['train_dir'] , image_format = config.data['image_format'])
        val_dataset = ProteinDataset( config , val_df ,  is_training = False , data_dir = config.data['train_dir'] , image_format = config.data['image_format'])

    train_dataloader = torch.utils.data.DataLoader(  train_dataset , batch_size = config.train['batch_size']  , collate_fn = mil_collate_fn ,  shuffle = True , drop_last = True , num_workers = 8 , pin_memory = False) 
    val_dataloader = torch.utils.data.DataLoader(  val_dataset , batch_size = config.train['val_batch_size']  , shuffle = False , drop_last = False , num_workers = 8 , pin_memory = False) 
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
    net = eval('models.{}'.format(net_name))( **net_kwargs)
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

    

    
    #train_loss_epoch_list = []

    t_list = [time()]
    #convlstm_params = net.module.convlstm.parameters()
    #net_params = net.module.parameters()
    t = time()

    weight = get_class_weight(train_distribution,config.loss['class_weight_dampening'])
    print( 'loss weight : ' , weight )
    compute_loss = eval( config.loss['name'] )(config = config , weight = weight).cuda()

   
    best_metric = {}
    for k,save_max in config.train['save_metric'].items():
        best_metric[k] = ((-1)**save_max ) *1e9  
    log_parser = LogParser()

    origin_curve = config.train['lr_curve']
    for epoch in tqdm(range( last_epoch + 1  , config.train['num_epochs'] ) , file = sys.stdout , desc = 'epoch' , leave=False ):



        #adjust learning rate
        if epoch < config.train['freeze_feature_layer_epochs'] :
            config.train['lr_curve'] = config.train['freeze_lr_curve']
        elif epoch == config.train['freeze_feature_layer_epochs']:
            #lr_find( partial( compute_loss , epoch = 0 ) , net , optimizer , train_dataloader , forward_fn = lambda batch : net( batch['img'] ) , plot_name = '../data/tmp/epoch_{}_begin.png'.format(epoch) )
            config.train['lr_curve'] = origin_curve 

        if config.train['lr_find'] and epoch in config.loss['stage_epoch']:
            lr_find( partial( compute_loss , epoch = epoch ) , net , optimizer , train_dataloader , forward_fn = lambda batch : net( batch['img'] ) , plot_name = '{}/lr_find_epoch_{}.png'.format(tb.path,epoch) )
            torch.cuda.empty_cache()

            

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
            net.train()
            compute_loss.train()
            log_t = time()
            train_loss_log_list = [] 
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

                if config.train['MIL']:
                    bag_sizes = [ len( v ) for v in batch['img'] ]
                    batch['img'] = torch.cat( batch['img'] , 0 )

                if config.train['mix_up']:
                    batch_size = batch['img'].shape[0]
                    '''
                    print('-------------------')
                    print('before mix up')
                    print(batch['label'][:5])
                    print(batch['label'][batch_size//2:batch_size//2+5])
                    '''
                    lambda_ = torch.Tensor( np.random.beta(0.2,0.2,batch_size//2) )
                    #print( 'lambda ' ,  lambda_[:5] )
                    for k,v in batch.items():
                        if isinstance(v,torch.Tensor):
                            lambda_view = lambda_.view( [batch_size//2] + [1 for i in range(len(v.shape)-1)] )
                            batch[k] = lambda_view * batch[k][:batch_size//2]  + ( 1 - lambda_view ) * batch[k][batch_size//2:]

                    '''
                    print('after mix up')
                    print(batch['label'][:5])
                    print(batch['label'][batch_size//2:batch_size//2+5])
                    '''


                for k in batch:
                    if not k in ['filename']:
                        batch[k] = batch[k].cuda(async =  True) 
                        batch[k].detach_() 

                results = net( batch['img'] )

                #aggregate results
                if config.train['MIL']:
                    new_results = {}
                    for k in results:
                        new_results[k] = []
                    cnt = 0
                    for bag_size in  bag_sizes : 
                        for k in results:
                            new_results[k].append( config.train['MIL_aggregate_fn']( results[k][cnt:cnt+bag_size ]  )  )
                            cnt += bag_size
                    new_results = { k:torch.stack( new_results[k] , 0  ) for k in results }
                    results = new_results


                loss_dict = compute_loss( results , batch  , epoch )
                optimizer.zero_grad()
                loss_dict['total'].backward()
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
                    #log_dict[k] = [dic[k] for dic in train_loss_log_list ]
            #log_dict = { k: float( np.mean( [ dic[k] for dic in train_loss_log_list ] )) for k in train_loss_log_list[0] }
            return log_dict

        log_dicts['train'] = train() 

        #validate
        net.eval()
        compute_loss.eval()
        def validate( val_dataloader):

            val_loss_log_list= [ ]
            with torch.no_grad():
                first_val = False
                for step , batch in tqdm( enumerate( val_dataloader ) , total = len( val_dataloader ) , desc = 'validating' , leave = False  ):


                    if config.train['MIL']:
                        bag_sizes = [ len( v ) for v in batch['img'] ]
                        batch['img'] = torch.cat( batch['img'] , 0 )

                    for k in batch:
                        if not k in ['filename']:
                            batch[k] = batch[k].cuda(async =  True) 
                            batch[k].requires_grad = False

                    results = net( batch['img'] )

                    #aggregate results
                    if config.train['MIL']:
                        new_results = {}
                        for k in results:
                            new_results[k] = []
                        cnt = 0
                        for bag_size in  bag_sizes : 
                            for k in results:
                                new_results[k].append( config.train['MIL_aggregate_fn']( results[k][cnt:cnt+bag_size ]  )  )
                                cnt += bag_size
                        new_results = { k:torch.stack( new_results[k] , 0  ) for k in results }
                        results = new_results

                    loss_dict = compute_loss( results , batch , epoch )
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
                        #log_dict[k] = [dic[k] for dic in val_loss_log_list ] 
                        log_dict[k] = np.concatenate(  [dic[k] for dic in val_loss_log_list ] , axis = 0  )
                return log_dict

        log_dicts['val'] =  validate(  val_dataloader  ) 

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

        #save

        for k , save_max in config.train['save_metric'].items():
            cmp_fn = max if save_max else min
            new_metric = log_dicts['val'][k]
            if cmp_fn( best_metric[k] , new_metric ) == new_metric :
                best_metric[k] = new_metric
                torch.save( { 
                    k:best_metric[k],
                    'epoch':epoch,
                    'model':net.state_dict(),
                    'optimizer':optimizer.state_dict()
                } , '{}/models/{}'.format(tb.path,'best_{}.pth'.format(k)))

        #save last snapshot
        torch.save( {
            **best_metric,
            'epoch':epoch,
            'model':net.state_dict(),
            'optimizer':optimizer.state_dict()
        }, '{}/models/{}'.format(tb.path,'last.pth') )
        
        if epoch % 10 == 10 -1 :
            torch.save( {
                **best_metric,
                'epoch':epoch,
                'model':net.state_dict(),
                'optimizer':optimizer.state_dict()
            }, '{}/models/{}'.format(tb.path,'_{}.pth'.format(epoch)) )





    #tb.write_log("best : {}".format( k ,best_metric[k]) )
    tb.write_log("best : {}".format( best_metric ) )
    return { 'log_path':tb.path , **best_metric }


if __name__ == '__main__':
    import train_config as config
    main(config)
