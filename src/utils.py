""" utils.py
"""

import os
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import time
import warnings
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

from copy import deepcopy
import sys

def aggregate_results( results , bag_sizes , aggregate_fn):
    new_results = {}
    for k in results:
        new_results[k] = []
    cnt = 0
    for bag_size in  bag_sizes : 
        for k in results:
            new_results[k].append( aggregate_fn( results[k][cnt:cnt+bag_size ]  )  )
            cnt += bag_size
    new_results = { k:torch.stack( new_results[k] , 0  ) for k in results }
    return new_results

def mannual_learning_rate( optimizer , epoch ,  step , num_step_epoch , config ):
    
    bounds = config.train['lr_bounds'] 
    lrs = config.train['lrs'] 
    for idx in range(len(bounds) - 1):
        if bounds[idx] <= epoch and epoch < bounds[idx+1]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lrs[idx] * param_group['lr_mult']
                param_group['true_weight_decay'] = config.loss['weight_l2_reg'] * param_group['decay_mult']
                if 'betas' in param_group:
                    param_group['betas'] = deepcopy( config.train['betas'] )
                    #print(config.train['betas'] )
                if 'momentum' in param_group:
                    param_group['momentum'] = config.train['momentum']
            break

    if config.train['lr_curve'] == 'normal':
        pass
    elif config.train['lr_curve'] == 'cosine':
        for param_group in optimizer.param_groups:
            length = config.train['lr_bounds'][idx+1] -  config.train['lr_bounds'][idx]
            param_group['lr'] *= np.cos( np.pi / 2 / (length * num_step_epoch) * (step + num_step_epoch * ( epoch - bounds[idx] )) )

    elif config.train['lr_curve'] == 'cyclical':
        length = (config.train['lr_bounds'][idx+1] -  config.train['lr_bounds'][idx]) * num_step_epoch
        x = (epoch - bounds[idx] )*num_step_epoch + step
        factor = config.train['cyclical_lr_init_factor']
        mid_x = length * config.train['cyclical_lr_inc_ratio']
        mom_min = config.train['cyclical_mom_min']
        mom_max = config.train['cyclical_mom_max']
        if x <= mid_x:
            y1 = x / mid_x * ( 1 - factor ) + factor
            k2 = ( mom_min - mom_max ) / mid_x 
            b2 = mom_max
            y2 = k2*x + b2
        else:
            k1 = ( 1 - factor )/ ( mid_x - length )
            b1 = (mid_x * factor - length ) / (mid_x - length)
            y1 = k1*x + b1  
            k2 = ( mom_min - mom_max ) / ( mid_x - length )
            b2 = ( mid_x - length * mom_min ) / ( mid_x - length )
            y2 = k2 * x + b2
        for param_group in optimizer.param_groups:
            param_group['lr'] *= y1
            if 'betas' in param_group:
                param_group['betas'][0] = y2
            if 'momentum' in param_group:
                param_group['momentum'] = y2
    elif config.train['lr_curve'] == 'one_cycle':
        length = (config.train['lr_bounds'][idx+1] -  config.train['lr_bounds'][idx]) * num_step_epoch
        x = (epoch - bounds[idx] )*num_step_epoch + step
        factor = config.train['cyclical_lr_init_factor']
        mid_x = length * config.train['cyclical_lr_inc_ratio']
        mom_min = config.train['cyclical_mom_min']
        mom_max = config.train['cyclical_mom_max']
        if x <= mid_x:
            y1 = x / mid_x * ( 1 - factor ) + factor
            k2 = ( mom_min - mom_max ) / mid_x 
            b2 = mom_max
            y2 = k2*x + b2
        else:
            y1 = ( np.cos( np.pi * (x - mid_x) / ( length - mid_x ) ) + 1 ) / 2 * ( 1 ) + 0 #from lr_max to 0 
            y2 = ( np.cos( np.pi + np.pi * ( x - mid_x ) / ( length - mid_x ) ) + 1  ) / 2 * ( mom_max - mom_min )   + mom_min
        for param_group in optimizer.param_groups:
            param_group['lr'] *= y1
            if 'betas' in param_group:
                param_group['betas'][0] = y2
            if 'momentum' in param_group:
                param_group['momentum'] = y2


def lr_find(loss_fn,net,optimizer,dataloader,forward_fn,warp_batch_fn = None , start_lr=1e-5,end_lr = 10 , num_iter = None ,  plot_name = None ):

    origin_net_state = deepcopy( net.state_dict() )
    origin_optimizer_state = deepcopy( optimizer.state_dict() )
    best_loss = 1e9
    loss_list = []

    if num_iter is None:
        num_iter = len(dataloader)
    ratio = end_lr / start_lr 
    lr_mult = ratio ** (1/num_iter)
    
    lr_list = []
    tqdm_it = tqdm( dataloader  , desc ='finding lr' , total = num_iter , file=sys.stdout , leave=False )
    for it , x  in enumerate(tqdm_it):
        lr = start_lr * lr_mult ** it
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
        temp_loss_list = []

        if warp_batch_fn is not None:
            x = warp_batch_fn( x )
        for k in x:
            if isinstance( x[k] , torch.Tensor ):
                x[k] = x[k].cuda()
                x[k].requires_grad = False
        loss_dict = loss_fn( forward_fn( x ) , x ) 
        #stop criterion
        loss = loss_dict['total']
        if math.isnan( loss ) or ( it > 10 and loss > 1.5*loss_list[0] ) or it > num_iter :
            del loss
            del loss_dict
            del x
            tqdm_it.close()
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        lr_list.append( lr )
        loss = loss.detach().cpu().numpy()
        loss_list.append( loss )

        #update best_loss
        if it > 10 and loss < best_loss:
            best_loss = loss


        


    net.load_state_dict( origin_net_state )
    optimizer.load_state_dict( origin_optimizer_state  )

    '''
    steepest_idx = np.argmin(loss_delta_list)
    lowest_idx = np.argmin( loss_list )
    final_lr = lr_list[int(np.floor( steepest_idx + (lowest_idx - steepest_idx) * 3/4   ))  ] 
    tqdm.write('steepest_lr {} lowest_lr {} final_lr {}'.format(lr_list[steepest_idx] , lr_list[lowest_idx] , final_lr))
    return final_lr
    '''
    sm = 0.90
    sm_loss_list = []
    for i in range(len(loss_list)):
        sm_loss_list.append( np.mean( loss_list[max(i-5,0):i+1] ) )
    d_sm_loss_list = [0] + [ sm_loss_list[i] - sm_loss_list[i-1] for i in range(1,len(sm_loss_list)) ]
    dd_sm_loss_list = [0] + [ d_sm_loss_list[i ] - d_sm_loss_list[i-1] for i in range(1,len(d_sm_loss_list)) ]


    plt.figure()
    f , axes = plt.subplots(1,1)
    axes.set_title('smoothed loss curve')
    axes.set_xlabel('learning rate')
    axes.set_ylabel('loss')
    axes.plot(  lr_list,sm_loss_list)
    axes.set_xscale('log')

    if plot_name is not None:
        plt.savefig( plot_name )


            





def detect_change(net , mode = 3  ):
    for name, module in net._modules.items():
        grads_list = []
        params_list = []
        for param in module.parameters():
            if param.grad is not None:
                grads_list.append( param.grad.view(-1) )
            params_list.append( param.view(-1) )
        if mode & 1 :
            if len(grads_list) == 0:
                norm = 0
            else:
                norm = torch.norm( torch.cat( grads_list , 0 ) )
            print( "  {} grads norm : {}".format(name,norm) )
        if mode & 2:
            norm = torch.norm( torch.cat( params_list , 0 ) )
            print( '  {} params norm : {}'.format( name , norm) )
        

name_dataparallel = torch.nn.DataParallel.__name__
def lr_warmup(epoch, warmup_length):
    if epoch < warmup_length:
        p = max(0.0, float(epoch)) / float(warmup_length)
        p = 1.0 - p
        return np.exp(-p*p*5.0)
    else:
        return 1.0
    torch.nn.PReLU
    

def load_optimizer(optimizer , model , path , epoch = None ):
    """
    return the epoch
    """
    if type(model).__name__ == name_dataparallel:
        model = model.module

    if epoch is None:
        for i in reversed( range(1000) ):
            p = "{}/{}_epoch{}.pth".format( path,type(optimizer).__name__+'_'+type(model).__name__,i )
            if os.path.exists( p ):
                optimizer.load_state_dict(  torch.load( p ) )
                print('Sucessfully resume optimizer {}'.format(p))
                return i
    else:
        p = "{}/{}_epoch{}.pth".format( path,type(optimizer).__name__+'_'+type(model).__name__,epoch )
        if os.path.exists( p ):
            optimizer.load_state_dict(  torch.load( p )   )
            print('Sucessfully resume optimizer {}'.format(p))
            return epoch
        else:
            warnings.warn("resume optimizer not found at {}".format(p))

    warnings.warn("resume model not found ")
    return -1

def load_model(model,path ,epoch = None , strict= True): 
    """
    return the last epoch
    """
    if type(model).__name__ == name_dataparallel:
        model = model.module
    if epoch is None:
        for i in reversed( range(1000) ):
            p = "{}/{}_epoch{}.pth".format( path,type(model).__name__,i )
            if os.path.exists( p ):
                model.load_state_dict(  torch.load( p ) , strict = strict)
                print('Sucessfully resume model {}'.format(p))
                return i
    else:
        p = "{}/{}_epoch{}.pth".format( path,type(model).__name__,epoch )
        if os.path.exists( p ):
            model.load_state_dict(  torch.load( p ) , strict = strict)
            print('Sucessfully resume model {}'.format(p))
            return epoch
        else:
            warnings.warn("resume model not found at {}".format(p))

    warnings.warn("resume model not found ")
    return -1

    
def set_requires_grad(module , b ):
    for parm in module.parameters():
        parm.requires_grad = b

def adjust_dyn_range(x, drange_in, drange_out):
    if not drange_in == drange_out:
        scale = float(drange_out[1]-drange_out[0])/float(drange_in[1]-drange_in[0])
        bias = drange_out[0]-drange_in[0]*scale
        x = x.mul(scale).add(bias)
    return x


def resize(x, size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Scale(size),
        transforms.ToTensor(),
        ])
    return transform(x)


def make_image_grid(x, ngrid):
    x = x.clone().cpu()
    if pow(ngrid,2) < x.size(0):
        grid = make_grid(x[:ngrid*ngrid], nrow=ngrid, padding=0, normalize=True, scale_each=False)
    else:
        grid = torch.FloatTensor(ngrid*ngrid, x.size(1), x.size(2), x.size(3)).fill_(1)
        grid[:x.size(0)].copy_(x)
        grid = make_grid(grid, nrow=ngrid, padding=0, normalize=True, scale_each=False)
    return grid


def save_image_single(x, path, imsize=512):
    from PIL import Image
    grid = make_image_grid(x, 1)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize((imsize,imsize), Image.NEAREST)
    im.save(path)


def save_image_grid(x, path, imsize=512, ngrid=4):
    from PIL import Image
    grid = make_image_grid(x, ngrid)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize((imsize,imsize), Image.NEAREST)
    im.save(path)


def save_model(model,dirname,epoch,mode = 'epoch'):
    if type(model).__name__ == name_dataparallel:
        model = model.module
    model_pathname = '{}/{}_{}{}.pth'.format(dirname,type(model).__name__,mode,epoch )
    torch.save( model.state_dict() , '{}/{}_{}{}.pth'.format(dirname,type(model).__name__,mode,epoch ) )

def del_model(model,dirname,epoch,mode = 'epoch'):
    if type(model).__name__ == name_dataparallel:
        model = model.module
    model_pathname = '{}/{}_{}{}.pth'.format(dirname,type(model).__name__,mode,epoch )
    if os.path.exists( model_pathname ):
        os.system('rm {}'.format(model_pathname))

def save_optimizer(optimizer,model,dirname,epoch,mode='epoch'):
    if type(model).__name__ == name_dataparallel:
        model = model.module
    torch.save( optimizer.state_dict() , '{}/{}_epoch{}.pth'.format(dirname,type(optimizer).__name__ +'_' +type(model).__name__,epoch ) )

def del_optimizer(optimizer,model,dirname,epoch,mode='epoch'):
    if type(model).__name__ == name_dataparallel:
        model = model.module
    model_pathname = '{}/{}_epoch{}.pth'.format(dirname,type(optimizer).__name__ +'_' +type(model).__name__,epoch )
    if os.path.exists( model_pathname ):
        os.system('rm {}'.format(model_pathname))


def make_summary(writer, key, value, step):
    if hasattr(value, '__len__'):
        for idx, img in enumerate(value):
            summary = tf.Summary()
            sio = BytesIO()
            scipy.misc.toimage(img).save(sio, format='png')
            image_summary = tf.Summary.Image(encoded_image_string=sio.getvalue())
            summary.value.add(tag="{}/{}".format(key, idx), image=image_summary)
            writer.add_summary(summary, global_step=step)
    else:
        summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
        writer.add_summary(summary, global_step=step)




import torch
import math
irange = range
def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        return tensor
    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                #for pytorch 0.3.0
                #norm_ip(t, t.min(), t.max())
                #for pytorch 0.4.0
                norm_ip(t, float(t.min()) , float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)
