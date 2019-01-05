import torch.utils.data as data
import cv2
import torch


import numpy as np
import torchvision
import torchvision.transforms
from sklearn.decomposition import PCA
import pandas as pd
from copy import deepcopy
from math import ceil
import os

from torch._six import container_abcs
from torch._six import string_classes, int_classes, FileNotFoundError
from torch.utils.data.dataloader import default_collate
import re



class ProteinDataset(data.Dataset):
    def __init__(self , config , df  , is_training , tta = 0,  data_dir = "" , image_format = 'png' , has_label = True ):
        
        self.config = config
        self.label_to_name_dict = {
            0:  'Nucleoplasm',
            1:  'Nuclear membrane',
            2:  'Nucleoli',   
            3:  'Nucleoli fibrillar center',
            4:  'Nuclear speckles',
            5:  'Nuclear bodies',
            6:  'Endoplasmic reticulum',   
            7:  'Golgi apparatus',
            8:  'Peroxisomes',
            9:  'Endosomes',
            10:  'Lysosomes',
            11:  'Intermediate filaments',
            12:  'Actin filaments',
            13:  'Focal adhesion sites',   
            14:  'Microtubules',
            15:  'Microtubule ends',  
            16:  'Cytokinetic bridge',   
            17:  'Mitotic spindle',
            18:  'Microtubule organizing center',  
            19:  'Centrosome',
            20:  'Lipid droplets',
            21:  'Plasma membrane',   
            22:  'Cell junctions', 
            23:  'Mitochondria',
            24:  'Aggresome',
            25:  'Cytosol',
            26:  'Cytoplasmic bodies',   
            27:  'Rods & rings'
        }
        self.has_label = has_label
        self.is_training = is_training
        self.tta = tta
        self.df = df
        self.data_dir = data_dir
        self.image_format = image_format
        self.to_tensor = torchvision.transforms.ToTensor()
        self.mean = np.array([0.08069, 0.05258, 0.05487, 0.08282])
        self.std = np.array([0.13704, 0.10145, 0.15313, 0.13814])

        self.normalize = torchvision.transforms.Normalize(self.mean,self.std)
        random_dihedral_tf = torchvision.transforms.Compose( [ torchvision.transforms.RandomHorizontalFlip() , 
                                                                torchvision.transforms.RandomVerticalFlip() , 
                                                                torchvision.transforms.RandomRotation( 90 ) 
                                                                ]
                                                            )
        random_rotate = torchvision.transforms.RandomRotation( (0,45) )
        color_jitter = torchvision.transforms.ColorJitter( 0.05 , 0.05 )
        self.aug = torchvision.transforms.Compose( [ random_dihedral_tf , random_rotate , color_jitter ]  )
        self.to_pil = torchvision.transforms.ToPILImage()
        self.tta_hor_flip = torchvision.transforms.RandomHorizontalFlip(1.0)
        self.tta_ver_flip = torchvision.transforms.RandomVerticalFlip(1.0)

        
    def __len__( self ):
        return len( self.df )
    def __getitem__( self , idx ):
        img_channel_list = []
        data_dir = self.data_dir
        image_format = self.image_format
        if hasattr( self.df , 'Directory'):
            data_dir = self.df.Directory[idx]
        if hasattr( self.df , 'ImageFormat' ):
            image_format = self.df.ImageFormat[idx]
            
        for color in ['red','green','blue','yellow']:
            #print(self.data_dir + '/' + self.df.index[idx] +'_{}.png'.format( color )) 
            fname = data_dir + '/' + self.df.index[idx] +'_{}.{}'.format( color , image_format )
            img = cv2.imread( fname , cv2.IMREAD_GRAYSCALE  )
            if not (self.config.net['input_shape'][0] == 512 and self.config.net['input_shape'][1] == 512):
                #try:
                img = cv2.resize( img , self.config.net['input_shape'] ,  cv2.INTER_LANCZOS4 )
                #except Exception as e:
                #    print( fname )
                #    raise e
            try:
                img = img.reshape( img.shape[0] , img.shape[1] , 1  )
            except Exception as e:
                print( fname )
                raise e
            img_channel_list.append( img )

        img =  np.concatenate( img_channel_list , axis = -1 )
                
        img = self.to_pil( img )

        if self.is_training:
            img = self.aug( img )
            img = self.to_tensor( img )
            img = self.normalize( img )
        elif self.tta:
            tta_list = []
            for i in range(self.tta):
                t_img = deepcopy( img )
                t_img = self.aug( img )
                #if i&2 : t_img = self.tta_hor_flip( t_img )
                #if i//2 : t_img = self.tta_ver_flip( t_img ) 
                t_img = self.to_tensor( t_img )
                t_img = self.normalize( t_img )
                tta_list.append( t_img )
            tta_list += [ self.normalize(self.to_tensor(img)) ] * ceil( self.tta / 4 )
            img = torch.stack( tta_list )
        else:
            img = self.to_tensor( img )
            img = self.normalize( img )

        #assert img.shape[0]==4 and img.shape[1]==512 and img.shape[2]==512
                
            #img = ( self.to_tensor( img ) - 0.5 ) *2.0
        ret_dict = { 'img' : img  }
        if self.has_label:
            ret_dict['label'] = np.zeros(len(self.label_to_name_dict),np.float32)
            ret_dict['label'][ np.array( self.df['Target'][idx].split(' ') , np.uint8 ) ] = 1
            k = self.config.net['num_classes']
            eps = self.config.data['smooth_label_epsilon']
            ret_dict['label'] = (1 - eps) * ret_dict['label'] +  eps * ( 1 - ret_dict['label'] )  / k
        ret_dict['filename'] = self.df.index[idx]
        return ret_dict


def mil_collate_fn( batch  ):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if torch.utils.data.dataloader._use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: mil_collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        #transposed = zip(*batch)
        #return [default_collate(samples) for samples in transposed]
        #print('------------------sequence--------------------')
        return [ mil_collate_fn( samples ) for samples in batch ]

    raise TypeError((error_msg.format(type(batch[0]))))


class MILProteinDataset(data.Dataset):
    def __init__(self , config , df  , is_training , tta = 0,  data_dir = "" , image_format = 'png' , has_label = True ):
        
        self.config = config
        self.label_to_name_dict = {
            0:  'Nucleoplasm',
            1:  'Nuclear membrane',
            2:  'Nucleoli',   
            3:  'Nucleoli fibrillar center',
            4:  'Nuclear speckles',
            5:  'Nuclear bodies',
            6:  'Endoplasmic reticulum',   
            7:  'Golgi apparatus',
            8:  'Peroxisomes',
            9:  'Endosomes',
            10:  'Lysosomes',
            11:  'Intermediate filaments',
            12:  'Actin filaments',
            13:  'Focal adhesion sites',   
            14:  'Microtubules',
            15:  'Microtubule ends',  
            16:  'Cytokinetic bridge',   
            17:  'Mitotic spindle',
            18:  'Microtubule organizing center',  
            19:  'Centrosome',
            20:  'Lipid droplets',
            21:  'Plasma membrane',   
            22:  'Cell junctions', 
            23:  'Mitochondria',
            24:  'Aggresome',
            25:  'Cytosol',
            26:  'Cytoplasmic bodies',   
            27:  'Rods & rings'
        }
        self.has_label = has_label
        self.is_training = is_training
        self.tta = tta
        self.df = df
        self.data_dir = data_dir
        self.image_foramt = image_format
        self.to_tensor = torchvision.transforms.ToTensor()
        self.mean = np.array([0.08069, 0.05258, 0.05487, 0.08282])
        self.std = np.array([0.13704, 0.10145, 0.15313, 0.13814])

        self.normalize = torchvision.transforms.Normalize(self.mean,self.std)
        random_dihedral_tf = torchvision.transforms.Compose( [ torchvision.transforms.RandomHorizontalFlip() , 
                                                                torchvision.transforms.RandomVerticalFlip() , 
                                                                torchvision.transforms.RandomRotation( 90 ) 
                                                                ]
                                                            )
        random_rotate = torchvision.transforms.RandomRotation( (0,45) )
        color_jitter = torchvision.transforms.ColorJitter( 0.05 , 0.05 )
        self.aug = torchvision.transforms.Compose( [ random_dihedral_tf , random_rotate , color_jitter ]  )
        self.to_pil = torchvision.transforms.ToPILImage()
        self.tta_hor_flip = torchvision.transforms.RandomHorizontalFlip(1.0)
        self.tta_ver_flip = torchvision.transforms.RandomVerticalFlip(1.0)

    def __len__( self ):
        return len( self.df )
    def __getitem__( self , idx ):
        temp = os.listdir( self.data_dir + '/' + self.df.index[idx] )
        #print(temp)
        num_imgs = len( temp ) //4
        img_list = []
        #print( num_imgs )
        for i in range( num_imgs ):
            img_channel_list = []
            for color in ['red','green','blue','yellow']:
                fname = self.data_dir + '/' + self.df.index[idx] + '/' + str(i) +'_{}.{}'.format( color , self.image_format )
                img = cv2.imread( fname , cv2.IMREAD_GRAYSCALE  )
                #print( img )
                if not (self.config.net['input_shape'][0] == 128 and self.config.net['input_shape'][1] == 128):
                    img = cv2.resize( img , self.config.net['input_shape'] ,  cv2.INTER_LANCZOS4 )
                img = img.reshape( img.shape[0] , img.shape[1] , 1  )
                img_channel_list.append( img )

            img =  np.concatenate( img_channel_list , axis = -1 )
                    
            img = self.to_pil( img )

            if self.is_training:
                img = self.aug( img )
                img = self.to_tensor( img )
                img = self.normalize( img )
            elif self.tta:
                tta_list = []
                for j in range(self.tta):
                    t_img = deepcopy( img )
                    t_img = self.aug( img )
                    #if i&2 : t_img = self.tta_hor_flip( t_img )
                    #if i//2 : t_img = self.tta_ver_flip( t_img ) 
                    t_img = self.to_tensor( t_img )
                    t_img = self.normalize( t_img )
                    tta_list.append( t_img )
                tta_list += [ self.normalize(self.to_tensor(img)) ] * ceil( self.tta / 4 )
                img = torch.stack( tta_list )
            else:
                img = self.to_tensor( img )
                img = self.normalize( img )
            img_list.append( img )
        imgs =  img_list 
        #for img in imgs:
        #    print(img.shape)


        #assert img.shape[0]==4 and img.shape[1]==512 and img.shape[2]==512
                
            #img = ( self.to_tensor( img ) - 0.5 ) *2.0
        ret_dict = { 'img' : imgs  }
        if self.has_label:
            ret_dict['label'] = np.zeros(len(self.label_to_name_dict),np.float32)
            ret_dict['label'][ np.array( self.df['Target'][idx].split(' ') , np.uint8 ) ] = 1
            k = self.config.net['num_classes']
            eps = self.config.data['smooth_label_epsilon']
            ret_dict['label'] = (1 - eps) * ret_dict['label'] +  eps * ( 1 - ret_dict['label'] )  / k
        ret_dict['filename'] = self.df.index[idx]
        return ret_dict



