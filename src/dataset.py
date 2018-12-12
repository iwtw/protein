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

class ProteinDataset(data.Dataset):
    def __init__(self , config , df  , is_training , tta = 0,  data_dir = "" , has_label = True ):
        
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
        for color in ['red','green','blue','yellow']:
            #print(self.data_dir + '/' + self.df.index[idx] +'_{}.png'.format( color )) 
            img = cv2.imread( self.data_dir + '/' + self.df.index[idx] +'_{}.png'.format( color ) , cv2.IMREAD_GRAYSCALE  )
            if not (self.config.net['input_shape'][0] == 512 and self.config.net['input_shape'][1] == 512):
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



