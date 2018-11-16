import torch.utils.data as data
from cv2 import imread , IMREAD_GRAYSCALE
import torch


import numpy as np
import torchvision
import torchvision.transforms
from custom_utils import *
from sklearn.decomposition import PCA
import pandas as pd

class ProteinDataset(data.Dataset):
    def __init__(self , config , df  , is_training , data_dir = "" , has_label = True ):
        
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
        self.df = df
        self.data_dir = data_dir
        self.to_tensor = torchvision.transforms.ToTensor()
        self.mean = np.array([0.08069, 0.05258, 0.05487, 0.08282])
        self.std = np.array([0.13704, 0.10145, 0.15313, 0.13814])

        self.normalize = torchvision.transforms.Normalize(self.mean,self.std)
        self.random_dihedral_tf = torchvision.transforms.Compose( [ torchvision.transforms.RandomHorizontalFlip() , 
                                                                torchvision.transforms.RandomVerticalFlip() , 
                                                                torchvision.transforms.RandomRotation( 90 ) 
                                                                ]
                                                            )
        self.random_rotate = torchvision.transforms.RandomRotation( (-45,45))
        self.color_jitter = torchvision.transforms.ColorJitter( 0.05 , 0.05 )
        self.to_pil = torchvision.transforms.ToPILImage()

    def __len__( self ):
        return len( self.df )
    def __getitem__( self , idx ):
        img_channel_list = []
        for color in ['red','green','blue','yellow']:
            #print(self.data_dir + '/' + self.df.index[idx] +'_{}.png'.format( color )) 
            img = imread( self.data_dir + '/' + self.df.index[idx] +'_{}.png'.format( color ) , IMREAD_GRAYSCALE  )  
            img = img.reshape( img.shape[0] , img.shape[1] , 1  )
            img = self.to_pil( img )

            if self.is_training:
                img = self.random_dihedral_tf(img)
                img = self.random_rotate( img )
                img = self.color_jitter( img )
            img = self.to_tensor( img )
            img_channel_list.append( img )

        img = torch.cat( img_channel_list  , dim = 0  )
        #assert img.shape[0]==4 and img.shape[1]==512 and img.shape[2]==512
        img = self.normalize( img )
                
            #img = ( self.to_tensor( img ) - 0.5 ) *2.0
        ret_dict = { 'img' : img  }
        if self.has_label:
            ret_dict['label'] = np.zeros(len(self.label_to_name_dict),np.float32)
            ret_dict['label'][ np.array( self.df['Target'][idx].split(' ') , np.uint8 ) ] = 1
        ret_dict['filename'] = self.df.index[idx]
        return ret_dict



