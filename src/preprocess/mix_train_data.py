import pandas as pd
import numpy as np

def mix(mix_class,out_csv_fname):
    original_train_df = pd.read_csv( '../../data/train.csv' , index_col = 0 )
    original_train_df['Directory'] = '/home/wtw/kaggle/human_protein/data/train'
    original_train_df['ImageFormat'] = 'png'
    external_train_df = pd.read_csv( '../../data/external/HPAv18RBGY_wodpl.csv' , index_col = 0 )
    external_train_df['Directory'] = '/home/wtw/kaggle/human_protein/data/external/HPAv18_images_512x512'
    external_train_df['ImageFormat'] = 'jpg'

    external_train_df.Target = external_train_df.Target.apply( lambda x : np.array( x.split(' ') , np.uint8 ) )
    print( external_train_df.Target.tail() )

    #t = external_train_df.loc[ external_train_df == np.array( [0] ) ]
    print(  external_train_df.Target.apply( lambda x : any( map( lambda y : y in mix_class , x  )) ).tail()  )  
    filter_external_train_df = external_train_df.loc[ external_train_df.Target.apply( lambda x : any( map( lambda y : y in mix_class , x  )) ) ] # lower than 1000 samples numbers
    filter_external_train_df.Target = filter_external_train_df.Target.apply( lambda x :  ' '.join( list(map( str , x )) )  )
    print( filter_external_train_df.Target.head() )

    mix_df = original_train_df.append( filter_external_train_df )
    mix_df.to_csv( out_csv_fname )

lt_1000 = [8,9,10,12,13,15,16,17,18,20,22,24,26,27]
mix(  lt_1000 ,  '../../data/train_mix1.csv')#samples less than 1000
mix(  [5,6,9,10,13,15,16,17,18,19,20,21,22,24,25,26,27] + lt_1000 ,  '../../data/train_mix2.csv')#recall lower than 60%
mix(  [6,10,13,15,16,17,18,19,20,21,22,27] + lt_1000 ,  '../../data/train_mix3.csv')#f1 lower than 60%

