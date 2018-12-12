'''
Given folders of the full images from the human protein atlas, extract single cell crops.

To obtain single cell crops, an otsu filter is used to segment the nuclei channel, and then large enough connected
components in the nuclei channel are used as the centers for these crops.

Author: Alex Lu
Email: alexlu@cs.toronto.edu
Copyright (C) 2018 Alex Lu
'''

from PIL import Image
import numpy as np
import os
import skimage
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from skimage.measure import label
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy.ndimage.measurements import center_of_mass
import pandas as pd
import cv2
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

'''
Given an image, extract single cell crops:

ARGUMENTS:
imagepath: full path of the image
foldername: name of the folder image is in
imagename: name of the jpeg file
savepath: directory to save single cell crops to (will save in a subdirectory named after the folder)
outfile: file to write coordinates of cell centers to (for debugging purposes)
scale: scale to downsize original images by
cropsize: size of square crops (in pixels on the rescaled image) to extract
'''

def find_centers_and_crop (imagepath,  imagename, savepath, outfile, scale=4, cropsize=128):
    # Get the image and resize
    #print( '{}/{}_{}.jpg'.format(imagepath , imagename, 'red' ) )
    imgs = [ cv2.imread('{}/{}_{}.jpg'.format(imagepath , imagename, color)  , cv2.IMREAD_GRAYSCALE)   for color in ['red','green','blue','yellow']  ]
    color_image = np.stack( imgs , -1 )
    #print( color_image.shape )
    #print ("Working on", imagename)
    image_shape = color_image.shape[:2]
    image_shape = tuple(ti//scale for ti in image_shape)
    color_image = resize(color_image, image_shape)
    

    # Split the image into channels
    microtubules = color_image[:, :, 0]
    antibody = color_image[:, :, 1]
    nuclei = color_image[:, :, 2]
    yellow_img = color_image[:,:3]

    # Segment the nuclear channel and get the nuclei
    min_nuc_size = 100.0

    val = threshold_otsu(nuclei)
    smoothed_nuclei = gaussian(nuclei, sigma=5.0)
    binary_nuclei = smoothed_nuclei > val
    binary_nuclei = remove_small_holes(binary_nuclei, min_size=300)
    labeled_nuclei = label(binary_nuclei)
    #labeled_nuclei = clear_border(labeled_nuclei)
    labeled_nuclei = remove_small_objects(labeled_nuclei, min_size=min_nuc_size)

    # Iterate through each nuclei and get their centers (if the object is valid), and save to directory
    for i in range(1, np.max(labeled_nuclei) + 1 ):
        current_nuc = labeled_nuclei == i
        if np.sum(current_nuc) > min_nuc_size:
            y, x = center_of_mass(current_nuc)
            x = np.int(x)
            y = np.int(y)

            c1 = y - cropsize // 2
            c2 = y + cropsize // 2
            c3 = x - cropsize // 2
            c4 = x + cropsize // 2

            cc1 = max( c1 , 0 )
            cc2 = min( c2 , image_shape[0] - 1 )
            cc3 = max( c3 , 0 )
            cc4 = min( c4 , image_shape[1] - 1 )

            d1 = cc1 - c1
            d2 = c2 - cc2
            d3 = cc3 - c3
            d4 = c4 - cc4
            #else:

            img_crop = color_image[cc1:cc2,cc3:cc4]
            if c1 < 0 or c3 < 0 or c2 > image_shape[0] or c4 > image_shape[1]:
                img_crop = np.pad(  img_crop , ((d1,d2) , (d3,d4) , (0,0) ), mode = 'constant' , constant_values = 0 )

            #folder_suffix = imagename.rsplit("_", 4)[0]
            outfolder = savepath + imagename #+ foldername + "_" + folder_suffix
            #outimagename = imagename.rsplit("_", 3)[0] + "_" + str(i)
            outimagename = str(i)

            if not os.path.exists(outfolder):
                os.mkdir(outfolder)

            for color_idx, color in enumerate(['red','green','blue','yellow']):
                Image.fromarray( img_crop[:,:,color_idx] ).save(outfolder + "/" + outimagename + "_{}.tif".format(color) )

            output = open(outfile, "a")
            #output.write(foldername + "_" + folder_suffix + "/" + outimagename)
            output.write(imagename+'/'+outimagename)
            output.write("\t")
            output.write(str(x))
            output.write("\t")
            output.write(str(y))
            output.write("\n")
            output.close()

if __name__ == "__main__":
    '''Loop to call the cell crop segmentation on all folders in a directory. If you used the download_hpa.py file
    to obtain the HPA images, they will already be in the format required by this script.'''

    csvpath = '../../data/external/HPAv18RBGY_wodpl.csv'
    filepath = '../../data/external/HPAv18_images'
    outpath = "../../data/external/HPAv18_images_single_cell_crop/"                 # Path to save the crops to
    outfile = "../../data/external/single_cell_crop_centers.txt"      # Path to save file of cell center coordinates to

    # Creates output folder if necessary
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    else:
        os.system( 'rm -rf {}'.format( outpath ) )
        os.mkdir(outpath)

    if os.path.exists( outfile ):
        os.remove( outfile )

    # Loop over all folders in the input folder and extract single cell crops for all images
    # Writes single cell crops to sub-directories in the outpath folder,
    # named identical to the sub-directories in the input folder

    df = pd.read_csv( csvpath  , index_col = 0 )
    for Id ,v  in tqdm( df.iterrows() , total=len(df)):
        imagepath =  Id
        find_centers_and_crop(filepath, Id, outpath, outfile)

