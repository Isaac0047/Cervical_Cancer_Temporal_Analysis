### In the previous code 'cervical_data_search.py', the 3D model of MR, PET and ADC are generated
### This code tries to extract the files and conduct statistical analysis

import nibabel as nib
import os
import scipy
import glob
import numpy as np

from matplotlib import pyplot as plt
from dltk.io.augmentation import *
from dltk.io.preprocessing import *

import pydicom as di
from rt_utils import RTStructBuilder

#%% Load the existing files

with open('adc_dir_A1.txt', 'r') as file:
    adc_dir_A1 = [line.strip() for line in file.readlines()]
with open('adc_dir_A2.txt', 'r') as file:
    adc_dir_A2 = [line.strip() for line in file.readlines()]
with open('adc_dir_A3.txt', 'r') as file:
    adc_dir_A3 = [line.strip() for line in file.readlines()]
with open('adc_dir_B1.txt', 'r') as file:
    adc_dir_B1 = [line.strip() for line in file.readlines()]
with open('adc_dir_B2.txt', 'r') as file:
    adc_dir_B2 = [line.strip() for line in file.readlines()]
with open('adc_dir_B3.txt', 'r') as file:
    adc_dir_B3 = [line.strip() for line in file.readlines()]
    
with open('dce_dir_A1.txt', 'r') as file:
    dce_dir_A1 = [line.strip() for line in file.readlines()]
with open('dce_dir_A2.txt', 'r') as file:
    dce_dir_A2 = [line.strip() for line in file.readlines()]
with open('dce_dir_A3.txt', 'r') as file:
    dce_dir_A3 = [line.strip() for line in file.readlines()]
with open('dce_dir_B1.txt', 'r') as file:
    dce_dir_B1 = [line.strip() for line in file.readlines()]
with open('dce_dir_B2.txt', 'r') as file:
    dce_dir_B2 = [line.strip() for line in file.readlines()]
with open('dce_dir_B3.txt', 'r') as file:
    dce_dir_B3 = [line.strip() for line in file.readlines()]
    
with open('pet_dir_A1.txt', 'r') as file:
    pet_dir_A1 = [line.strip() for line in file.readlines()]
with open('pet_dir_A2.txt', 'r') as file:
    pet_dir_A2 = [line.strip() for line in file.readlines()]
with open('pet_dir_A3.txt', 'r') as file:
    pet_dir_A3 = [line.strip() for line in file.readlines()]
with open('pet_dir_B1.txt', 'r') as file:
    pet_dir_B1 = [line.strip() for line in file.readlines()]
with open('pet_dir_B2.txt', 'r') as file:
    pet_dir_B2 = [line.strip() for line in file.readlines()]
with open('pet_dir_B3.txt', 'r') as file:
    pet_dir_B3 = [line.strip() for line in file.readlines()]
    
#%% Load the constructed 3D models

def resize(img, shape, mode='nearest', orig_shape=None, order=3):
    """
    Wrapper for scipy.ndimage.zoom suited for MRI images.
    """

    if orig_shape == None: orig_shape = img.shape

    assert len(shape) == 3, "Can not have more than 3 dimensions"
    factors = (
        shape[0]/orig_shape[0],
        shape[1]/orig_shape[1], 
        shape[2]/orig_shape[2]
    )

    # Resize to the given shape
    return scipy.ndimage.zoom(img, factors, mode=mode, order=order)

shape = [256,256,32]

from scipy.ndimage import binary_dilation
from scipy.ndimage import zoom

def resize_binary_3d(binary_data, shape):
    # Compute scaling factors along each axis
    scale_factors = [shape[i] / binary_data.shape[i] for i in range(3)]
    
    # Perform binary dilation
    resized_data = binary_dilation(binary_data, structure=np.ones((3, 3, 3)))
    resized_data = zoom(resized_data, zoom=scale_factors, order=0)

    return resized_data

shape     = [256,256,32]

#%% Load the 3D images

adc_3d = []

for fp in adc_dir:
    os.chdir(fp)
    foldname = glob.glob('*.dcm')
    for fn in os.listdir(foldname):
        ds = di.filereader.dcmread(fn)
        im = di.dcmread(fn)
        plt.imshow(im.pixel_array, cmap=plt.cm.bone)

dce_3d = []
    
pet_3d = []

#%% Another test

adc_3d = []

for fp in adc_dir:
    os.chdir(fp)
    dciom_files = []
    
    for root, dirs, files in os.walk(fp):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os)
    foldname = glob.glob('*.dcm')
    for fn in os.listdir(foldname):
        ds = di.filereader.dcmread(fn)
        im = di.dcmread(fn)
        plt.imshow(im.pixel_array, cmap=plt.cm.bone)

#%% Next, Load the region ROI to the file

os.chdir('/Users/hafeng/.spyder-py3/cervical_cancer_analysis/')

p1_c_data = np.load('mask_3ds_p1_c_resize.npy')
p2_c_data = np.load('mask_3ds_p2_c_resize.npy')
p3_c_data = np.load('mask_3ds_p3_c_resize.npy')

p1_c_data = p1_c_data.astype('double')
p2_c_data = p2_c_data.astype('double')
p3_c_data = p3_c_data.astype('double')

p1_p1_data = np.load('mask_3ds_p1_p1_resize.npy')
p2_p1_data = np.load('mask_3ds_p2_p1_resize.npy')
p3_p1_data = np.load('mask_3ds_p3_p1_resize.npy')

p1_p1_data = p1_p1_data.astype('double')
p2_p1_data = p2_p1_data.astype('double')
p3_p1_data = p3_p1_data.astype('double')

p1_p2_data = np.load('mask_3ds_p1_p2_resize.npy')
p2_p2_data = np.load('mask_3ds_p2_p2_resize.npy')
p3_p2_data = np.load('mask_3ds_p3_p2_resize.npy')

p1_p2_data = p1_p2_data.astype('double')
p2_p2_data = p2_p2_data.astype('double')
p3_p2_data = p3_p2_data.astype('double')

#%% Next, apply the mask to different images and see if they match

# First step is to convert the 3d image list to 3d matrices


