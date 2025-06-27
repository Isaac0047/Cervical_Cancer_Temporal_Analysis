#%% This code finalize the 2D statistical analysis of cervical data set

#%% Load important modules

import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd

from dltk.io.augmentation import *
from dltk.io.preprocessing import *

import pydicom as di
from rt_utils import RTStructBuilder

import skimage
from skimage.feature import graycoprops
import pickle

#%% Load the zero order features

# Feature follows (PET: t1_max, t1_mean, t1_size, t2_max, t2_mean, t2_size, t3_max, t3_mean, t3_size)
zero_feature_1   = np.load('zero_order_feature_1.npy')
zero_feature_12  = np.load('zero_order_feature_12.npy')
zero_feature_123 = np.load('zero_order_feature_123.npy')

# Feature follows (t1: mean, quantile, coeff_var, skewness, kurtosis, max_value. Then t2, t3 following)
first_feature_adc = np.load('first_order_feature_adc_all.npy') 
first_feature_dce = np.load('first_order_feature_dce_all.npy') 
first_feature_pet = np.load('first_order_feature_pet_all.npy') 

first_feature_1   = np.concatenate((first_feature_adc[:,:6],first_feature_dce[:,:6],first_feature_pet[:,:6]),axis=1)
first_feature_12  = np.concatenate((first_feature_adc[:,:12],first_feature_dce[:,:12],first_feature_pet[:,:12]),axis=1)
first_feature_123 =np.concatenate((first_feature_adc,first_feature_dce,first_feature_pet),axis=1)

#%% Load the data for analysis

# ### ADC FEATURES
# with open('adc_A1_contrast_3d.pkl', 'rb') as file:
#     adc_A1_contrast_3d = pickle.load(file)
# with open('adc_A2_contrast_3d.pkl', 'rb') as file:
#     adc_A2_contrast_3d = pickle.load(file)
# with open('adc_A3_contrast_3d.pkl', 'rb') as file:
#     adc_A3_contrast_3d = pickle.load(file)
# with open('adc_B1_contrast_3d.pkl', 'rb') as file:
#     adc_B1_contrast_3d = pickle.load(file)
# with open('adc_B2_contrast_3d.pkl', 'rb') as file:
#     adc_B2_contrast_3d = pickle.load(file)
# with open('adc_B3_contrast_3d.pkl', 'rb') as file:
#     adc_B3_contrast_3d = pickle.load(file)
    
# with open('adc_A1_energy_3d.pkl', 'rb') as file:
#     adc_A1_energy_3d = pickle.load(file)
# with open('adc_A2_energy_3d.pkl', 'rb') as file:
#     adc_A2_energy_3d = pickle.load(file)
# with open('adc_A3_energy_3d.pkl', 'rb') as file:
#     adc_A3_energy_3d = pickle.load(file)
# with open('adc_A1_energy_3d.pkl', 'rb') as file:
#     adc_B1_energy_3d = pickle.load(file)
# with open('adc_A2_energy_3d.pkl', 'rb') as file:
#     adc_B2_energy_3d = pickle.load(file)
# with open('adc_A3_energy_3d.pkl', 'rb') as file:
#     adc_B3_energy_3d = pickle.load(file)
    
# with open('adc_A1_correlation_3d.pkl', 'rb') as file:
#     adc_A1_correlation_3d = pickle.load(file)
# with open('adc_A2_correlation_3d.pkl', 'rb') as file:
#     adc_A2_correlation_3d = pickle.load(file)
# with open('adc_A3_correlation_3d.pkl', 'rb') as file:
#     adc_A3_correlation_3d = pickle.load(file)
# with open('adc_B1_correlation_3d.pkl', 'rb') as file:
#     adc_B1_correlation_3d = pickle.load(file)
# with open('adc_B2_correlation_3d.pkl', 'rb') as file:
#     adc_B2_correlation_3d = pickle.load(file)
# with open('adc_B3_correlation_3d.pkl', 'rb') as file:
#     adc_B3_correlation_3d = pickle.load(file)
    
# with open('adc_A1_homogeneity_3d.pkl', 'rb') as file:
#     adc_A1_homogeneity_3d = pickle.load(file)
# with open('adc_A2_homogeneity_3d.pkl', 'rb') as file:
#     adc_A2_homogeneity_3d = pickle.load(file)
# with open('adc_A3_homogeneity_3d.pkl', 'rb') as file:
#     adc_A3_homogeneity_3d = pickle.load(file)
# with open('adc_B1_homogeneity_3d.pkl', 'rb') as file:
#     adc_B1_homogeneity_3d = pickle.load(file)
# with open('adc_B2_homogeneity_3d.pkl', 'rb') as file:
#     adc_B2_homogeneity_3d = pickle.load(file)
# with open('adc_B3_homogeneity_3d.pkl', 'rb') as file:
#     adc_B3_homogeneity_3d = pickle.load(file)
    
    
# ### DCE FEATURES
# with open('dce_A1_contrast_3d.pkl', 'rb') as file:
#     dce_A1_contrast_3d = pickle.load(file)
# with open('dce_A2_contrast_3d.pkl', 'rb') as file:
#     dce_A2_contrast_3d = pickle.load(file)
# with open('dce_A3_contrast_3d.pkl', 'rb') as file:
#     dce_A3_contrast_3d = pickle.load(file)
# with open('dce_B1_contrast_3d.pkl', 'rb') as file:
#     dce_B1_contrast_3d = pickle.load(file)
# with open('dce_B2_contrast_3d.pkl', 'rb') as file:
#     dce_B2_contrast_3d = pickle.load(file)
# with open('dce_B3_contrast_3d.pkl', 'rb') as file:
#     dce_B3_contrast_3d = pickle.load(file)
    
# with open('dce_A1_energy_3d.pkl', 'rb') as file:
#     dce_A1_energy_3d = pickle.load(file)
# with open('dce_A2_energy_3d.pkl', 'rb') as file:
#     dce_A2_energy_3d = pickle.load(file)
# with open('dce_A3_energy_3d.pkl', 'rb') as file:
#     dce_A3_energy_3d = pickle.load(file)
# with open('dce_B1_energy_3d.pkl', 'rb') as file:
#     dce_B1_energy_3d = pickle.load(file)
# with open('dce_B2_energy_3d.pkl', 'rb') as file:
#     dce_B2_energy_3d = pickle.load(file)
# with open('dce_B3_energy_3d.pkl', 'rb') as file:
#     dce_B3_energy_3d = pickle.load(file)
    
# with open('dce_A1_correlation_3d.pkl', 'rb') as file:
#     dce_A1_correlation_3d = pickle.load(file)
# with open('dce_A2_correlation_3d.pkl', 'rb') as file:
#     dce_A2_correlation_3d = pickle.load(file)
# with open('dce_A3_correlation_3d.pkl', 'rb') as file:
#     dce_A3_correlation_3d = pickle.load(file)
# with open('dce_B1_correlation_3d.pkl', 'rb') as file:
#     dce_B1_correlation_3d = pickle.load(file)
# with open('dce_B2_correlation_3d.pkl', 'rb') as file:
#     dce_B2_correlation_3d = pickle.load(file)
# with open('dce_B3_correlation_3d.pkl', 'rb') as file:
#     dce_B3_correlation_3d = pickle.load(file)
    
# with open('dce_A1_homogeneity_3d.pkl', 'rb') as file:
#     dce_A1_homogeneity_3d = pickle.load(file)
# with open('dce_A2_homogeneity_3d.pkl', 'rb') as file:
#     dce_A2_homogeneity_3d = pickle.load(file)
# with open('dce_A3_homogeneity_3d.pkl', 'rb') as file:
#     dce_A3_homogeneity_3d = pickle.load(file)
# with open('dce_B1_homogeneity_3d.pkl', 'rb') as file:
#     dce_B1_homogeneity_3d = pickle.load(file)
# with open('dce_B2_homogeneity_3d.pkl', 'rb') as file:
#     dce_B2_homogeneity_3d = pickle.load(file)
# with open('dce_B3_homogeneity_3d.pkl', 'rb') as file:
#     dce_B3_homogeneity_3d = pickle.load(file)
    
    
# ### PET FEATURES
# with open('pet_A1_contrast_3d.pkl', 'rb') as file:
#     pet_A1_contrast_3d = pickle.load(file)
# with open('pet_A2_contrast_3d.pkl', 'rb') as file:
#     pet_A2_contrast_3d = pickle.load(file)
# with open('pet_A3_contrast_3d.pkl', 'rb') as file:
#     pet_A3_contrast_3d = pickle.load(file)
# with open('pet_B1_contrast_3d.pkl', 'rb') as file:
#     pet_B1_contrast_3d = pickle.load(file)
# with open('pet_B2_contrast_3d.pkl', 'rb') as file:
#     pet_B2_contrast_3d = pickle.load(file)
# with open('pet_B3_contrast_3d.pkl', 'rb') as file:
#     pet_B3_contrast_3d = pickle.load(file)
    
# with open('pet_A1_energy_3d.pkl', 'rb') as file:
#     pet_A1_energy_3d = pickle.load(file)
# with open('pet_A2_energy_3d.pkl', 'rb') as file:
#     pet_A2_energy_3d = pickle.load(file)
# with open('pet_A3_energy_3d.pkl', 'rb') as file:
#     pet_A3_energy_3d = pickle.load(file)
# with open('pet_B1_energy_3d.pkl', 'rb') as file:
#     pet_B1_energy_3d = pickle.load(file)
# with open('pet_B2_energy_3d.pkl', 'rb') as file:
#     pet_B2_energy_3d = pickle.load(file)
# with open('pet_B3_energy_3d.pkl', 'rb') as file:
#     pet_B3_energy_3d = pickle.load(file)
    
# with open('pet_A1_correlation_3d.pkl', 'rb') as file:
#     pet_A1_correlation_3d = pickle.load(file)
# with open('pet_A2_correlation_3d.pkl', 'rb') as file:
#     pet_A2_correlation_3d = pickle.load(file)
# with open('pet_A3_correlation_3d.pkl', 'rb') as file:
#     pet_A3_correlation_3d = pickle.load(file)
# with open('pet_B1_correlation_3d.pkl', 'rb') as file:
#     pet_B1_correlation_3d = pickle.load(file)
# with open('pet_B2_correlation_3d.pkl', 'rb') as file:
#     pet_B2_correlation_3d = pickle.load(file)
# with open('pet_B3_correlation_3d.pkl', 'rb') as file:
#     pet_B3_correlation_3d = pickle.load(file)
    
# with open('pet_A1_homogeneity_3d.pkl', 'rb') as file:
#     pet_A1_homogeneity_3d = pickle.load(file)
# with open('pet_A2_homogeneity_3d.pkl', 'rb') as file:
#     pet_A2_homogeneity_3d = pickle.load(file)
# with open('pet_A3_homogeneity_3d.pkl', 'rb') as file:
#     pet_A3_homogeneity_3d = pickle.load(file)
# with open('pet_B1_homogeneity_3d.pkl', 'rb') as file:
#     pet_B1_homogeneity_3d = pickle.load(file)
# with open('pet_B2_homogeneity_3d.pkl', 'rb') as file:
#     pet_B2_homogeneity_3d = pickle.load(file)
# with open('pet_B3_homogeneity_3d.pkl', 'rb') as file:
#     pet_B3_homogeneity_3d = pickle.load(file)
    
### ADC FEATURES
with open('adc_A1_contrast_3d_origin.pkl', 'rb') as file:
    adc_A1_contrast_3d = pickle.load(file)
with open('adc_A2_contrast_3d_origin.pkl', 'rb') as file:
    adc_A2_contrast_3d = pickle.load(file)
with open('adc_A3_contrast_3d_origin.pkl', 'rb') as file:
    adc_A3_contrast_3d = pickle.load(file)
with open('adc_B1_contrast_3d_origin.pkl', 'rb') as file:
    adc_B1_contrast_3d = pickle.load(file)
with open('adc_B2_contrast_3d_origin.pkl', 'rb') as file:
    adc_B2_contrast_3d = pickle.load(file)
with open('adc_B3_contrast_3d_origin.pkl', 'rb') as file:
    adc_B3_contrast_3d = pickle.load(file)
    
with open('adc_A1_energy_3d_origin.pkl', 'rb') as file:
    adc_A1_energy_3d = pickle.load(file)
with open('adc_A2_energy_3d_origin.pkl', 'rb') as file:
    adc_A2_energy_3d = pickle.load(file)
with open('adc_A3_energy_3d_origin.pkl', 'rb') as file:
    adc_A3_energy_3d = pickle.load(file)
with open('adc_A1_energy_3d_origin.pkl', 'rb') as file:
    adc_B1_energy_3d = pickle.load(file)
with open('adc_A2_energy_3d_origin.pkl', 'rb') as file:
    adc_B2_energy_3d = pickle.load(file)
with open('adc_A3_energy_3d_origin.pkl', 'rb') as file:
    adc_B3_energy_3d = pickle.load(file)
    
with open('adc_A1_correlation_3d_origin.pkl', 'rb') as file:
    adc_A1_correlation_3d = pickle.load(file)
with open('adc_A2_correlation_3d_origin.pkl', 'rb') as file:
    adc_A2_correlation_3d = pickle.load(file)
with open('adc_A3_correlation_3d_origin.pkl', 'rb') as file:
    adc_A3_correlation_3d = pickle.load(file)
with open('adc_B1_correlation_3d_origin.pkl', 'rb') as file:
    adc_B1_correlation_3d = pickle.load(file)
with open('adc_B2_correlation_3d_origin.pkl', 'rb') as file:
    adc_B2_correlation_3d = pickle.load(file)
with open('adc_B3_correlation_3d_origin.pkl', 'rb') as file:
    adc_B3_correlation_3d = pickle.load(file)
    
with open('adc_A1_homogeneity_3d_origin.pkl', 'rb') as file:
    adc_A1_homogeneity_3d = pickle.load(file)
with open('adc_A2_homogeneity_3d_origin.pkl', 'rb') as file:
    adc_A2_homogeneity_3d = pickle.load(file)
with open('adc_A3_homogeneity_3d_origin.pkl', 'rb') as file:
    adc_A3_homogeneity_3d = pickle.load(file)
with open('adc_B1_homogeneity_3d_origin.pkl', 'rb') as file:
    adc_B1_homogeneity_3d = pickle.load(file)
with open('adc_B2_homogeneity_3d_origin.pkl', 'rb') as file:
    adc_B2_homogeneity_3d = pickle.load(file)
with open('adc_B3_homogeneity_3d_origin.pkl', 'rb') as file:
    adc_B3_homogeneity_3d = pickle.load(file)
    
    
### DCE FEATURES
with open('dce_A1_contrast_3d_origin.pkl', 'rb') as file:
    dce_A1_contrast_3d = pickle.load(file)
with open('dce_A2_contrast_3d_origin.pkl', 'rb') as file:
    dce_A2_contrast_3d = pickle.load(file)
with open('dce_A3_contrast_3d_origin.pkl', 'rb') as file:
    dce_A3_contrast_3d = pickle.load(file)
with open('dce_B1_contrast_3d_origin.pkl', 'rb') as file:
    dce_B1_contrast_3d = pickle.load(file)
with open('dce_B2_contrast_3d_origin.pkl', 'rb') as file:
    dce_B2_contrast_3d = pickle.load(file)
with open('dce_B3_contrast_3d_origin.pkl', 'rb') as file:
    dce_B3_contrast_3d = pickle.load(file)
    
with open('dce_A1_energy_3d_origin.pkl', 'rb') as file:
    dce_A1_energy_3d = pickle.load(file)
with open('dce_A2_energy_3d_origin.pkl', 'rb') as file:
    dce_A2_energy_3d = pickle.load(file)
with open('dce_A3_energy_3d_origin.pkl', 'rb') as file:
    dce_A3_energy_3d = pickle.load(file)
with open('dce_B1_energy_3d_origin.pkl', 'rb') as file:
    dce_B1_energy_3d = pickle.load(file)
with open('dce_B2_energy_3d_origin.pkl', 'rb') as file:
    dce_B2_energy_3d = pickle.load(file)
with open('dce_B3_energy_3d_origin.pkl', 'rb') as file:
    dce_B3_energy_3d = pickle.load(file)
    
with open('dce_A1_correlation_3d_origin.pkl', 'rb') as file:
    dce_A1_correlation_3d = pickle.load(file)
with open('dce_A2_correlation_3d_origin.pkl', 'rb') as file:
    dce_A2_correlation_3d = pickle.load(file)
with open('dce_A3_correlation_3d_origin.pkl', 'rb') as file:
    dce_A3_correlation_3d = pickle.load(file)
with open('dce_B1_correlation_3d_origin.pkl', 'rb') as file:
    dce_B1_correlation_3d = pickle.load(file)
with open('dce_B2_correlation_3d_origin.pkl', 'rb') as file:
    dce_B2_correlation_3d = pickle.load(file)
with open('dce_B3_correlation_3d_origin.pkl', 'rb') as file:
    dce_B3_correlation_3d = pickle.load(file)
    
with open('dce_A1_homogeneity_3d_origin.pkl', 'rb') as file:
    dce_A1_homogeneity_3d = pickle.load(file)
with open('dce_A2_homogeneity_3d_origin.pkl', 'rb') as file:
    dce_A2_homogeneity_3d = pickle.load(file)
with open('dce_A3_homogeneity_3d_origin.pkl', 'rb') as file:
    dce_A3_homogeneity_3d = pickle.load(file)
with open('dce_B1_homogeneity_3d_origin.pkl', 'rb') as file:
    dce_B1_homogeneity_3d = pickle.load(file)
with open('dce_B2_homogeneity_3d_origin.pkl', 'rb') as file:
    dce_B2_homogeneity_3d = pickle.load(file)
with open('dce_B3_homogeneity_3d_origin.pkl', 'rb') as file:
    dce_B3_homogeneity_3d = pickle.load(file)
    
    
### PET FEATURES
with open('pet_A1_contrast_3d_origin.pkl', 'rb') as file:
    pet_A1_contrast_3d = pickle.load(file)
with open('pet_A2_contrast_3d_origin.pkl', 'rb') as file:
    pet_A2_contrast_3d = pickle.load(file)
with open('pet_A3_contrast_3d_origin.pkl', 'rb') as file:
    pet_A3_contrast_3d = pickle.load(file)
with open('pet_B1_contrast_3d_origin.pkl', 'rb') as file:
    pet_B1_contrast_3d = pickle.load(file)
with open('pet_B2_contrast_3d_origin.pkl', 'rb') as file:
    pet_B2_contrast_3d = pickle.load(file)
with open('pet_B3_contrast_3d_origin.pkl', 'rb') as file:
    pet_B3_contrast_3d = pickle.load(file)
    
with open('pet_A1_energy_3d_origin.pkl', 'rb') as file:
    pet_A1_energy_3d = pickle.load(file)
with open('pet_A2_energy_3d_origin.pkl', 'rb') as file:
    pet_A2_energy_3d = pickle.load(file)
with open('pet_A3_energy_3d_origin.pkl', 'rb') as file:
    pet_A3_energy_3d = pickle.load(file)
with open('pet_B1_energy_3d_origin.pkl', 'rb') as file:
    pet_B1_energy_3d = pickle.load(file)
with open('pet_B2_energy_3d_origin.pkl', 'rb') as file:
    pet_B2_energy_3d = pickle.load(file)
with open('pet_B3_energy_3d_origin.pkl', 'rb') as file:
    pet_B3_energy_3d = pickle.load(file)
    
with open('pet_A1_correlation_3d_origin.pkl', 'rb') as file:
    pet_A1_correlation_3d = pickle.load(file)
with open('pet_A2_correlation_3d_origin.pkl', 'rb') as file:
    pet_A2_correlation_3d = pickle.load(file)
with open('pet_A3_correlation_3d_origin.pkl', 'rb') as file:
    pet_A3_correlation_3d = pickle.load(file)
with open('pet_B1_correlation_3d_origin.pkl', 'rb') as file:
    pet_B1_correlation_3d = pickle.load(file)
with open('pet_B2_correlation_3d_origin.pkl', 'rb') as file:
    pet_B2_correlation_3d = pickle.load(file)
with open('pet_B3_correlation_3d_origin.pkl', 'rb') as file:
    pet_B3_correlation_3d = pickle.load(file)
    
with open('pet_A1_homogeneity_3d_origin.pkl', 'rb') as file:
    pet_A1_homogeneity_3d = pickle.load(file)
with open('pet_A2_homogeneity_3d_origin.pkl', 'rb') as file:
    pet_A2_homogeneity_3d = pickle.load(file)
with open('pet_A3_homogeneity_3d_origin.pkl', 'rb') as file:
    pet_A3_homogeneity_3d = pickle.load(file)
with open('pet_B1_homogeneity_3d_origin.pkl', 'rb') as file:
    pet_B1_homogeneity_3d = pickle.load(file)
with open('pet_B2_homogeneity_3d_origin.pkl', 'rb') as file:
    pet_B2_homogeneity_3d = pickle.load(file)
with open('pet_B3_homogeneity_3d_origin.pkl', 'rb') as file:
    pet_B3_homogeneity_3d = pickle.load(file)

#%% Pick the correct geometry

pet_A1_contrast_3d_final = []
pet_A2_contrast_3d_final = []
pet_A3_contrast_3d_final = []
pet_B1_contrast_3d_final = []
pet_B2_contrast_3d_final = []
pet_B3_contrast_3d_final = []
pet_A1_energy_3d_final = []
pet_A2_energy_3d_final = []
pet_A3_energy_3d_final = []
pet_B1_energy_3d_final = []
pet_B2_energy_3d_final = []
pet_B3_energy_3d_final = []
pet_A1_correlation_3d_final = []
pet_A2_correlation_3d_final = []
pet_A3_correlation_3d_final = []
pet_B1_correlation_3d_final = []
pet_B2_correlation_3d_final = []
pet_B3_correlation_3d_final = []
pet_A1_homogeneity_3d_final = []
pet_A2_homogeneity_3d_final = []
pet_A3_homogeneity_3d_final = []
pet_B1_homogeneity_3d_final = []
pet_B2_homogeneity_3d_final = []
pet_B3_homogeneity_3d_final = []

adc_A1_contrast_3d_final = []
adc_A2_contrast_3d_final = []
adc_A3_contrast_3d_final = []
adc_B1_contrast_3d_final = []
adc_B2_contrast_3d_final = []
adc_B3_contrast_3d_final = []
adc_A1_energy_3d_final = []
adc_A2_energy_3d_final = []
adc_A3_energy_3d_final = []
adc_B1_energy_3d_final = []
adc_B2_energy_3d_final = []
adc_B3_energy_3d_final = []
adc_A1_correlation_3d_final = []
adc_A2_correlation_3d_final = []
adc_A3_correlation_3d_final = []
adc_B1_correlation_3d_final = []
adc_B2_correlation_3d_final = []
adc_B3_correlation_3d_final = []
adc_A1_homogeneity_3d_final = []
adc_A2_homogeneity_3d_final = []
adc_A3_homogeneity_3d_final = []
adc_B1_homogeneity_3d_final = []
adc_B2_homogeneity_3d_final = []
adc_B3_homogeneity_3d_final = []

dce_A1_contrast_3d_final = []
dce_A2_contrast_3d_final = []
dce_A3_contrast_3d_final = []
dce_B1_contrast_3d_final = []
dce_B2_contrast_3d_final = []
dce_B3_contrast_3d_final = []
dce_A1_energy_3d_final = []
dce_A2_energy_3d_final = []
dce_A3_energy_3d_final = []
dce_B1_energy_3d_final = []
dce_B2_energy_3d_final = []
dce_B3_energy_3d_final = []
dce_A1_correlation_3d_final = []
dce_A2_correlation_3d_final = []
dce_A3_correlation_3d_final = []
dce_B1_correlation_3d_final = []
dce_B2_correlation_3d_final = []
dce_B3_correlation_3d_final = []
dce_A1_homogeneity_3d_final = []
dce_A2_homogeneity_3d_final = []
dce_A3_homogeneity_3d_final = []
dce_B1_homogeneity_3d_final = []
dce_B2_homogeneity_3d_final = []
dce_B3_homogeneity_3d_final = []


for i in range(len(pet_A1_contrast_3d)):
    pet_A1_contrast_3d_final.append(pet_A1_contrast_3d[i][0][0][0])
    pet_A2_contrast_3d_final.append(pet_A2_contrast_3d[i][0][0][0])
    pet_A3_contrast_3d_final.append(pet_A3_contrast_3d[i][0][0][0])
    pet_A1_correlation_3d_final.append(pet_A1_contrast_3d[i][0][0][0])
    pet_A2_correlation_3d_final.append(pet_A2_contrast_3d[i][0][0][0])
    pet_A3_correlation_3d_final.append(pet_A3_contrast_3d[i][0][0][0])
    pet_A1_energy_3d_final.append(pet_A1_contrast_3d[i][0][0][0])
    pet_A2_energy_3d_final.append(pet_A2_contrast_3d[i][0][0][0])
    pet_A3_energy_3d_final.append(pet_A3_contrast_3d[i][0][0][0])
    pet_A1_homogeneity_3d_final.append(pet_A1_homogeneity_3d[i][0][0][0])
    pet_A2_homogeneity_3d_final.append(pet_A2_homogeneity_3d[i][0][0][0])
    pet_A3_homogeneity_3d_final.append(pet_A3_homogeneity_3d[i][0][0][0])
    
    dce_A1_contrast_3d_final.append(dce_A1_contrast_3d[i][0][0][0])
    dce_A2_contrast_3d_final.append(dce_A2_contrast_3d[i][0][0][0])
    dce_A3_contrast_3d_final.append(dce_A3_contrast_3d[i][0][0][0])
    dce_A1_correlation_3d_final.append(dce_A1_contrast_3d[i][0][0][0])
    dce_A2_correlation_3d_final.append(dce_A2_contrast_3d[i][0][0][0])
    dce_A3_correlation_3d_final.append(dce_A3_contrast_3d[i][0][0][0])
    dce_A1_energy_3d_final.append(dce_A1_contrast_3d[i][0][0][0])
    dce_A2_energy_3d_final.append(dce_A2_contrast_3d[i][0][0][0])
    dce_A3_energy_3d_final.append(dce_A3_contrast_3d[i][0][0][0])
    dce_A1_homogeneity_3d_final.append(dce_A1_homogeneity_3d[i][0][0][0])
    dce_A2_homogeneity_3d_final.append(dce_A2_homogeneity_3d[i][0][0][0])
    dce_A3_homogeneity_3d_final.append(dce_A3_homogeneity_3d[i][0][0][0])
    
    adc_A1_contrast_3d_final.append(adc_A1_contrast_3d[i][0][0][0])
    adc_A2_contrast_3d_final.append(adc_A2_contrast_3d[i][0][0][0])
    adc_A3_contrast_3d_final.append(adc_A3_contrast_3d[i][0][0][0])
    adc_A1_correlation_3d_final.append(adc_A1_contrast_3d[i][0][0][0])
    adc_A2_correlation_3d_final.append(adc_A2_contrast_3d[i][0][0][0])
    adc_A3_correlation_3d_final.append(adc_A3_contrast_3d[i][0][0][0])
    adc_A1_energy_3d_final.append(adc_A1_contrast_3d[i][0][0][0])
    adc_A2_energy_3d_final.append(adc_A2_contrast_3d[i][0][0][0])
    adc_A3_energy_3d_final.append(adc_A3_contrast_3d[i][0][0][0])
    adc_A1_homogeneity_3d_final.append(adc_A1_homogeneity_3d[i][0][0][0])
    adc_A2_homogeneity_3d_final.append(adc_A2_homogeneity_3d[i][0][0][0])
    adc_A3_homogeneity_3d_final.append(adc_A3_homogeneity_3d[i][0][0][0])
    
for j in range(len(pet_B1_contrast_3d)):
    pet_B1_contrast_3d_final.append(pet_B1_contrast_3d[j][0][0][0])
    pet_B2_contrast_3d_final.append(pet_B2_contrast_3d[j][0][0][0])
    pet_B3_contrast_3d_final.append(pet_B3_contrast_3d[j][0][0][0])
    pet_B1_correlation_3d_final.append(pet_B1_contrast_3d[j][0][0][0])
    pet_B2_correlation_3d_final.append(pet_B2_contrast_3d[j][0][0][0])
    pet_B3_correlation_3d_final.append(pet_B3_contrast_3d[j][0][0][0])
    pet_B1_energy_3d_final.append(pet_B1_contrast_3d[j][0][0][0])
    pet_B2_energy_3d_final.append(pet_B2_contrast_3d[j][0][0][0])
    pet_B3_energy_3d_final.append(pet_B3_contrast_3d[j][0][0][0])
    pet_B1_homogeneity_3d_final.append(pet_B1_homogeneity_3d[j][0][0][0])
    pet_B2_homogeneity_3d_final.append(pet_B2_homogeneity_3d[j][0][0][0])
    pet_B3_homogeneity_3d_final.append(pet_B3_homogeneity_3d[j][0][0][0])
    
    dce_B1_contrast_3d_final.append(dce_B1_contrast_3d[j][0][0][0])
    dce_B2_contrast_3d_final.append(dce_B2_contrast_3d[j][0][0][0])
    dce_B3_contrast_3d_final.append(dce_B3_contrast_3d[j][0][0][0])
    dce_B1_correlation_3d_final.append(dce_B1_contrast_3d[j][0][0][0])
    dce_B2_correlation_3d_final.append(dce_B2_contrast_3d[j][0][0][0])
    dce_B3_correlation_3d_final.append(dce_B3_contrast_3d[j][0][0][0])
    dce_B1_energy_3d_final.append(dce_B1_contrast_3d[j][0][0][0])
    dce_B2_energy_3d_final.append(dce_B2_contrast_3d[j][0][0][0])
    dce_B3_energy_3d_final.append(dce_B3_contrast_3d[j][0][0][0])
    dce_B1_homogeneity_3d_final.append(dce_B1_homogeneity_3d[j][0][0][0])
    dce_B2_homogeneity_3d_final.append(dce_B2_homogeneity_3d[j][0][0][0])
    dce_B3_homogeneity_3d_final.append(dce_B3_homogeneity_3d[j][0][0][0])
    
    adc_B1_contrast_3d_final.append(adc_B1_contrast_3d[j][0][0][0])
    adc_B2_contrast_3d_final.append(adc_B2_contrast_3d[j][0][0][0])
    adc_B3_contrast_3d_final.append(adc_B3_contrast_3d[j][0][0][0])
    adc_B1_correlation_3d_final.append(adc_B1_contrast_3d[j][0][0][0])
    adc_B2_correlation_3d_final.append(adc_B2_contrast_3d[j][0][0][0])
    adc_B3_correlation_3d_final.append(adc_B3_contrast_3d[j][0][0][0])
    adc_B1_energy_3d_final.append(adc_B1_contrast_3d[j][0][0][0])
    adc_B2_energy_3d_final.append(adc_B2_contrast_3d[j][0][0][0])
    adc_B3_energy_3d_final.append(adc_B3_contrast_3d[j][0][0][0])
    adc_B1_homogeneity_3d_final.append(adc_B1_homogeneity_3d[j][0][0][0])
    adc_B2_homogeneity_3d_final.append(adc_B2_homogeneity_3d[j][0][0][0])
    adc_B3_homogeneity_3d_final.append(adc_B3_homogeneity_3d[j][0][0][0])
    

#%% Conduct statistical analysis

### Step 0: Combine vector in one dimension

X1A = np.concatenate((np.array(pet_A1_contrast_3d_final), np.array(pet_A2_contrast_3d_final), np.array(pet_A3_contrast_3d_final)), axis=1)
X2A = np.concatenate((np.array(dce_A1_contrast_3d_final), np.array(dce_A2_contrast_3d_final), np.array(dce_A3_contrast_3d_final)), axis=1)
X3A = np.concatenate((np.array(adc_A1_contrast_3d_final), np.array(adc_A2_contrast_3d_final), np.array(adc_A3_contrast_3d_final)), axis=1)
X1B = np.concatenate((np.array(pet_B1_contrast_3d_final), np.array(pet_B2_contrast_3d_final), np.array(pet_B3_contrast_3d_final)), axis=1)
X2B = np.concatenate((np.array(dce_B1_contrast_3d_final), np.array(dce_B2_contrast_3d_final), np.array(dce_B3_contrast_3d_final)), axis=1)
X3B = np.concatenate((np.array(adc_B1_contrast_3d_final), np.array(adc_B2_contrast_3d_final), np.array(adc_B3_contrast_3d_final)), axis=1)
XcMax = np.max(np.concatenate((X1A,X2A,X3A,X1B,X2B,X3B),axis=0))
X1A = X1A / XcMax
X2A = X2A / XcMax
X3A = X3A / XcMax
X1B = X1B / XcMax
X2B = X2B / XcMax
X3B = X3B / XcMax

X4A = np.concatenate((np.array(pet_A1_correlation_3d_final), np.array(pet_A2_correlation_3d_final), np.array(pet_A3_correlation_3d_final)), axis=1)
X5A = np.concatenate((np.array(dce_A1_correlation_3d_final), np.array(dce_A2_correlation_3d_final), np.array(dce_A3_correlation_3d_final)), axis=1)
X6A = np.concatenate((np.array(adc_A1_correlation_3d_final), np.array(adc_A2_correlation_3d_final), np.array(adc_A3_correlation_3d_final)), axis=1)
X4B = np.concatenate((np.array(pet_B1_correlation_3d_final), np.array(pet_B2_correlation_3d_final), np.array(pet_B3_correlation_3d_final)), axis=1)
X5B = np.concatenate((np.array(dce_B1_correlation_3d_final), np.array(dce_B2_correlation_3d_final), np.array(dce_B3_correlation_3d_final)), axis=1)
X6B = np.concatenate((np.array(adc_B1_correlation_3d_final), np.array(adc_B2_correlation_3d_final), np.array(adc_B3_correlation_3d_final)), axis=1)
# XrMax = np.max(np.concatenate((X4A,X5A,X6A,X4B,X5B,X6B),axis=0))
# X4A = X4A / XrMax
# X5A = X5A / XrMax
# X6A = X6A / XrMax
# X4B = X4B / XrMax
# X5B = X5B / XrMax
# X6B = X6B / XrMax

X7A = np.concatenate((np.array(pet_A1_energy_3d_final), np.array(pet_A2_energy_3d_final), np.array(pet_A3_energy_3d_final)), axis=1)
X8A = np.concatenate((np.array(dce_A1_energy_3d_final), np.array(dce_A2_energy_3d_final), np.array(dce_A3_energy_3d_final)), axis=1)
X9A = np.concatenate((np.array(adc_A1_energy_3d_final), np.array(adc_A2_energy_3d_final), np.array(adc_A3_energy_3d_final)), axis=1)
X7B = np.concatenate((np.array(pet_B1_energy_3d_final), np.array(pet_B2_energy_3d_final), np.array(pet_B3_energy_3d_final)), axis=1)
X8B = np.concatenate((np.array(dce_B1_energy_3d_final), np.array(dce_B2_energy_3d_final), np.array(dce_B3_energy_3d_final)), axis=1)
X9B = np.concatenate((np.array(adc_B1_energy_3d_final), np.array(adc_B2_energy_3d_final), np.array(adc_B3_energy_3d_final)), axis=1)
# XeMax = np.max(np.concatenate((X7A,X8A,X9A,X7B,X8B,X9B),axis=0))
# X7A = X7A / XeMax
# X8A = X8A / XeMax
# X9A = X9A / XeMax
# X7B = X7B / XeMax
# X8B = X8B / XeMax
# X9B = X9B / XeMax

X10A = np.concatenate((np.array(pet_A1_homogeneity_3d_final), np.array(pet_A2_homogeneity_3d_final), np.array(pet_A3_homogeneity_3d_final)), axis=1)
X11A = np.concatenate((np.array(dce_A1_homogeneity_3d_final), np.array(dce_A2_homogeneity_3d_final), np.array(dce_A3_homogeneity_3d_final)), axis=1)
X12A = np.concatenate((np.array(adc_A1_homogeneity_3d_final), np.array(adc_A2_homogeneity_3d_final), np.array(adc_A3_homogeneity_3d_final)), axis=1)
X10B = np.concatenate((np.array(pet_B1_homogeneity_3d_final), np.array(pet_B2_homogeneity_3d_final), np.array(pet_B3_homogeneity_3d_final)), axis=1)
X11B = np.concatenate((np.array(dce_B1_homogeneity_3d_final), np.array(dce_B2_homogeneity_3d_final), np.array(dce_B3_homogeneity_3d_final)), axis=1)
X12B = np.concatenate((np.array(adc_B1_homogeneity_3d_final), np.array(adc_B2_homogeneity_3d_final), np.array(adc_B3_homogeneity_3d_final)), axis=1)
# XhMax = np.max(np.concatenate((X10A,X11A,X12A,X10B,X11B,X12B),axis=0))
# X10A = X10A / XhMax
# X11A = X11A / XhMax
# X12A = X12A / XhMax
# X10B = X10B / XhMax
# X11B = X11B / XhMax
# X12B = X12B / XhMax

### Step 0.25: Comnine in the other direction

XA = np.concatenate((X1A, X2A, X3A, X4A, X5A, X6A, X7A, X8A, X9A, X10A, X11A, X12A), axis=1)
XB = np.concatenate((X1B, X2B, X3B, X4B, X5B, X6B, X7B, X8B, X9B, X10B, X11B, X12B), axis=1)

### Step 0.50: Combine features together

X = np.concatenate((XA, XB), axis=0)

#%% Step 0.5: load the label vector ###

X = np.array(X)
X = np.concatenate((X,first_feature_123),axis=1)

Y = [1,1,0,1,1,1,0,0,1,1,0,0,0,0,1,0,1,1,0,0,1,1,1]
Y = np.array([1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,1,1,0,0,1])

### Step 1: Recursive feature elimination

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
num_features = X.shape[-1]  # Adjust this to the desired number of features 
              
#% Step 2: Check generalization with k-fold cross-validation

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

# Define the number of splits (k)
k = 3

# Initialize the KFold object
kf = KFold(n_splits=k)

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Lists to store evaluation metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

auc_train = []
auc_test  = []
accuracy_train = []

# Perform k-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    # rfe = RFE(model, n_features_to_select = num_features)
    # selector = rfe.fit_transform(X_train, y_train)
    
    # X_train  = X_train[:, rfe.support_]
    # X_test   = X_test[:,  rfe.support_]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # Calculate evaluation metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
   
    accuracy_train.append(accuracy_score(y_train, y_pred_train))
    auc_train.append(roc_auc_score(y_train, y_pred_train))
    auc_test.append(roc_auc_score(y_test,  y_pred))

# Calculate average evaluation metrics
average_accuracy = np.mean(accuracy_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)
average_train_score = np.mean(accuracy_train)
auc_train_score     = np.mean(auc_train)
auc_test_score      = np.mean(auc_test)

# Print results
print(f'Average Accuracy: {average_accuracy}')
print(f'Average Precision: {average_precision}')
print(f'Average Recall: {average_recall}')
print(f'Average F1-score: {average_f1}')
print(f'Average Accuracy Train: {average_train_score}')
print(f'Average AUC Train: {auc_train_score}')
print(f'Average AUC Test: {auc_test_score}')









