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

# Feature follows (PET-ADC-DCE: t1_max, t1_mean, t2_max, t2_mean, t3_max, t3_mean, size*3)
zero_feature_1   = np.load('zero_order_feature_1.npy')
zero_feature_12  = np.load('zero_order_feature_12.npy')
zero_feature_123 = np.load('zero_order_feature_123.npy')

zero_feature_pet = np.concatenate((zero_feature_123[:,:6],  zero_feature_123[:,18:]), axis=1)
zero_feature_adc = np.concatenate((zero_feature_123[:,6:12],zero_feature_123[:,18:]), axis=1)
zero_feature_dce = np.concatenate((zero_feature_123[:,12:], zero_feature_123[:,18:]), axis=1)

# Feature follows (t1: mean, quantile, coeff_var, skewness, kurtosis, max_value. Then t2, t3 following)
first_feature_adc = np.load('first_order_feature_adc_all.npy') 
first_feature_dce = np.load('first_order_feature_dce_all.npy') 
first_feature_pet = np.load('first_order_feature_pet_all.npy') 

first_feature_1   = np.concatenate((first_feature_adc[:,:6],first_feature_dce[:,:6],first_feature_pet[:,:6]),axis=1)
first_feature_12  = np.concatenate((first_feature_adc[:,:12],first_feature_dce[:,:12],first_feature_pet[:,:12]),axis=1)
first_feature_123 =np.concatenate((first_feature_adc,first_feature_dce,first_feature_pet),axis=1)

#%% Load the data for analysis

### ADC FEATURES
with open('adc_T1_contrast_origin_new.pkl', 'rb') as file:
    adc_T1_contrast = pickle.load(file)
with open('adc_T2_contrast_origin_new.pkl', 'rb') as file:
    adc_T2_contrast = pickle.load(file)
with open('adc_T3_contrast_origin_new.pkl', 'rb') as file:
    adc_T3_contrast = pickle.load(file)
    
with open('adc_T1_energy_origin_new.pkl', 'rb') as file:
    adc_T1_energy = pickle.load(file)
with open('adc_T2_energy_origin_new.pkl', 'rb') as file:
    adc_T2_energy = pickle.load(file)
with open('adc_T3_energy_origin_new.pkl', 'rb') as file:
    adc_T3_energy = pickle.load(file)
    
with open('adc_T1_correlation_origin_new.pkl', 'rb') as file:
    adc_T1_correlation = pickle.load(file)
with open('adc_T2_correlation_origin_new.pkl', 'rb') as file:
    adc_T2_correlation = pickle.load(file)
with open('adc_T3_correlation_origin_new.pkl', 'rb') as file:
    adc_T3_correlation = pickle.load(file)
    
with open('adc_T1_homogeneity_origin_new.pkl', 'rb') as file:
    adc_T1_homogeneity = pickle.load(file)
with open('adc_T2_homogeneity_origin_new.pkl', 'rb') as file:
    adc_T2_homogeneity = pickle.load(file)
with open('adc_T3_homogeneity_origin_new.pkl', 'rb') as file:
    adc_T3_homogeneity = pickle.load(file)
    
    
### DCE FEATURES
with open('dce_T1_contrast_origin_new.pkl', 'rb') as file:
    dce_T1_contrast = pickle.load(file)
with open('dce_T2_contrast_origin_new.pkl', 'rb') as file:
    dce_T2_contrast = pickle.load(file)
with open('dce_T3_contrast_origin_new.pkl', 'rb') as file:
    dce_T3_contrast = pickle.load(file)
    
with open('dce_T1_energy_origin_new.pkl', 'rb') as file:
    dce_T1_energy = pickle.load(file)
with open('dce_T2_energy_origin_new.pkl', 'rb') as file:
    dce_T2_energy = pickle.load(file)
with open('dce_T3_energy_origin_new.pkl', 'rb') as file:
    dce_T3_energy = pickle.load(file)
    
with open('dce_T1_correlation_origin_new.pkl', 'rb') as file:
    dce_T1_correlation = pickle.load(file)
with open('dce_T2_correlation_origin_new.pkl', 'rb') as file:
    dce_T2_correlation = pickle.load(file)
with open('dce_T3_correlation_origin_new.pkl', 'rb') as file:
    dce_T3_correlation = pickle.load(file)
    
with open('dce_T1_homogeneity_origin_new.pkl', 'rb') as file:
    dce_T1_homogeneity = pickle.load(file)
with open('dce_T2_homogeneity_origin_new.pkl', 'rb') as file:
    dce_T2_homogeneity = pickle.load(file)
with open('dce_T3_homogeneity_origin_new.pkl', 'rb') as file:
    dce_T3_homogeneity = pickle.load(file)
    
    
### PET FEATURES
with open('pet_T1_contrast_origin_new.pkl', 'rb') as file:
    pet_T1_contrast = pickle.load(file)
with open('pet_T2_contrast_origin_new.pkl', 'rb') as file:
    pet_T2_contrast = pickle.load(file)
with open('pet_T3_contrast_origin_new.pkl', 'rb') as file:
    pet_T3_contrast = pickle.load(file)
    
with open('pet_T1_energy_origin_new.pkl', 'rb') as file:
    pet_T1_energy = pickle.load(file)
with open('pet_T2_energy_origin_new.pkl', 'rb') as file:
    pet_T2_energy = pickle.load(file)
with open('pet_T3_energy_origin_new.pkl', 'rb') as file:
    pet_T3_energy = pickle.load(file)
    
with open('pet_T1_correlation_origin_new.pkl', 'rb') as file:
    pet_T1_correlation = pickle.load(file)
with open('pet_T2_correlation_origin_new.pkl', 'rb') as file:
    pet_T2_correlation = pickle.load(file)
with open('pet_T3_correlation_origin_new.pkl', 'rb') as file:
    pet_T3_correlation = pickle.load(file)
    
with open('pet_T1_homogeneity_origin_new.pkl', 'rb') as file:
    pet_T1_homogeneity = pickle.load(file)
with open('pet_T2_homogeneity_origin_new.pkl', 'rb') as file:
    pet_T2_homogeneity = pickle.load(file)
with open('pet_T3_homogeneity_origin_new.pkl', 'rb') as file:
    pet_T3_homogeneity = pickle.load(file)
    
#%% Step 0.25: Preprocessing the data

adc_T1_contrast_final = []
adc_T2_contrast_final = []
adc_T3_contrast_final = []
dce_T1_contrast_final = []
dce_T2_contrast_final = []
dce_T3_contrast_final = []
pet_T1_contrast_final = []
pet_T2_contrast_final = []
pet_T3_contrast_final = []

adc_T1_correlation_final = []
adc_T2_correlation_final = []
adc_T3_correlation_final = []
dce_T1_correlation_final = []
dce_T2_correlation_final = []
dce_T3_correlation_final = []
pet_T1_correlation_final = []
pet_T2_correlation_final = []
pet_T3_correlation_final = []

adc_T1_energy_final = []
adc_T2_energy_final = []
adc_T3_energy_final = []
dce_T1_energy_final = []
dce_T2_energy_final = []
dce_T3_energy_final = []
pet_T1_energy_final = []
pet_T2_energy_final = []
pet_T3_energy_final = []

adc_T1_homogeneity_final = []
adc_T2_homogeneity_final = []
adc_T3_homogeneity_final = []
dce_T1_homogeneity_final = []
dce_T2_homogeneity_final = []
dce_T3_homogeneity_final = []
pet_T1_homogeneity_final = []
pet_T2_homogeneity_final = []
pet_T3_homogeneity_final = []

L = len(adc_T1_contrast)

for i in range(L):
    adc_T1_contrast_final.append([arr for arr in adc_T1_contrast[i] if np.var(arr) != 0])
for i in range(L):
    adc_T2_contrast_final.append([arr for arr in adc_T2_contrast[i] if np.var(arr) != 0])
for i in range(L):
    adc_T3_contrast_final.append([arr for arr in adc_T3_contrast[i] if np.var(arr) != 0])
for i in range(L):
    adc_T1_correlation_final.append([arr for arr in adc_T1_correlation[i] if np.var(arr) != 0])
for i in range(L):
    adc_T2_correlation_final.append([arr for arr in adc_T2_correlation[i] if np.var(arr) != 0])
for i in range(L):
    adc_T3_correlation_final.append([arr for arr in adc_T3_correlation[i] if np.var(arr) != 0])
for i in range(L):
    adc_T1_energy_final.append([arr for arr in adc_T1_energy[i] if np.var(arr) != 0])
for i in range(L):
    adc_T2_energy_final.append([arr for arr in adc_T2_energy[i] if np.var(arr) != 0])
for i in range(L):
    adc_T3_energy_final.append([arr for arr in adc_T3_energy[i] if np.var(arr) != 0])
for i in range(L):
    adc_T1_homogeneity_final.append([arr for arr in adc_T1_homogeneity[i] if np.var(arr) != 0])
for i in range(L):
    adc_T2_homogeneity_final.append([arr for arr in adc_T2_homogeneity[i] if np.var(arr) != 0])
for i in range(L):
    adc_T3_homogeneity_final.append([arr for arr in adc_T3_homogeneity[i] if np.var(arr) != 0])
    
for i in range(L):
    dce_T1_contrast_final.append([arr for arr in dce_T1_contrast[i] if np.var(arr) != 0])
for i in range(L):
    dce_T2_contrast_final.append([arr for arr in dce_T2_contrast[i] if np.var(arr) != 0])
for i in range(L):
    dce_T3_contrast_final.append([arr for arr in dce_T3_contrast[i] if np.var(arr) != 0])
for i in range(L):
    dce_T1_correlation_final.append([arr for arr in dce_T1_correlation[i] if np.var(arr) != 0])
for i in range(L):
    dce_T2_correlation_final.append([arr for arr in dce_T2_correlation[i] if np.var(arr) != 0])
for i in range(L):
    dce_T3_correlation_final.append([arr for arr in dce_T3_correlation[i] if np.var(arr) != 0])
for i in range(L):
    dce_T1_energy_final.append([arr for arr in dce_T1_energy[i] if np.var(arr) != 0])
for i in range(L):
    dce_T2_energy_final.append([arr for arr in dce_T2_energy[i] if np.var(arr) != 0])
for i in range(L):
    dce_T3_energy_final.append([arr for arr in dce_T3_energy[i] if np.var(arr) != 0])
for i in range(L):
    dce_T1_homogeneity_final.append([arr for arr in dce_T1_homogeneity[i] if np.var(arr) != 0])
for i in range(L):
    dce_T2_homogeneity_final.append([arr for arr in dce_T2_homogeneity[i] if np.var(arr) != 0])
for i in range(L):
    dce_T3_homogeneity_final.append([arr for arr in dce_T3_homogeneity[i] if np.var(arr) != 0])
    
for i in range(L):
    pet_T1_contrast_final.append([arr for arr in pet_T1_contrast[i] if np.var(arr) != 0])
for i in range(L):
    pet_T2_contrast_final.append([arr for arr in pet_T2_contrast[i] if np.var(arr) != 0])
for i in range(L):
    pet_T3_contrast_final.append([arr for arr in pet_T3_contrast[i] if np.var(arr) != 0])
for i in range(L):
    pet_T1_correlation_final.append([arr for arr in pet_T1_correlation[i] if np.var(arr) != 0])
for i in range(L):
    pet_T2_correlation_final.append([arr for arr in pet_T2_correlation[i] if np.var(arr) != 0])
for i in range(L):
    pet_T3_correlation_final.append([arr for arr in pet_T3_correlation[i] if np.var(arr) != 0])
for i in range(L):
    pet_T1_energy_final.append([arr for arr in pet_T1_energy[i] if np.var(arr) != 0])
for i in range(L):
    pet_T2_energy_final.append([arr for arr in pet_T2_energy[i] if np.var(arr) != 0])
for i in range(L):
    pet_T3_energy_final.append([arr for arr in pet_T3_energy[i] if np.var(arr) != 0])
for i in range(L):
    pet_T1_homogeneity_final.append([arr for arr in pet_T1_homogeneity[i] if np.var(arr) != 0])
for i in range(L):
    pet_T2_homogeneity_final.append([arr for arr in pet_T2_homogeneity[i] if np.var(arr) != 0])
for i in range(L):
    pet_T3_homogeneity_final.append([arr for arr in pet_T3_homogeneity[i] if np.var(arr) != 0])
    
#%% Step 0.4: Preprocessing the contrast value

flat_adc_T1_contrast = []
flat_adc_T2_contrast = []
flat_adc_T3_contrast = []
flat_dce_T1_contrast = []
flat_dce_T2_contrast = []
flat_dce_T3_contrast = []
flat_pet_T1_contrast = []
flat_pet_T2_contrast = []
flat_pet_T3_contrast = []

flat_adc_T1_correlation = []
flat_adc_T2_correlation = []
flat_adc_T3_correlation = []
flat_dce_T1_correlation = []
flat_dce_T2_correlation = []
flat_dce_T3_correlation = []
flat_pet_T1_correlation = []
flat_pet_T2_correlation = []
flat_pet_T3_correlation = []

flat_adc_T1_energy = []
flat_adc_T2_energy = []
flat_adc_T3_energy = []
flat_dce_T1_energy = []
flat_dce_T2_energy = []
flat_dce_T3_energy = []
flat_pet_T1_energy = []
flat_pet_T2_energy = []
flat_pet_T3_energy = []

flat_adc_T1_homogeneity = []
flat_adc_T2_homogeneity = []
flat_adc_T3_homogeneity = []
flat_dce_T1_homogeneity = []
flat_dce_T2_homogeneity = []
flat_dce_T3_homogeneity = []
flat_pet_T1_homogeneity = []
flat_pet_T2_homogeneity = []
flat_pet_T3_homogeneity = []

for i in range(L):

    flat_pet_T1_contrast.append(np.concatenate(pet_T1_contrast_final[i]).flatten())
    flat_pet_T2_contrast.append(np.concatenate(pet_T2_contrast_final[i]).flatten())
    flat_pet_T3_contrast.append(np.concatenate(pet_T3_contrast_final[i]).flatten())
    flat_dce_T1_contrast.append(np.concatenate(dce_T1_contrast_final[i]).flatten())
    flat_dce_T2_contrast.append(np.concatenate(dce_T2_contrast_final[i]).flatten())
    flat_dce_T3_contrast.append(np.concatenate(dce_T3_contrast_final[i]).flatten())
    flat_adc_T1_contrast.append(np.concatenate(adc_T1_contrast_final[i]).flatten())
    flat_adc_T2_contrast.append(np.concatenate(adc_T2_contrast_final[i]).flatten())
    flat_adc_T3_contrast.append(np.concatenate(adc_T3_contrast_final[i]).flatten())
    
    flat_pet_T1_correlation.append(np.concatenate(pet_T1_correlation_final[i]).flatten())
    flat_pet_T2_correlation.append(np.concatenate(pet_T2_correlation_final[i]).flatten())
    flat_pet_T3_correlation.append(np.concatenate(pet_T3_correlation_final[i]).flatten())
    flat_dce_T1_correlation.append(np.concatenate(dce_T1_correlation_final[i]).flatten())
    flat_dce_T2_correlation.append(np.concatenate(dce_T2_correlation_final[i]).flatten())
    flat_dce_T3_correlation.append(np.concatenate(dce_T3_correlation_final[i]).flatten())
    flat_adc_T1_correlation.append(np.concatenate(adc_T1_correlation_final[i]).flatten())
    flat_adc_T2_correlation.append(np.concatenate(adc_T2_correlation_final[i]).flatten())
    flat_adc_T3_correlation.append(np.concatenate(adc_T3_correlation_final[i]).flatten())
    
    flat_pet_T1_energy.append(np.concatenate(pet_T1_energy_final[i]).flatten())
    flat_pet_T2_energy.append(np.concatenate(pet_T2_energy_final[i]).flatten())
    flat_pet_T3_energy.append(np.concatenate(pet_T3_energy_final[i]).flatten())
    flat_dce_T1_energy.append(np.concatenate(dce_T1_energy_final[i]).flatten())
    flat_dce_T2_energy.append(np.concatenate(dce_T2_energy_final[i]).flatten())
    flat_dce_T3_energy.append(np.concatenate(dce_T3_energy_final[i]).flatten())
    flat_adc_T1_energy.append(np.concatenate(adc_T1_energy_final[i]).flatten())
    flat_adc_T2_energy.append(np.concatenate(adc_T2_energy_final[i]).flatten())
    flat_adc_T3_energy.append(np.concatenate(adc_T3_energy_final[i]).flatten())
    
    flat_pet_T1_homogeneity.append(np.concatenate(pet_T1_homogeneity_final[i]).flatten())
    flat_pet_T2_homogeneity.append(np.concatenate(pet_T2_homogeneity_final[i]).flatten())
    flat_pet_T3_homogeneity.append(np.concatenate(pet_T3_homogeneity_final[i]).flatten())
    flat_dce_T1_homogeneity.append(np.concatenate(dce_T1_homogeneity_final[i]).flatten())
    flat_dce_T2_homogeneity.append(np.concatenate(dce_T2_homogeneity_final[i]).flatten())
    flat_dce_T3_homogeneity.append(np.concatenate(dce_T3_homogeneity_final[i]).flatten())
    flat_adc_T1_homogeneity.append(np.concatenate(adc_T1_homogeneity_final[i]).flatten())
    flat_adc_T2_homogeneity.append(np.concatenate(adc_T2_homogeneity_final[i]).flatten())
    flat_adc_T3_homogeneity.append(np.concatenate(adc_T3_homogeneity_final[i]).flatten())

#%% Record the length of different patients

len_patient_T1 = []
len_patient_T2 = []
len_patient_T3 = []

for i in range(L):
    # list_e1 = [flat_dce_T1_energy[i],      flat_dce_T2_energy[i],      flat_dce_T3_energy[i]]
    len_patient_T1.append(len(flat_dce_T1_energy[i]))
    len_patient_T2.append(len(flat_dce_T2_energy[i]))
    len_patient_T3.append(len(flat_dce_T3_energy[i]))

min_len_T1 = np.min(len_patient_T1)
min_len_T2 = np.min(len_patient_T2)
min_len_T3 = np.min(len_patient_T3)

#%% Step 0.45: Combine all values

List_contrast = []
List_energy   = []
List_correlation = []
List_homogeneity = []

for i in range(L):
    
    # lists1 = [flat_dce_T1_energy[i]]
    # lists1 = [flat_dce_T1_energy[i],      flat_dce_T2_energy[i]]
    lists1 = [flat_dce_T1_energy[i][:min_len_T1],      flat_dce_T2_energy[i][:min_len_T2],      flat_dce_T3_energy[i][:min_len_T3]]
    List_energy.append(np.concatenate(lists1))
     
    # lists2 = [flat_dce_T1_contrast[i]]
    # lists2 = [flat_dce_T1_contrast[i], flat_dce_T2_contrast[i]]
    lists2 = [flat_dce_T1_contrast[i][:min_len_T1], flat_dce_T2_contrast[i][:min_len_T2], flat_dce_T3_contrast[i][:min_len_T3]]
    # lists2 = [flat_dce_T1_contrast[i]/np.max(flat_dce_T1_contrast[i]),flat_dce_T2_contrast[i]/np.max(flat_dce_T2_contrast[i]),flat_dce_T3_contrast[i]/np.max(flat_dce_T3_contrast[i])]
    List_contrast.append(np.concatenate(lists2))
    
    # lists3 = [flat_dce_T1_correlation[i]]
    # lists3 = [flat_dce_T1_correlation[i], flat_dce_T2_correlation[i]]
    lists3 = [flat_dce_T1_correlation[i][:min_len_T1], flat_dce_T2_correlation[i][:min_len_T2], flat_dce_T3_correlation[i][:min_len_T3]]
    List_correlation.append(np.concatenate(lists3))
    
    # lists4 = [flat_dce_T1_homogeneity[i]]
    # lists4 = [flat_dce_T1_homogeneity[i], flat_dce_T2_homogeneity[i]]
    lists4 = [flat_dce_T1_homogeneity[i][:min_len_T1], flat_dce_T2_homogeneity[i][:min_len_T2], flat_dce_T3_homogeneity[i][:min_len_T3]]
    List_homogeneity.append(np.concatenate(lists4))

L_en_list = []
L_ct_list = []
L_cr_list = []
L_hm_list = []

for j in range(L):
    L_en_list.append(len(List_energy[j]))
    L_ct_list.append(len(List_contrast[j]))
    L_cr_list.append(len(List_correlation[j]))
    L_hm_list.append(len(List_homogeneity[j]))

L_en = np.min(L_en_list)
L_ct = np.min(L_ct_list)
L_cr = np.min(L_cr_list)
L_hm = np.min(L_hm_list)
    
List_contrast_uniform = []
List_energy_uniform   = []
List_correlation_uniform = []
List_homogeneity_uniform = []

for i in range(L):
    # lists1_uniform = [flat_dce_T1_energy[i][:min_len_T1]]
    # lists1_uniform = [flat_dce_T1_energy[i][:min_len_T1],      flat_dce_T2_energy[i][:min_len_T2]]
    lists1_uniform = [flat_dce_T1_energy[i][:min_len_T1],      flat_dce_T2_energy[i][:min_len_T2],      flat_dce_T3_energy[i][:min_len_T3]]
    List_energy_uniform.append(np.concatenate(lists1_uniform))
     
    # lists2_uniform = [flat_dce_T1_contrast[i][:min_len_T1],    flat_dce_T2_contrast[i][:min_len_T2],    flat_dce_T3_contrast[i][:min_len_T3]]
    # lists2_uniform = [flat_dce_T1_contrast[i][:min_len_T1]/np.max(flat_dce_T1_contrast[i])]
    # lists2_uniform = [flat_dce_T1_contrast[i][:min_len_T1]/np.max(flat_dce_T1_contrast[i]),flat_dce_T2_contrast[i][:min_len_T2]/np.max(flat_dce_T2_contrast[i])]
    lists2_uniform = [flat_dce_T1_contrast[i][:min_len_T1]/np.max(flat_dce_T1_contrast[i]),flat_dce_T2_contrast[i][:min_len_T2]/np.max(flat_dce_T2_contrast[i]),flat_dce_T3_contrast[i][:min_len_T3]/np.max(flat_dce_T3_contrast[i])]
    List_contrast_uniform.append(np.concatenate(lists2_uniform))
    
    # lists3_uniform = [flat_dce_T1_correlation[i][:min_len_T1]]
    # lists3_uniform = [flat_dce_T1_correlation[i][:min_len_T1], flat_dce_T2_correlation[i][:min_len_T2]]
    lists3_uniform = [flat_dce_T1_correlation[i][:min_len_T1], flat_dce_T2_correlation[i][:min_len_T2], flat_dce_T3_correlation[i][:min_len_T3]]
    List_correlation_uniform.append(np.concatenate(lists3_uniform))
    
    # lists4_uniform = [flat_dce_T1_homogeneity[i][:min_len_T1]]
    # lists4_uniform = [flat_dce_T1_homogeneity[i][:min_len_T1], flat_dce_T2_homogeneity[i][:min_len_T2]]
    lists4_uniform = [flat_dce_T1_homogeneity[i][:min_len_T1], flat_dce_T2_homogeneity[i][:min_len_T2], flat_dce_T3_homogeneity[i][:min_len_T3]]
    List_homogeneity_uniform.append(np.concatenate(lists4_uniform))
    

######### This is only temporary testing code, please don't use it for 3 period testing ##############

# List_contrast_uniform = []
# List_energy_uniform   = []
# List_correlation_uniform = []
# List_homogeneity_uniform = []

# for i in range(L):
#     lists1 = [flat_dce_T1_energy[i][:min_len_T1]]
#     List_energy_uniform.append(np.concatenate(lists1))
    
#     lists2 = [flat_dce_T1_contrast[i][:min_len_T1]]
#     List_contrast_uniform.append(np.concatenate(lists2))
    
#     lists3 = [flat_dce_T1_correlation[i][:min_len_T1]]
#     List_correlation_uniform.append(np.concatenate(lists3))
    
#     lists4 = [flat_dce_T1_homogeneity[i][:min_len_T1]]
#     List_homogeneity_uniform.append(np.concatenate(lists4))    
    
#####################################################################################################    
    
List_final = []

for i in range(L):
    List = [List_energy[i], List_contrast[i], List_correlation[i], List_homogeneity[i]]
    List_final.append(np.concatenate(List))
    
List_final_uniform = []

for i in range(L):
    print('Working on ID: ' + str(i))
    List_uniform = [List_energy_uniform[i], List_contrast_uniform[i], List_correlation_uniform[i], List_homogeneity_uniform[i]]
    List_final_uniform.append(np.concatenate(List_uniform))
    

#%% Obtain the minimum length
len_list_energy   = []
len_list_contrast = []
len_list_correlation = []
len_list_homogeneity = []

for i in range(L):
    len_list_energy.append(len(List_energy[i]))
    len_list_contrast.append(len(List_contrast[i]))
    len_list_correlation.append(len(List_correlation[i]))
    len_list_homogeneity.append(len(List_homogeneity[i]))
    
min_len = np.min(len_list_energy) + np.min(len_list_contrast) + np.min(len_list_correlation) + np.min(len_list_homogeneity)
print('The common length should at least be: ', min_len)
print('The Energy length should be: ',          np.min(len_list_energy))
print('The Contrast length should be: ',        np.min(len_list_contrast))
print('The Correlation length should be: ',     np.min(len_list_correlation))
print('The Homogeneity length should be: ',     np.min(len_list_homogeneity))

#%% Step 0.5 Randomly pick some features

#%% Step 0.5 Randomly pick some features

import random   
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.linear_model import LogisticRegression
List_final = List_final_uniform

# Get the minimum length
len_record = []

for i in range(len(List_final)):
    len_record.append(len(List_final[i]))
    
min_len = np.min(len_record)

# Define the number of splits (k)
k = 3

# Initialize the KFold object
kf = KFold(n_splits=k)

sample_len = []

for ii in range(20):
    sample_len.append(len(List_final[ii]))

num_values_to_select = np.min(sample_len)

# Lists to store evaluation metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

auc_train = []
auc_test  = []
accuracy_train = []

model = LogisticRegression(max_iter=1000)
num_features = 144
# num_features = 18
 # Adjust this to the desired number of features               

for kkk in range(1):
    
    # sampled_values = []

    # for arr in List_final_uniform:
    #     sampled_indices = random.sample(range(len(arr)), min(len(arr), num_values_to_select))
    #     # sampled_indices = np.arange(0.,480)
    #     sampled_values.append(arr[sampled_indices])
        
    # print(len(sampled_values))
    
    X = np.array(List_final)
    X = np.concatenate((X,first_dce), axis=1)
    # X = np.concatenate((X,zero_dce), axis=1)
    # X = zero_all
    
    # X = np.concatenate((X1,X2,X3,X4))
    
    # Y = np.array([1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,1,1,0,0,1])
    Y = np.array([1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,1,1,0,0,1])
    
    # Step 1: Recursive feature elimination

    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.linear_model import LogisticRegression
    
    # Define the number of splits (k)
    # k = 5
    
    # Initialize the KFold object
    kf = KFold(n_splits=k)
    
    # Initialize the model
    # model = LogisticRegression()
    
    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(X): 
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        # rfe = RFE(model, n_features_to_select = num_features)
        # selector = rfe.fit_transform(X_train, y_train)
        # X_train  = X_train[:, rfe.support_]
        # X_test   = X_test[:,  rfe.support_]
        
        # Fit the model on the training data
        model.fit(X_train, y_train)
        
        # Evaluate the model on the test data
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
average_accuracy  = np.mean(accuracy_scores)
average_precision = np.mean(precision_scores)
average_recall    = np.mean(recall_scores)
average_f1        = np.mean(f1_scores)

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

#%% Feature Importance Testing

def rand_select(mtx, label, num):
    np.random.seed(0)
    num_rows_to_select = num
    indices = np.random.choice(19, num_rows_to_select, replace=False)
    select_rows  = mtx[indices]
    print(select_rows.shape)
    select_label = label[indices]
    print(select_label.shape)
    
    return select_rows, select_label

import statsmodels.api as sm

# Get the minimum length
len_record = []

List_final = List_final_uniform

for i in range(len(List_final)):
    len_record.append(len(List_final[i]))
    
min_len = np.min(len_record)

sampled_values = []

for arr in List_final:
    # sampled_indices = random.sample(range(len(arr)), min(len(arr), num_values_to_select))
    sampled_indices = list(np.arange(0,min_len))
    sampled_values.append(arr[sampled_indices])
    
# print(len(sampled_values))

X = np.vstack(sampled_values)

Y = [1,1,0,1,1,1,0,0,1,1,0,0,0,0,1,0,1,1,0,0,1,1,1]
Y = np.array([1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,1,1,0,0,1])

num_features = 12
rfe = RFE(model, n_features_to_select = num_features)
coefficients = rfe.estimator.coef_

# Fit RFE on your data
X_selected = rfe.fit_transform(X, Y)

# Get the selected features
selected_feature_indices = rfe.support_
selected_features = X[:,selected_feature_indices]

# Get the indices of the selected features
coeff_id = 0
# LEN = min_len_T1
# LEN = min_len_T1 + min_len_T2
LEN = min_len_T1 + min_len_T2 + min_len_T3
# LEN = min_len_T1

for i in range(len(selected_feature_indices)):
    if selected_feature_indices[i] == True:
        if i < LEN:
            lab_text = 'ENERGY'
            coeff_id = coeff_id + 1
        elif i < LEN*2:
            lab_text = 'CONTRAST'
            coeff_id = coeff_id + 1
        elif i < LEN*3:
            lab_text = 'CORRELATION'
            coeff_id = coeff_id + 1
        else:
            lab_text = 'HOMOGENEITY'
            coeff_id = coeff_id + 1
            
        print('Feature is selected at location: ' + str(i) + ' with Feature name: ' + lab_text + ' with the coefficient value: ' + str(coefficients[0][coeff_id-1]))

# X_train = sm.add_constant(X)
# y_train = Responder_reduce
# Fit the regression model
# model = sm.OLS(y_train, X_train)
# results = model.fit()

# print(results.summary())




























