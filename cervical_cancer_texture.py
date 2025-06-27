#%% This code finalize the statistical analysis of cervical data set

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

#%% Load the PARTs and ROIs

### First p#: period (time) Second p#: part (p1: cervical, p2:tumor, c:combine)
# os.chdir('/Users/hafeng/.spyder-py3/cervical_cancer_analysis/')

# p1_p1_data = np.load('mask_3ds_p1_p1_resize_all.npy')
# p1_p2_data = np.load('mask_3ds_p1_p2_resize_all.npy')
# p1_c_data  = np.load('mask_3ds_p1_c_resize_all.npy')

# p2_p1_data = np.load('mask_3ds_p2_p1_resize_all.npy')
# p2_c_data  = np.load('mask_3ds_p2_c_resize_all.npy')

# p3_p1_data = np.load('mask_3ds_p3_p1_resize_all.npy')
# p3_p2_data = np.load('mask_3ds_p3_p2_resize_all.npy')
# p3_c_data  = np.load('mask_3ds_p3_c_resize_all.npy')

# p1_p1_data = p1_p1_data.astype('double')
# p2_p1_data = p2_p1_data.astype('double')
# p3_p1_data = p3_p1_data.astype('double')

# p1_p2_data = p1_p2_data.astype('double')
# p2_p2_data = p2_p2_data.astype('double')
# p3_p2_data = p3_p2_data.astype('double')

# p1_c_data  = p1_c_data.astype('double')
# p2_c_data  = p2_c_data.astype('double')
# p3_c_data  = p3_c_data.astype('double') 
    
p1_p1_data = np.load('mask_3ds_p1_p1_resize_di.npy')
p1_p2_data = np.load('mask_3ds_p1_p2_resize_di.npy')
p1_c_data  = np.load('mask_3ds_p1_c_resize_di.npy')

p2_p1_data = np.load('mask_3ds_p2_p1_resize_di.npy')
p2_p2_data = np.load('mask_3ds_p2_p2_resize_di.npy')
p2_c_data  = np.load('mask_3ds_p2_c_resize_di.npy')

p3_p1_data = np.load('mask_3ds_p3_p1_resize_di.npy')
p3_p2_data = np.load('mask_3ds_p3_p2_resize_di.npy')
p3_c_data  = np.load('mask_3ds_p3_c_resize_di.npy')

p1_p1_data = p1_p1_data.astype('double')
p2_p1_data = p2_p1_data.astype('double')
p3_p1_data = p3_p1_data.astype('double')

p1_p2_data = p1_p2_data.astype('double')
p2_p2_data = p2_p2_data.astype('double')
p3_p2_data = p3_p2_data.astype('double')

p1_c_data  = p1_c_data.astype('double')
p2_c_data  = p2_c_data.astype('double')
p3_c_data  = p3_c_data.astype('double') 

#%% Split the mask to A and B patients

t1_p1_mask_A = p1_p1_data[:12]
t1_p1_mask_B = p1_p1_data[12:]
t1_p2_mask_A = p1_p2_data[:12]
t1_p2_mask_B = p1_p2_data[12:]
t2_p1_mask_A = p2_p1_data[:12]
t2_p1_mask_B = p2_p1_data[12:]
t2_p2_mask_A = p2_p2_data[:12]
t2_p2_mask_B = p2_p2_data[12:]
t3_p1_mask_A = p3_p1_data[:12]
t3_p1_mask_B = p3_p1_data[12:]
t3_p2_mask_A = p3_p2_data[:12]
t3_p2_mask_B = p3_p2_data[12:]

#%% Load the Medical Image Matrices

pet_A1 = np.load('pet_A1_3d_di.npy')
pet_A2 = np.load('pet_A2_3d_di.npy')
pet_A3 = np.load('pet_A3_3d_di.npy')
pet_B1 = np.load('pet_B1_3d_di.npy')
pet_B2 = np.load('pet_B2_3d_di.npy')
pet_B3 = np.load('pet_B3_3d_di.npy')

dce_A1 = np.load('dce_A1_3d_di.npy')
dce_A2 = np.load('dce_A2_3d_di.npy')
dce_A3 = np.load('dce_A3_3d_di.npy')
dce_B1 = np.load('dce_B1_3d_di.npy')
dce_B2 = np.load('dce_B2_3d_di.npy')
dce_B3 = np.load('dce_B3_3d_di.npy')

adc_A1 = np.load('adc_A1_3d_di.npy')
adc_A2 = np.load('adc_A2_3d_di.npy')
adc_A3 = np.load('adc_A3_3d_di.npy')
adc_B1 = np.load('adc_B1_3d_di.npy')
adc_B2 = np.load('adc_B2_3d_di.npy')
adc_B3 = np.load('adc_B3_3d_di.npy')

#%% Compute Texture Features

from radiomics import featureextractor
import SimpleITK as sitk

# image = adc_A1[0,:,:,0]

# # Create a whole-image mask
# mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
# mask.SetSpacing(image.GetSpacing())
# mask.SetOrigin(image.GetOrigin())
# mask.SetDirection(image.GetDirection())
# mask_array = sitk.GetArrayFromImage(mask)
# mask_array.fill(1)
# mask = sitk.GetImageFromArray(mask_array)

#%% 2D GLCM Code Package

from skimage.feature import graycomatrix
import numpy as np

# Assuming 'image' is your 2D grayscale image
# Make sure 'image' is properly loaded and in a format where intensity values can be interpreted (e.g., 8-bit grayscale)
image = np.array([[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]])


# Define the distance and angles for co-occurrence computation
distances = [1]  # You can specify multiple distances
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # You can use different angles (in radians)

# Compute GLCM
glcm = graycomatrix(image.astype(np.uint8), distances=distances, angles=angles, symmetric=True, normed=True)

# 'glcm' now contains the computed GLCM

#%% 2D GLCM Code Self-Implement

import numpy as np

volume = np.array([[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]])

def compute_2d_glcm(volume, distances=[1], angles=[(1,0), (0,1), (1,1), (1,-1)]):
    glcm = np.zeros((len(distances), len(angles), np.max(volume)+1, np.max(volume)+1), dtype=np.uint32)

    for d, distance in enumerate(distances):
        for a, angle in enumerate(angles):
            for x in range(volume.shape[0]):
                for y in range(volume.shape[1]):
                    
                    if (0 <= x + distance*angle[0] < volume.shape[0] and 0 <= y + distance*angle[1] < volume.shape[1]):
                        glcm[d, a, volume[y, x], volume[y + distance*angle[1], x + distance*angle[0]]] += 1

    return glcm

# Example Usage
# Assuming 'volume' is your 3D grayscale image in NumPy array format
# 'volume' should be a 3D array where each element represents a voxel with a grayscale value
# 'distances' is a list of distances you want to consider
# 'angles' is a list of 3D directions in which co-occurrence is calculated
glcm_matrix = compute_2d_glcm(volume)

#%% 2D GLCM Code Haralick Self-Implement

import numpy as np

volume = np.array([[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]])

def compute_2d_glcm_ha(volume, distances=[1], angles=[(1,0), (0,1), (1,1), (1,-1)]):
    glcm = np.zeros((len(distances), len(angles), np.max(volume)+1, np.max(volume)+1), dtype=np.uint32)

    for d, distance in enumerate(distances):
        for a, angle in enumerate(angles):
            for x in range(volume.shape[0]):
                for y in range(volume.shape[1]):
                    
                    if (0 <= x + distance*angle[0] < volume.shape[0] and 0 <= y + distance*angle[1] < volume.shape[1]):
                        glcm[d, a, volume[y, x], volume[y + distance*angle[1], x + distance*angle[0]]] += 1
                        # Add reverse direction
                        glcm[d, a, volume[y + distance*angle[1], x + distance*angle[0]], volume[y, x]] += 1

    return glcm

# Example Usage
# Assuming 'volume' is your 3D grayscale image in NumPy array format
# 'volume' should be a 3D array where each element represents a voxel with a grayscale value
# 'distances' is a list of distances you want to consider
# 'angles' is a list of 3D directions in which co-occurrence is calculated
glcm_matrix = compute_2d_glcm_ha(volume)

#%% 3D GLCM Code

import numpy as np

# volume = np.array([[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]])
volume = np.array([[[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                   [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                   [[1, 0, 1], [0, 1, 0], [1, 0, 1]]], dtype=np.uint8)

def compute_3d_glcm(volume, distances=[1], angles=[(1,0,0), (0,1,0), (0,0,1)]):
    glcm = np.zeros((len(distances), len(angles), np.max(volume)+1, np.max(volume)+1), dtype=np.uint32)

    for d, distance in enumerate(distances):
        for a, angle in enumerate(angles):
            for z in range(volume.shape[0]):
                for y in range(volume.shape[1]):
                    for x in range(volume.shape[2]):
                        if (0 <= z + distance*angle[0] < volume.shape[0] and
                            0 <= y + distance*angle[1] < volume.shape[1] and
                            0 <= x + distance*angle[2] < volume.shape[2]):
                            
                            glcm[d, a, volume[z, y, x], volume[z + distance*angle[0], y + distance*angle[1], x + distance*angle[2]]] += 1
                            glcm[d, a, volume[z + distance*angle[0], y + distance*angle[1], x + distance*angle[2]], volume[z, y, x]] += 1
    
    return glcm

# Example Usage
# Assuming 'volume' is your 3D grayscale image in NumPy array format
# 'volume' should be a 3D array where each element represents a voxel with a grayscale value
# 'distances' is a list of distances you want to consider
# 'angles' is a list of 3D directions in which co-occurrence is calculated

angles_inplane  = [(1,0,0), (0,1,0), (1,1,0), (-1,1,0)]
angles_outplane = [(-1,1,1),(0,1,1), (1,1,1), (-1,0,1),(0,0,1), (1,0,1), (-1,-1,1),(0,-1,1),(1,-1,1)]
angle  = angles_inplane + angles_outplane

glcm_matrix = compute_3d_glcm(volume, distances=[1], angles=angle)

### --> After Validation, GLCM3D is confirmed to be correct ###
### From Grey-Level Matarix, we can further compute statistics including GLCM, GLDM and GLZSM

#%% Compute the GLCM Statistics
##### To validate the computation method, 2D GLCM to benchmarking MATLAB #####

volume = np.array([[1,0,1,0],[0,1,0,1],[1,0,1,0],[0,1,0,1]])

def glcm_contrast(glcm_matrix):
    
    glcm_contrast = np.zeros((glcm_matrix.shape[0])) 
    
    for n in range(glcm_matrix.shape[0]):
        for i in range(glcm_matrix.shape[1]):
            for j in range(glcm_matrix.shape[2]):
                
                glcm_contrast[n] += (i-j)**2 * glcm_matrix[n][i][j]
                
    return glcm_contrast

    
def glcm_correlation(glcm_matrix):
    
    glcm_correlation = np.zeros((glcm_matrix.shape[0])) 
    
    for n in range(glcm_matrix.shape[0]):
        
        px = np.sum(glcm_matrix[n], axis=1) / np.sum(glcm_matrix[n])
        py = np.sum(glcm_matrix[n], axis=0) / np.sum(glcm_matrix[n])
        # px = np.sum(glcm_matrix[n], axis=1)
        # py = np.sum(glcm_matrix[n], axis=0)
        
        mu_i = np.sum(np.arange(glcm_matrix[n].shape[0]) * px)
        mu_j = np.sum(np.arange(glcm_matrix[n].shape[1]) * py)
        
        sigma_i = np.sqrt(np.sum((np.arange(glcm_matrix[n].shape[0]) - mu_i)**2 * px))
        sigma_j = np.sqrt(np.sum((np.arange(glcm_matrix[n].shape[1]) - mu_j)**2 * py))
                
        for i in range(glcm_matrix.shape[1]):
            for j in range(glcm_matrix.shape[2]):
                
                # Calculate correlation
                glcm_correlation[n] += glcm_matrix[n][i][j] * (i-mu_i) * (j-mu_j) / sigma_i / sigma_j
    
    return glcm_correlation
    
    
def glcm_energy(glcm_matrix):
        
    glcm_energy = np.zeros((glcm_matrix.shape[0])) 
    
    for n in range(glcm_matrix.shape[0]):
        for i in range(glcm_matrix.shape[1]):
            for j in range(glcm_matrix.shape[2]):
                
                glcm_energy[n] += glcm_matrix[n][i][j]**2
                
    return glcm_energy
       
    
def glcm_homogeneity(glcm_matrix):
       
    glcm_homogeneity = np.zeros((glcm_matrix.shape[0])) 
    
    for n in range(glcm_matrix.shape[0]):
        for i in range(glcm_matrix.shape[1]):
            for j in range(glcm_matrix.shape[2]):
                
                glcm_homogeneity[n] += glcm_matrix[n][i][j] / (1+np.abs(i-j))
                
    return glcm_homogeneity
    

glcm_matrix = compute_2d_glcm_ha(volume)
glcm_matrix = np.squeeze(glcm_matrix)
    
# glcm_matrix[0] = glcm_matrix[0] / np.sum(glcm_matrix[0])

contrast    = glcm_contrast(glcm_matrix)
correlation = glcm_correlation(glcm_matrix)
energy      = glcm_energy(glcm_matrix)
homogeneity = glcm_homogeneity(glcm_matrix)

print('The Contrast List is: ', contrast)
print('The Correlation List is: ', correlation)
print('The Energy List is: ', energy)
print('The Homogeneity List is: ', homogeneity)

#%% Try with package Skimage

glcm_matrix = compute_2d_glcm_ha(volume)
glcm_matrix = np.squeeze(glcm_matrix)
glcm_matrix = np.transpose(glcm_matrix, (1,2,0))
glcm_matrix = glcm_matrix[:, :, np.newaxis, :]

contra = []
ener   = []
corre  = []
homo   = []

contra.append(graycoprops(glcm_matrix, 'contrast'))
corre.append(graycoprops(glcm_matrix, 'correlation'))
ener.append(graycoprops(glcm_matrix, 'energy'))
homo.append(graycoprops(glcm_matrix, 'homogeneity'))

#%% Extract the tumor ROI (stated by paper) Since just tumor, so all use p2

# Apply tumor ROI to different files

pet_A1_roi = pet_A1 * t1_p2_mask_A
pet_A2_roi = pet_A2 * t2_p2_mask_A
pet_A3_roi = pet_A3 * t3_p2_mask_A
pet_B1_roi = pet_B1 * t1_p2_mask_B
pet_B2_roi = pet_B2 * t2_p2_mask_B
pet_B3_roi = pet_B3 * t3_p2_mask_B

dce_A1_roi = dce_A1 * t1_p2_mask_A
dce_A2_roi = dce_A2 * t2_p2_mask_A
dce_A3_roi = dce_A3 * t3_p2_mask_A
dce_B1_roi = dce_B1 * t1_p2_mask_B
dce_B2_roi = dce_B3 * t2_p2_mask_B
dce_B3_roi = dce_B3 * t3_p2_mask_B

adc_A1_roi = adc_A1 * t1_p2_mask_A
adc_A2_roi = adc_A2 * t2_p2_mask_A
adc_A3_roi = adc_A3 * t3_p2_mask_A
adc_B1_roi = adc_B1 * t1_p2_mask_B
adc_B2_roi = adc_B3 * t2_p2_mask_B
adc_B3_roi = adc_B3 * t3_p2_mask_B

#%% Let's visualize the images on slices

#%%
# Plot the ROI images
# for i in range(pet_B3_roi.shape[-1]):
#     plt.figure()
#     plt.imshow(pet_B3_roi[0,:,:,i])
#     plt.title('PET ROI Image at slice: ' + str(i))
#     plt.colorbar()
    
# #%%
# for i in range(pet_A1.shape[-1]):
#     plt.figure()
#     plt.imshow(pet_A1[0,:,:,i])
#     plt.title('PET Image at slice: ' + str(i))
#     plt.colorbar()

# #%%
# # Plot the ROI images
# for i in range(dce_A3_roi.shape[-1]):
#     plt.figure()
#     plt.imshow(dce_A3_roi[0,:,:,i])
#     plt.title('DCE ROI Image at slice: ' + str(i))
#     plt.colorbar()
    
# #%%
# for i in range(dce_A1.shape[-1]):
#     plt.figure()
#     plt.imshow(dce_A1[0,:,:,i])
#     plt.title('DCE Image at slice: ' + str(i))
#     plt.colorbar()

# #%%
# # Plot the ROI images
# for i in range(adc_A3_roi.shape[-1]):
#     plt.figure()
#     plt.imshow(adc_A3_roi[0,:,:,i])
#     plt.title('ADC ROI Image at slice: ' + str(i))
#     plt.colorbar()
    
# #%%
# for i in range(adc_A1.shape[-1]):
#     plt.figure()
#     plt.imshow(adc_A1[0,:,:,i])
#     plt.title('ADC Image at slice: ' + str(i))
#     plt.colorbar()
   

#%% Move on to extract GLCM from the medical ROI images
#################################################
# Step 0: Data Preprocessing
# Step 1: Extract GLCM 2D features from slices
# Step 2: Extract GLCM 3D features from slices
#################################################
#################################################

#%% Take out all valid ROI images (Remove A1 A5 B9 B10)

# Double check everything again

def sanity_check(matrix):
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[-1]):
            
            if np.max(matrix[i,:,:,j]) == 0:
                print('Abnormalty happens at Patient: ' + str(i) + ' and on slice: ' + str(j))
                
# Check PET                
print('Working on A1')    
sanity_check(pet_A1)
print('Working on A2')    
sanity_check(pet_A2)
print('Working on A3')    
sanity_check(pet_A3)
print('Working on B1')    
sanity_check(pet_B1)
print('Working on B2')    
sanity_check(pet_B2)
print('Working on B3')    
sanity_check(pet_B3)

# Check ADC
print('Working on A1')    
sanity_check(adc_A1)
print('Working on A2')    
sanity_check(adc_A2)
print('Working on A3')    
sanity_check(adc_A3)
print('Working on B1')    
sanity_check(adc_B1)
print('Working on B2')    
sanity_check(adc_B2)
print('Working on B3')    
sanity_check(adc_B3)

# Check DCE
print('Working on A1')    
sanity_check(dce_A1)
print('Working on A2')    
sanity_check(dce_A2)
print('Working on A3')    
sanity_check(dce_A3)
print('Working on B1')    
sanity_check(dce_B1)
print('Working on B2')    
sanity_check(dce_B2)
print('Working on B3')    
sanity_check(dce_B3)

#%% Step 0: Preprocessing -- Get the proper matrix

# Note: Exclude A1 A5 B9 B10 for all ROI matrices

pet_A1_roi_new = np.delete(pet_A1_roi, [5],  axis=0)
pet_A2_roi_new = np.delete(pet_A2_roi, [5],  axis=0)
pet_A3_roi_new = np.delete(pet_A3_roi, [5],  axis=0)
pet_B1_roi_new = np.delete(pet_B1_roi, [9,10], axis=0)
pet_B2_roi_new = np.delete(pet_B2_roi, [9,10], axis=0)
pet_B3_roi_new = np.delete(pet_B3_roi, [9,10], axis=0)

adc_A1_roi_new = np.delete(adc_A1_roi, [5],  axis=0)
adc_A2_roi_new = np.delete(adc_A2_roi, [5],  axis=0)
adc_A3_roi_new = np.delete(adc_A3_roi, [5],  axis=0)
adc_B1_roi_new = np.delete(adc_B1_roi, [9,10], axis=0)
adc_B2_roi_new = np.delete(adc_B2_roi, [9,10], axis=0)
adc_B3_roi_new = np.delete(adc_B3_roi, [9,10], axis=0)

dce_A1_roi_new = np.delete(dce_A1_roi, [5],  axis=0)
dce_A2_roi_new = np.delete(dce_A2_roi, [5],  axis=0)
dce_A3_roi_new = np.delete(dce_A3_roi, [5],  axis=0)
dce_B1_roi_new = np.delete(dce_B1_roi, [9,10], axis=0)
dce_B2_roi_new = np.delete(dce_B2_roi, [9,10], axis=0)
dce_B3_roi_new = np.delete(dce_B3_roi, [9,10], axis=0)

#%%
pet_A1_roi_new = pet_A1_roi_new.astype(np.int32)
pet_A2_roi_new = pet_A2_roi_new.astype(np.int32)
pet_A3_roi_new = pet_A3_roi_new.astype(np.int32)
pet_B1_roi_new = pet_B1_roi_new.astype(np.int32)
pet_B2_roi_new = pet_B2_roi_new.astype(np.int32)
pet_B3_roi_new = pet_B3_roi_new.astype(np.int32)

adc_A1_roi_new = adc_A1_roi_new.astype(np.int32)
adc_A2_roi_new = adc_A2_roi_new.astype(np.int32)
adc_A3_roi_new = adc_A3_roi_new.astype(np.int32)
adc_B1_roi_new = adc_B1_roi_new.astype(np.int32)
adc_B2_roi_new = adc_B2_roi_new.astype(np.int32)
adc_B3_roi_new = adc_B3_roi_new.astype(np.int32)

dce_A1_roi_new = dce_A1_roi_new.astype(np.int32)
dce_A2_roi_new = dce_A2_roi_new.astype(np.int32)
dce_A3_roi_new = dce_A3_roi_new.astype(np.int32)
dce_B1_roi_new = dce_B1_roi_new.astype(np.int32)
dce_B2_roi_new = dce_B2_roi_new.astype(np.int32)
dce_B3_roi_new = dce_B3_roi_new.astype(np.int32)

#%% Step 0.5-1: Value to Ratio through all samples

def max_num_ratio(matrix):
    
    ratio = []
    
    for j in range(matrix.shape[-1]):
        max_val = np.max(matrix[:,:,j])
        
        if max_val > 0:
            non_val = np.count_nonzero(matrix[:,:,j])
            ratio.append(max_val / non_val)
        
    ratio_max = np.max(ratio)
    ratio_min = np.min(ratio)
    
    return ratio_max, ratio_min

# Apply for different images

pet_ratio_max = []
pet_ratio_min = []

for i in range(pet_A1_roi_new.shape[0]):  
    max_val, min_val = max_num_ratio(pet_A1_roi_new[i,:,:,:])
    pet_ratio_max.append(max_val)
    pet_ratio_min.append(min_val)

for i in range(pet_A2_roi_new.shape[0]):  
    max_val, min_val = max_num_ratio(pet_A2_roi_new[i,:,:,:])
    pet_ratio_max.append(max_val)
    pet_ratio_min.append(min_val)
    
for i in range(pet_A3_roi_new.shape[0]):  
    max_val, min_val = max_num_ratio(pet_A3_roi_new[i,:,:,:])
    pet_ratio_max.append(max_val)
    pet_ratio_min.append(min_val)

### Conclusion: This code has shown that that range of images could from 1 to 10000, making GLCM super large.
###             So some normalization is needed for extracting GLCM

#%% Step 0.5-2: Matrix Quantization

def matrix_norm(matrix, num_levels):
    
    if np.min(matrix) < 0:
        matrix = matrix - np.min(matrix)
    
    quantized_mtx = np.floor_divide(matrix, np.ceil(np.max(matrix) / num_levels))
    
    return quantized_mtx

num_levels = 256

#%%
################################# PET #########################################
m, n, p, q = pet_A1_roi_new.shape 
pet_A1_roi_norm = np.zeros((m,n,p,q))
for i in range(m):
    pet_A1_roi_norm[i,:,:,:] = matrix_norm(pet_A1_roi_new[i,:,:,:], num_levels)
    
m, n, p, q = pet_A2_roi_new.shape 
pet_A2_roi_norm = np.zeros((m,n,p,q))
for i in range(m):
    pet_A2_roi_norm[i,:,:,:] = matrix_norm(pet_A2_roi_new[i,:,:,:], num_levels)
    
m, n, p, q = pet_A3_roi_new.shape 
pet_A3_roi_norm = np.zeros((m,n,p,q))
for i in range(m):
    pet_A3_roi_norm[i,:,:,:] = matrix_norm(pet_A3_roi_new[i,:,:,:], num_levels)
    
m, n, p, q = pet_B1_roi_new.shape 
pet_B1_roi_norm = np.zeros((m,n,p,q))
for i in range(m):
    pet_B1_roi_norm[i,:,:,:] = matrix_norm(pet_B1_roi_new[i,:,:,:], num_levels)
    
m, n, p, q = pet_B2_roi_new.shape 
pet_B2_roi_norm = np.zeros((m,n,p,q))
for i in range(m):
    pet_B2_roi_norm[i,:,:,:] = matrix_norm(pet_B2_roi_new[i,:,:,:], num_levels)
    
m, n, p, q = pet_B3_roi_new.shape 
pet_B3_roi_norm = np.zeros((m,n,p,q))
for i in range(m):
    pet_B3_roi_norm[i,:,:,:] = matrix_norm(pet_B3_roi_new[i,:,:,:], num_levels)
    
#%%    
################################# DCE #########################################
m, n, p, q = dce_A1_roi_new.shape 
dce_A1_roi_norm = np.zeros((m,n,p,q))
for i in range(m):
    dce_A1_roi_norm[i,:,:,:] = matrix_norm(dce_A1_roi_new[i,:,:,:], num_levels)
    
m, n, p, q = dce_A2_roi_new.shape 
dce_A2_roi_norm = np.zeros((m,n,p,q))
for i in range(m):
    dce_A2_roi_norm[i,:,:,:] = matrix_norm(dce_A2_roi_new[i,:,:,:], num_levels)
    
m, n, p, q = dce_A3_roi_new.shape 
dce_A3_roi_norm = np.zeros((m,n,p,q))
for i in range(m):
    dce_A3_roi_norm[i,:,:,:] = matrix_norm(dce_A3_roi_new[i,:,:,:], num_levels)
    
m, n, p, q = dce_B1_roi_new.shape 
dce_B1_roi_norm = np.zeros((m,n,p,q))
for i in range(m):
    dce_B1_roi_norm[i,:,:,:] = matrix_norm(dce_B1_roi_new[i,:,:,:], num_levels)
    
m, n, p, q = dce_B2_roi_new.shape 
dce_B2_roi_norm = np.zeros((m,n,p,q))
for i in range(m):
    dce_B2_roi_norm[i,:,:,:] = matrix_norm(dce_B2_roi_new[i,:,:,:], num_levels)
    
m, n, p, q = dce_B3_roi_new.shape 
dce_B3_roi_norm = np.zeros((m,n,p,q))
for i in range(m):
    dce_B3_roi_norm[i,:,:,:] = matrix_norm(dce_B3_roi_new[i,:,:,:], num_levels)
#%%    
################################# ADC #########################################
m, n, p, q = adc_A1_roi_new.shape 
adc_A1_roi_norm = np.zeros((m,n,p,q))
for i in range(m):
    adc_A1_roi_norm[i,:,:,:] = matrix_norm(adc_A1_roi_new[i,:,:,:], num_levels)
    
m, n, p, q = adc_A2_roi_new.shape 
adc_A2_roi_norm = np.zeros((m,n,p,q))
for i in range(m):
    adc_A2_roi_norm[i,:,:,:] = matrix_norm(adc_A2_roi_new[i,:,:,:], num_levels)
    
m, n, p, q = adc_A3_roi_new.shape 
adc_A3_roi_norm = np.zeros((m,n,p,q))
for i in range(m):
    adc_A3_roi_norm[i,:,:,:] = matrix_norm(adc_A3_roi_new[i,:,:,:], num_levels)
    
m, n, p, q = adc_B1_roi_new.shape 
adc_B1_roi_norm = np.zeros((m,n,p,q))
for i in range(m):
    adc_B1_roi_norm[i,:,:,:] = matrix_norm(adc_B1_roi_new[i,:,:,:], num_levels)
    
m, n, p, q = adc_B2_roi_new.shape 
adc_B2_roi_norm = np.zeros((m,n,p,q))
for i in range(m):
    adc_B2_roi_norm[i,:,:,:] = matrix_norm(adc_B2_roi_new[i,:,:,:], num_levels)
    
m, n, p, q = adc_B3_roi_new.shape 
adc_B3_roi_norm = np.zeros((m,n,p,q))
for i in range(m):
    adc_B3_roi_norm[i,:,:,:] = matrix_norm(adc_B3_roi_new[i,:,:,:], num_levels)

#%% Renormalize the parameters

pet_A1_roi_norm = pet_A1_roi_norm.astype(np.int32)
pet_A2_roi_norm = pet_A2_roi_norm.astype(np.int32)
pet_A3_roi_norm = pet_A3_roi_norm.astype(np.int32)
pet_B1_roi_norm = pet_B1_roi_norm.astype(np.int32)
pet_B2_roi_norm = pet_B2_roi_norm.astype(np.int32)
pet_B3_roi_norm = pet_B3_roi_norm.astype(np.int32)

adc_A1_roi_norm = adc_A1_roi_norm.astype(np.int32)
adc_A2_roi_norm = adc_A2_roi_norm.astype(np.int32)
adc_A3_roi_norm = adc_A3_roi_norm.astype(np.int32)
adc_B1_roi_norm = adc_B1_roi_norm.astype(np.int32)
adc_B2_roi_norm = adc_B2_roi_norm.astype(np.int32)
adc_B3_roi_norm = adc_B3_roi_norm.astype(np.int32)

dce_A1_roi_norm = dce_A1_roi_norm.astype(np.int32)
dce_A2_roi_norm = dce_A2_roi_norm.astype(np.int32)
dce_A3_roi_norm = dce_A3_roi_norm.astype(np.int32)
dce_B1_roi_norm = dce_B1_roi_norm.astype(np.int32)
dce_B2_roi_norm = dce_B2_roi_norm.astype(np.int32)
dce_B3_roi_norm = dce_B3_roi_norm.astype(np.int32)
    
#%% Now validate the normalization result for comparison

# for i in range(adc_A1_roi_new.shape[-1]):
#     plt.figure()
#     plt.imshow(adc_A1_roi_new[2,:,:,i])
#     plt.title('ADC ROI MASK Image at slice: ' + str(i))
#     plt.colorbar()
    
# for i in range(adc_A1_roi_norm.shape[-1]):
#     plt.figure()
#     plt.imshow(adc_A1_roi_norm[2,:,:,i])
#     plt.title('ADC ROI NORM Image at slice: ' + str(i))
#     plt.colorbar()

#%% Step 1: extract 2D GLCMs from slices (will stack GLCM features)

### PET Images ###

# Define function for extracting 2D GLCM for single patient

def glcm_stats_skimage(glcm_matrix):
    
    glcm_matrix = np.squeeze(glcm_matrix)
    glcm_matrix = np.transpose(glcm_matrix, (1,2,0))
    glcm_matrix = glcm_matrix[:, :, np.newaxis, :]

    contra = []
    ener   = []
    corre  = []
    homo   = []

    contra.append(graycoprops(glcm_matrix, 'contrast'))
    corre.append(graycoprops(glcm_matrix,  'correlation'))
    ener.append(graycoprops(glcm_matrix,   'energy'))
    homo.append(graycoprops(glcm_matrix,   'homogeneity'))
    
    return contra, corre, ener, homo

def glcm_slice_extract(matrix, pid):
    
    glcm_matrix_2d = []
    valid_id       = []
    contrast       = []
    correlation    = []
    energy         = []
    homogeneity    = []

    for i in range(matrix[pid].shape[-1]):
        
        # Check to get rid of only 0->0 transitions
        if np.max(matrix[pid,:,:,i]) > 0:
            print(matrix[pid,:,:,i].shape)
            
            glcm_temp = compute_2d_glcm_ha(matrix[pid,:,:,i])
        
            glcm_matrix_2d.append(glcm_temp)
            print(glcm_temp.shape)
            valid_id.append(i)    
            
            contra, corre, ener, homo = glcm_stats_skimage(glcm_temp)
            
            contrast.append(contra)
            correlation.append(corre)
            energy.append(ener)
            homogeneity.append(homo)
            
    return contrast, correlation, energy, homogeneity, glcm_matrix_2d, valid_id

def glcm_slice_extract_test(matrix, pid):
    
    glcm_matrix_2d = []
    valid_id       = []
    contrast       = []
    correlation    = []
    energy         = []
    homogeneity    = []
    original_matrix = []

    for i in range(matrix[pid].shape[-1]):
        
        # Check to get rid of only 0->0 transitions
        if np.max(matrix[pid,:,:,i]) > 0:
            print('Patient ID: ', pid)
            print('Nonzero works work on slice:', i)
        
            glcm_temp = compute_2d_glcm_ha(matrix[pid,:,:,i])
        
            glcm_matrix_2d.append(glcm_temp)
            valid_id.append(i)    
            original_matrix.append(matrix[pid,:,:,i])
            
            contra, corre, ener, homo = glcm_stats_skimage(glcm_temp)
            
            contrast.append(contra)
            correlation.append(corre)
            energy.append(ener)
            homogeneity.append(homo)
            
    return contrast, correlation, energy, homogeneity, original_matrix, glcm_matrix_2d, valid_id

#%% Get for PET Images

### Get for PET A1
pet_A1_contrast    = []
pet_A1_correlation = []
pet_A1_energy      = []
pet_A1_homogeneity = []

for pid in range(pet_A1_roi_new.shape[0]):
    
    contr, corre, ener, homo, ori_mat, glcm_mat, val = glcm_slice_extract_test(pet_A1_roi_norm, pid)
    pet_A1_contrast.append(contr)
    pet_A1_correlation.append(corre)
    pet_A1_energy.append(ener)
    pet_A1_homogeneity.append(homo)

#%% Get for PET A2
pet_A2_contrast    = []
pet_A2_correlation = []
pet_A2_energy      = []
pet_A2_homogeneity = []

for pid in range(pet_A2_roi_new.shape[0]):
    
    contr, corre, ener, homo, mat, val = glcm_slice_extract(pet_A2_roi_norm, pid)
    pet_A2_contrast.append(contr)
    pet_A2_correlation.append(corre)
    pet_A2_energy.append(ener)
    pet_A2_homogeneity.append(homo)

#%% Get for PET A3
pet_A3_contrast    = []
pet_A3_correlation = []
pet_A3_energy      = []
pet_A3_homogeneity = []

for pid in range(pet_A3_roi_new.shape[0]):
    
    contr, corre, ener, homo, mat, val = glcm_slice_extract(pet_A3_roi_norm, pid)
    pet_A3_contrast.append(contr)
    pet_A3_correlation.append(corre)
    pet_A3_energy.append(ener)
    pet_A3_homogeneity.append(homo)

#%% Get for PET B1
pet_B1_contrast    = []
pet_B1_correlation = []
pet_B1_energy      = []
pet_B1_homogeneity = []

for pid in range(pet_B1_roi_new.shape[0]):
    
    contr, corre, ener, homo, mat, val = glcm_slice_extract(pet_B1_roi_norm, pid)
    pet_B1_contrast.append(contr)
    pet_B1_correlation.append(corre)
    pet_B1_energy.append(ener)
    pet_B1_homogeneity.append(homo)

#%% Get for PET B2
pet_B2_contrast    = []
pet_B2_correlation = []
pet_B2_energy      = []
pet_B2_homogeneity = []

for pid in range(pet_B2_roi_new.shape[0]):
    
    contr, corre, ener, homo, mat, val = glcm_slice_extract(pet_B2_roi_norm, pid)
    pet_B2_contrast.append(contr)
    pet_B2_correlation.append(corre)
    pet_B2_energy.append(ener)
    pet_B2_homogeneity.append(homo)

#%% Get for PET B3
pet_B3_contrast    = []
pet_B3_correlation = []
pet_B3_energy      = []
pet_B3_homogeneity = []

for pid in range(pet_B3_roi_new.shape[0]):
    
    contr, corre, ener, homo, mat, val = glcm_slice_extract(pet_B3_roi_norm, pid)
    pet_B3_contrast.append(contr)
    pet_B3_correlation.append(corre)
    pet_B3_energy.append(ener)
    pet_B3_homogeneity.append(homo)


#%% Get for DCE Images

### Get for DCE A1
dce_A1_contrast    = []
dce_A1_correlation = []
dce_A1_energy      = []
dce_A1_homogeneity = []

for pid in range(dce_A1_roi_new.shape[0]):
    
    contr, corre, ener, homo, mat, val = glcm_slice_extract(dce_A1_roi_norm, pid)
    dce_A1_contrast.append(contr)
    dce_A1_correlation.append(corre)
    dce_A1_energy.append(ener)
    dce_A1_homogeneity.append(homo)

### Get for DCE A2
dce_A2_contrast    = []
dce_A2_correlation = []
dce_A2_energy      = []
dce_A2_homogeneity = []

for pid in range(pet_A2_roi_new.shape[0]):
    
    contr, corre, ener, homo, mat, val = glcm_slice_extract(dce_A2_roi_norm, pid)
    dce_A2_contrast.append(contr)
    dce_A2_correlation.append(corre)
    dce_A2_energy.append(ener)
    dce_A2_homogeneity.append(homo)

### Get for DCE A3
dce_A3_contrast    = []
dce_A3_correlation = []
dce_A3_energy      = []
dce_A3_homogeneity = []

for pid in range(dce_A3_roi_new.shape[0]):
    
    contr, corre, ener, homo, mat, val = glcm_slice_extract(dce_A3_roi_norm, pid)
    dce_A3_contrast.append(contr)
    dce_A3_correlation.append(corre)
    dce_A3_energy.append(ener)
    dce_A3_homogeneity.append(homo)

### Get for DCE B1
dce_B1_contrast    = []
dce_B1_correlation = []
dce_B1_energy      = []
dce_B1_homogeneity = []

for pid in range(dce_B1_roi_new.shape[0]):
    
    contr, corre, ener, homo, mat, val = glcm_slice_extract(dce_B1_roi_norm, pid)
    dce_B1_contrast.append(contr)
    dce_B1_correlation.append(corre)
    dce_B1_energy.append(ener)
    dce_B1_homogeneity.append(homo)

### Get for DCE B2
dce_B2_contrast    = []
dce_B2_correlation = []
dce_B2_energy      = []
dce_B2_homogeneity = []

for pid in range(dce_B2_roi_new.shape[0]):
    
    contr, corre, ener, homo, mat, val = glcm_slice_extract(dce_B2_roi_norm, pid)
    dce_B2_contrast.append(contr)
    dce_B2_correlation.append(corre)
    dce_B2_energy.append(ener)
    dce_B2_homogeneity.append(homo)

### Get for DCE B3
dce_B3_contrast    = []
dce_B3_correlation = []
dce_B3_energy      = []
dce_B3_homogeneity = []

for pid in range(dce_B3_roi_new.shape[0]):
    
    contr, corre, ener, homo, mat, val = glcm_slice_extract(dce_B3_roi_norm, pid)
    dce_B3_contrast.append(contr)
    dce_B3_correlation.append(corre)
    dce_B3_energy.append(ener)
    dce_B3_homogeneity.append(homo)


#%% Get for ADC Images

### Get for ADC A1
adc_A1_contrast    = []
adc_A1_correlation = []
adc_A1_energy      = []
adc_A1_homogeneity = []

for pid in range(adc_A1_roi_new.shape[0]):
    
    contr, corre, ener, homo, mat, val = glcm_slice_extract(adc_A1_roi_norm, pid)
    adc_A1_contrast.append(contr)
    adc_A1_correlation.append(corre)
    adc_A1_energy.append(ener)
    adc_A1_homogeneity.append(homo)

### Get for ADC A2
adc_A2_contrast    = []
adc_A2_correlation = []
adc_A2_energy      = []
adc_A2_homogeneity = []

for pid in range(adc_A2_roi_new.shape[0]):
    
    contr, corre, ener, homo, mat, val = glcm_slice_extract(adc_A2_roi_norm, pid)
    adc_A2_contrast.append(contr)
    adc_A2_correlation.append(corre)
    adc_A2_energy.append(ener)
    adc_A2_homogeneity.append(homo)

### Get for ADC A3
adc_A3_contrast    = []
adc_A3_correlation = []
adc_A3_energy      = []
adc_A3_homogeneity = []

for pid in range(adc_A3_roi_new.shape[0]):
    
    contr, corre, ener, homo, mat, val = glcm_slice_extract(adc_A3_roi_norm, pid)
    adc_A3_contrast.append(contr)
    adc_A3_correlation.append(corre)
    adc_A3_energy.append(ener)
    adc_A3_homogeneity.append(homo)

### Get for ADC B1
adc_B1_contrast    = []
adc_B1_correlation = []
adc_B1_energy      = []
adc_B1_homogeneity = []

for pid in range(adc_B1_roi_new.shape[0]):
    
    contr, corre, ener, homo, mat, val = glcm_slice_extract(adc_B1_roi_norm, pid)
    adc_B1_contrast.append(contr)
    adc_B1_correlation.append(corre)
    adc_B1_energy.append(ener)
    adc_B1_homogeneity.append(homo)

### Get for ADC B2
adc_B2_contrast    = []
adc_B2_correlation = []
adc_B2_energy      = []
adc_B2_homogeneity = []

for pid in range(adc_B2_roi_new.shape[0]):
    
    contr, corre, ener, homo, mat, val = glcm_slice_extract(adc_B2_roi_norm, pid)
    adc_B2_contrast.append(contr)
    adc_B2_correlation.append(corre)
    adc_B2_energy.append(ener)
    adc_B2_homogeneity.append(homo)

### Get for ADC B3
adc_B3_contrast    = []
adc_B3_correlation = []
adc_B3_energy      = []
adc_B3_homogeneity = []

for pid in range(adc_B3_roi_new.shape[0]):
    
    contr, corre, ener, homo, mat, val = glcm_slice_extract(adc_B3_roi_norm, pid)
    adc_B3_contrast.append(contr)
    adc_B3_correlation.append(corre)
    adc_B3_energy.append(ener)
    adc_B3_homogeneity.append(homo)

#%% Save all the 2D GLCM features
adc_T1_contrast    = adc_A1_contrast    + adc_B1_contrast
adc_T2_contrast    = adc_A2_contrast    + adc_B2_contrast
adc_T3_contrast    = adc_A3_contrast    + adc_B3_contrast
adc_T1_correlation = adc_A1_correlation + adc_B1_correlation
adc_T2_correlation = adc_A2_correlation + adc_B2_correlation
adc_T3_correlation = adc_A3_correlation + adc_B3_correlation
adc_T1_energy      = adc_A1_energy      + adc_B1_energy
adc_T2_energy      = adc_A2_energy      + adc_B2_energy
adc_T3_energy      = adc_A3_energy      + adc_B3_energy
adc_T1_homogeneity = adc_A1_homogeneity + adc_B1_homogeneity
adc_T2_homogeneity = adc_A2_homogeneity + adc_B2_homogeneity
adc_T3_homogeneity = adc_A3_homogeneity + adc_B3_homogeneity

dce_T1_contrast    = dce_A1_contrast    + dce_B1_contrast
dce_T2_contrast    = dce_A2_contrast    + dce_B2_contrast
dce_T3_contrast    = dce_A3_contrast    + dce_B3_contrast
dce_T1_correlation = dce_A1_correlation + dce_B1_correlation
dce_T2_correlation = dce_A2_correlation + dce_B2_correlation
dce_T3_correlation = dce_A3_correlation + dce_B3_correlation
dce_T1_energy      = dce_A1_energy      + dce_B1_energy
dce_T2_energy      = dce_A2_energy      + dce_B2_energy
dce_T3_energy      = dce_A3_energy      + dce_B3_energy
dce_T1_homogeneity = dce_A1_homogeneity + dce_B1_homogeneity
dce_T2_homogeneity = dce_A2_homogeneity + dce_B2_homogeneity
dce_T3_homogeneity = dce_A3_homogeneity + dce_B3_homogeneity

pet_T1_contrast    = pet_A1_contrast    + pet_B1_contrast
pet_T2_contrast    = pet_A2_contrast    + pet_B2_contrast
pet_T3_contrast    = pet_A3_contrast    + pet_B3_contrast
pet_T1_correlation = pet_A1_correlation + pet_B1_correlation
pet_T2_correlation = pet_A2_correlation + pet_B2_correlation
pet_T3_correlation = pet_A3_correlation + pet_B3_correlation
pet_T1_energy      = pet_A1_energy      + pet_B1_energy
pet_T2_energy      = pet_A2_energy      + pet_B2_energy
pet_T3_energy      = pet_A3_energy      + pet_B3_energy
pet_T1_homogeneity = pet_A1_homogeneity + pet_B1_homogeneity
pet_T2_homogeneity = pet_A2_homogeneity + pet_B2_homogeneity
pet_T3_homogeneity = pet_A3_homogeneity + pet_B3_homogeneity

#%% Now save all the 2D stats

with open('adc_T1_contrast_di_new.pkl', 'wb') as file:
    pickle.dump(adc_T1_contrast, file)
with open('adc_T2_contrast_di_new.pkl', 'wb') as file:
    pickle.dump(adc_T2_contrast, file)
with open('adc_T3_contrast_di_new.pkl', 'wb') as file:
    pickle.dump(adc_T3_contrast, file)
with open('adc_T1_correlation_di_new.pkl', 'wb') as file:
    pickle.dump(adc_T1_correlation, file)
with open('adc_T2_correlation_di_new.pkl', 'wb') as file:
    pickle.dump(adc_T2_correlation, file)
with open('adc_T3_correlation_di_new.pkl', 'wb') as file:
    pickle.dump(adc_T3_correlation, file)
with open('adc_T1_energy_di_new.pkl', 'wb') as file:
    pickle.dump(adc_T1_energy, file)
with open('adc_T2_energy_di_new.pkl', 'wb') as file:
    pickle.dump(adc_T2_energy, file)
with open('adc_T3_energy_di_new.pkl', 'wb') as file:
    pickle.dump(adc_T3_energy, file)
with open('adc_T1_homogeneity_di_new.pkl', 'wb') as file:
    pickle.dump(adc_T1_homogeneity, file)
with open('adc_T2_homogeneity_di_new.pkl', 'wb') as file:
    pickle.dump(adc_T2_homogeneity, file)
with open('adc_T3_homogeneity_di_new.pkl', 'wb') as file:
    pickle.dump(adc_T3_homogeneity, file)
    
    
with open('dce_T1_contrast_di_new.pkl', 'wb') as file:
    pickle.dump(dce_T1_contrast, file)
with open('dce_T2_contrast_di_new.pkl', 'wb') as file:
    pickle.dump(dce_T2_contrast, file)
with open('dce_T3_contrast_di_new.pkl', 'wb') as file:
    pickle.dump(dce_T3_contrast, file)
with open('dce_T1_correlation_di_new.pkl', 'wb') as file:
    pickle.dump(dce_T1_correlation, file)
with open('dce_T2_correlation_di_new.pkl', 'wb') as file:
    pickle.dump(dce_T2_correlation, file)
with open('dce_T3_correlation_di_new.pkl', 'wb') as file:
    pickle.dump(dce_T3_correlation, file)
with open('dce_T1_energy_di_new.pkl', 'wb') as file:
    pickle.dump(dce_T1_energy, file)
with open('dce_T2_energy_di_new.pkl', 'wb') as file:
    pickle.dump(dce_T2_energy, file)
with open('dce_T3_energy_di_new.pkl', 'wb') as file:
    pickle.dump(dce_T3_energy, file)
with open('dce_T1_homogeneity_di_new.pkl', 'wb') as file:
    pickle.dump(dce_T1_homogeneity, file)
with open('dce_T2_homogeneity_di_new.pkl', 'wb') as file:
    pickle.dump(dce_T2_homogeneity, file)
with open('dce_T3_homogeneity_di_new.pkl', 'wb') as file:
    pickle.dump(dce_T3_homogeneity, file)  
    
     
with open('pet_T1_contrast_di_new.pkl', 'wb') as file:
    pickle.dump(pet_T1_contrast, file)
with open('pet_T2_contrast_di_new.pkl', 'wb') as file:
    pickle.dump(pet_T2_contrast, file)
with open('pet_T3_contrast_di_new.pkl', 'wb') as file:
    pickle.dump(pet_T3_contrast, file)
with open('pet_T1_correlation_di_new.pkl', 'wb') as file:
    pickle.dump(pet_T1_correlation, file)
with open('pet_T2_correlation_di_new.pkl', 'wb') as file:
    pickle.dump(pet_T2_correlation, file)
with open('pet_T3_correlation_di_new.pkl', 'wb') as file:
    pickle.dump(pet_T3_correlation, file)
with open('pet_T1_energy_di_new.pkl', 'wb') as file:
    pickle.dump(pet_T1_energy, file)
with open('pet_T2_energy_di_new.pkl', 'wb') as file:
    pickle.dump(pet_T2_energy, file)
with open('pet_T3_energy_di_new.pkl', 'wb') as file:
    pickle.dump(pet_T3_energy, file)
with open('pet_T1_homogeneity_di_new.pkl', 'wb') as file:
    pickle.dump(pet_T1_homogeneity, file)
with open('pet_T2_homogeneity_di_new.pkl', 'wb') as file:
    pickle.dump(pet_T2_homogeneity, file)
with open('pet_T3_homogeneity_di_new.pkl', 'wb') as file:
    pickle.dump(pet_T3_homogeneity, file)
        
    
# with open('adc_T1_contrast.pkl', 'rb') as file:
#     load_list = pickle.load(file)
    
#%% Step 2: extract 3D GLCMs from the 3D matrix directly

### PET Images ###

# glcm_matrix = compute_3d_glcm(pet_A1_roi_new[0], distances=[1], angles=angle)

# Define function for extracting 2D GLCM for single patient

# def glcm_stats_skimage(glcm_matrix):
    
#     glcm_matrix = np.squeeze(glcm_matrix)
#     glcm_matrix = np.transpose(glcm_matrix, (1,2,0))
#     glcm_matrix = glcm_matrix[:, :, np.newaxis, :]

#     contra = []
#     ener   = []
#     corre  = []
#     homo   = []

#     contra.append(graycoprops(glcm_matrix, 'contrast'))
#     corre.append(graycoprops(glcm_matrix, 'correlation'))
#     ener.append(graycoprops(glcm_matrix, 'energy'))
#     homo.append(graycoprops(glcm_matrix, 'homogeneity'))
    
#     return contra, corre, ener, homo

# def glcm_slice_extract_3d(matrix, pid):
    
#     glcm_matrix_3d = []
#     contrast       = []
#     correlation    = []
#     energy         = []
#     homogeneity    = []

#     for i in range(matrix[pid].shape[-1]):
        
#         # Check to get rid of only 0->0 transitions
#         if np.max(matrix[pid,:,:,i]) > 0:
#             print('Currently working on valid slice: ', i)
        
#             print('Generating the glcm')
#             glcm_temp = compute_3d_glcm(matrix[pid,:,:,:], distances=[1], angles=angle)
        
#             glcm_matrix_3d.append(glcm_temp)  
            
#             print('Computing the stats')
#             contra, corre, ener, homo = glcm_stats_skimage(glcm_temp)
            
#             contrast.append(contra)
#             correlation.append(corre)
#             energy.append(ener)
#             homogeneity.append(homo)
             
#     return contrast, correlation, energy, homogeneity, glcm_matrix_3d


# #%% Get for PET Images

# ### Get for PET A1
# pet_A1_contrast_3d    = []
# pet_A1_correlation_3d = []
# pet_A1_energy_3d      = []
# pet_A1_homogeneity_3d = []

# for pid in range(pet_A1_roi_new.shape[0]):
    
#     print('Currently working on Patient: ', pid)
    
#     contr, corre, ener, homo, mat = glcm_slice_extract_3d(pet_A1_roi_norm, pid)
#     pet_A1_contrast_3d.append(contr)
#     pet_A1_correlation_3d.append(corre)
#     pet_A1_energy_3d.append(ener)
#     pet_A1_homogeneity_3d.append(homo)

# #%% Get for PET A2
# pet_A2_contrast_3d    = []
# pet_A2_correlation_3d = []
# pet_A2_energy_3d      = []
# pet_A2_homogeneity_3d = []

# for pid in range(pet_A2_roi_new.shape[0]):
    
#     contr, corre, ener, homo, mat = glcm_slice_extract_3d(pet_A2_roi_norm, pid)
#     pet_A2_contrast_3d.append(contr)
#     pet_A2_correlation_3d.append(corre)
#     pet_A2_energy_3d.append(ener)
#     pet_A2_homogeneity_3d.append(homo)

# ### Get for PET A3
# pet_A3_contrast_3d    = []
# pet_A3_correlation_3d = []
# pet_A3_energy_3d      = []
# pet_A3_homogeneity_3d = []

# for pid in range(pet_A3_roi_new.shape[0]):
    
#     contr, corre, ener, homo, mat = glcm_slice_extract_3d(pet_A3_roi_norm, pid)
#     pet_A3_contrast_3d.append(contr)
#     pet_A3_correlation_3d.append(corre)
#     pet_A3_energy_3d.append(ener)
#     pet_A3_homogeneity_3d.append(homo)

# ### Get for PET B1
# pet_B1_contrast_3d    = []
# pet_B1_correlation_3d = []
# pet_B1_energy_3d      = []
# pet_B1_homogeneity_3d = []

# for pid in range(pet_B1_roi_new.shape[0]):
    
#     contr, corre, ener, homo, mat = glcm_slice_extract_3d(pet_B1_roi_norm, pid)
#     pet_B1_contrast_3d.append(contr)
#     pet_B1_correlation_3d.append(corre)
#     pet_B1_energy_3d.append(ener)
#     pet_B1_homogeneity_3d.append(homo)

# ### Get for PET B2
# pet_B2_contrast_3d    = []
# pet_B2_correlation_3d = []
# pet_B2_energy_3d      = []
# pet_B2_homogeneity_3d = []

# for pid in range(pet_B2_roi_new.shape[0]):
    
#     contr, corre, ener, homo, mat = glcm_slice_extract_3d(pet_B2_roi_norm, pid)
#     pet_B2_contrast_3d.append(contr)
#     pet_B2_correlation_3d.append(corre)
#     pet_B2_energy_3d.append(ener)
#     pet_B2_homogeneity_3d.append(homo)

# ### Get for PET B3
# pet_B3_contrast_3d    = []
# pet_B3_correlation_3d = []
# pet_B3_energy_3d      = []
# pet_B3_homogeneity_3d = []

# for pid in range(pet_B3_roi_new.shape[0]):
    
#     contr, corre, ener, homo, mat = glcm_slice_extract_3d(pet_B3_roi_norm, pid)
#     pet_B3_contrast_3d.append(contr)
#     pet_B3_correlation_3d.append(corre)
#     pet_B3_energy_3d.append(ener)
#     pet_B3_homogeneity_3d.append(homo)



# #%% Get for DCE Images

# ### Get for DCE A1
# dce_A1_contrast_3d    = []
# dce_A1_correlation_3d = []
# dce_A1_energy_3d      = []
# dce_A1_homogeneity_3d = []

# for pid in range(dce_A1_roi_new.shape[0]):
    
#     contr, corre, ener, homo, mat = glcm_slice_extract_3d(dce_A1_roi_norm, pid)
#     dce_A1_contrast_3d.append(contr)
#     dce_A1_correlation_3d.append(corre)
#     dce_A1_energy_3d.append(ener)
#     dce_A1_homogeneity_3d.append(homo)

# ### Get for DCE A2
# dce_A2_contrast_3d    = []
# dce_A2_correlation_3d = []
# dce_A2_energy_3d      = []
# dce_A2_homogeneity_3d = []

# for pid in range(dce_A2_roi_new.shape[0]):
    
#     contr, corre, ener, homo, mat = glcm_slice_extract_3d(dce_A2_roi_norm, pid)
#     dce_A2_contrast_3d.append(contr)
#     dce_A2_correlation_3d.append(corre)
#     dce_A2_energy_3d.append(ener)
#     dce_A2_homogeneity_3d.append(homo)

# ### Get for DCE A3
# dce_A3_contrast_3d    = []
# dce_A3_correlation_3d = []
# dce_A3_energy_3d      = []
# dce_A3_homogeneity_3d = []

# for pid in range(dce_A3_roi_new.shape[0]):
    
#     contr, corre, ener, homo, mat = glcm_slice_extract_3d(dce_A3_roi_norm, pid)
#     dce_A3_contrast_3d.append(contr)
#     dce_A3_correlation_3d.append(corre)
#     dce_A3_energy_3d.append(ener)
#     dce_A3_homogeneity_3d.append(homo)

# ### Get for DCE B1
# dce_B1_contrast_3d    = []
# dce_B1_correlation_3d = []
# dce_B1_energy_3d      = []
# dce_B1_homogeneity_3d = []

# for pid in range(dce_B1_roi_new.shape[0]):
    
#     contr, corre, ener, homo, mat = glcm_slice_extract_3d(dce_B1_roi_norm, pid)
#     dce_B1_contrast_3d.append(contr)
#     dce_B1_correlation_3d.append(corre)
#     dce_B1_energy_3d.append(ener)
#     dce_B1_homogeneity_3d.append(homo)

# ### Get for DCE B2
# dce_B2_contrast_3d    = []
# dce_B2_correlation_3d = []
# dce_B2_energy_3d      = []
# dce_B2_homogeneity_3d = []

# for pid in range(dce_B2_roi_new.shape[0]):
    
#     contr, corre, ener, homo, mat = glcm_slice_extract_3d(dce_B2_roi_norm, pid)
#     dce_B2_contrast_3d.append(contr)
#     dce_B2_correlation_3d.append(corre)
#     dce_B2_energy_3d.append(ener)
#     dce_B2_homogeneity_3d.append(homo)

# ### Get for DCE B3
# dce_B3_contrast_3d    = []
# dce_B3_correlation_3d = []
# dce_B3_energy_3d      = []
# dce_B3_homogeneity_3d = []

# for pid in range(dce_B3_roi_new.shape[0]):
    
#     contr, corre, ener, homo, mat = glcm_slice_extract_3d(dce_B3_roi_norm, pid)
#     dce_B3_contrast_3d.append(contr)
#     dce_B3_correlation_3d.append(corre)
#     dce_B3_energy_3d.append(ener)
#     dce_B3_homogeneity_3d.append(homo)

# #%% Get for ADC Images

# ### Get for ADC A1
# adc_A1_contrast_3d    = []
# adc_A1_correlation_3d = []
# adc_A1_energy_3d      = []
# adc_A1_homogeneity_3d = []

# for pid in range(adc_A1_roi_new.shape[0]):
    
#     contr, corre, ener, homo, mat = glcm_slice_extract_3d(adc_A1_roi_norm, pid)
#     adc_A1_contrast_3d.append(contr)
#     adc_A1_correlation_3d.append(corre)
#     adc_A1_energy_3d.append(ener)
#     adc_A1_homogeneity_3d.append(homo)

# ### Get for ADC A2
# adc_A2_contrast_3d    = []
# adc_A2_correlation_3d = []
# adc_A2_energy_3d      = []
# adc_A2_homogeneity_3d = []

# for pid in range(adc_A2_roi_new.shape[0]):
    
#     contr, corre, ener, homo, mat = glcm_slice_extract_3d(adc_A2_roi_norm, pid)
#     adc_A2_contrast_3d.append(contr)
#     adc_A2_correlation_3d.append(corre)
#     adc_A2_energy_3d.append(ener)
#     adc_A2_homogeneity_3d.append(homo)

# ### Get for ADC A3
# adc_A3_contrast_3d    = []
# adc_A3_correlation_3d = []
# adc_A3_energy_3d      = []
# adc_A3_homogeneity_3d = []

# for pid in range(adc_A3_roi_new.shape[0]):
    
#     contr, corre, ener, homo, mat = glcm_slice_extract_3d(adc_A3_roi_norm, pid)
#     adc_A3_contrast_3d.append(contr)
#     adc_A3_correlation_3d.append(corre)
#     adc_A3_energy_3d.append(ener)
#     adc_A3_homogeneity_3d.append(homo)

# ### Get for ADC B1
# adc_B1_contrast_3d    = []
# adc_B1_correlation_3d = []
# adc_B1_energy_3d      = []
# adc_B1_homogeneity_3d = []

# for pid in range(adc_B1_roi_new.shape[0]):
    
#     contr, corre, ener, homo, mat = glcm_slice_extract_3d(adc_B1_roi_norm, pid)
#     adc_B1_contrast_3d.append(contr)
#     adc_B1_correlation_3d.append(corre)
#     adc_B1_energy_3d.append(ener)
#     adc_B1_homogeneity_3d.append(homo)

# ### Get for ADC B2
# adc_B2_contrast_3d    = []
# adc_B2_correlation_3d = []
# adc_B2_energy_3d      = []
# adc_B2_homogeneity_3d = []

# for pid in range(adc_B2_roi_new.shape[0]):
    
#     contr, corre, ener, homo, mat = glcm_slice_extract_3d(adc_B2_roi_norm, pid)
#     adc_B2_contrast_3d.append(contr)
#     adc_B2_correlation_3d.append(corre)
#     adc_B2_energy_3d.append(ener)
#     adc_B2_homogeneity_3d.append(homo)

# ### Get for ADC B3
# adc_B3_contrast_3d    = []
# adc_B3_correlation_3d = []
# adc_B3_energy_3d      = []
# adc_B3_homogeneity_3d = []

# for pid in range(adc_B3_roi_new.shape[0]):
    
#     contr, corre, ener, homo, mat = glcm_slice_extract_3d(adc_B3_roi_norm, pid)
#     adc_B3_contrast_3d.append(contr)
#     adc_B3_correlation_3d.append(corre)
#     adc_B3_energy_3d.append(ener)
#     adc_B3_homogeneity_3d.append(homo)


# #%% Save all the 3D GLCM features
# adc_T1_contrast_3d    = adc_A1_contrast_3d    + adc_B1_contrast_3d
# adc_T2_contrast_3d    = adc_A2_contrast_3d    + adc_B2_contrast_3d
# adc_T3_contrast_3d    = adc_A3_contrast_3d    + adc_B3_contrast_3d
# adc_T1_correlation_3d = adc_A1_correlation_3d + adc_B1_correlation_3d
# adc_T2_correlation_3d = adc_A2_correlation_3d + adc_B2_correlation_3d
# adc_T3_correlation_3d = adc_A3_correlation_3d + adc_B3_correlation_3d
# adc_T1_energy_3d      = adc_A1_energy_3d      + adc_B1_energy_3d
# adc_T2_energy_3d      = adc_A2_energy_3d      + adc_B2_energy_3d
# adc_T3_energy_3d      = adc_A3_energy_3d      + adc_B3_energy_3d
# adc_T1_homogeneity_3d = adc_A1_homogeneity_3d + adc_B1_homogeneity_3d
# adc_T2_homogeneity_3d = adc_A2_homogeneity_3d + adc_B2_homogeneity_3d
# adc_T3_homogeneity_3d = adc_A3_homogeneity_3d + adc_B3_homogeneity_3d

# dce_T1_contrast_3d    = adc_A1_contrast_3d    + adc_B1_contrast_3d
# dce_T2_contrast_3d    = adc_A2_contrast_3d    + adc_B2_contrast_3d
# dce_T3_contrast_3d    = adc_A3_contrast_3d    + adc_B3_contrast_3d
# dce_T1_correlation_3d = adc_A1_correlation_3d + adc_B1_correlation_3d
# dce_T2_correlation_3d = adc_A2_correlation_3d + adc_B2_correlation_3d
# dce_T3_correlation_3d = adc_A3_correlation_3d + adc_B3_correlation_3d
# dce_T1_energy_3d      = adc_A1_energy_3d      + adc_B1_energy_3d
# dce_T2_energy_3d      = adc_A2_energy_3d      + adc_B2_energy_3d
# dce_T3_energy_3d      = adc_A3_energy_3d      + adc_B3_energy_3d
# dce_T1_homogeneity_3d = adc_A1_homogeneity_3d + adc_B1_homogeneity_3d
# dce_T2_homogeneity_3d = adc_A2_homogeneity_3d + adc_B2_homogeneity_3d
# dce_T3_homogeneity_3d = adc_A3_homogeneity_3d + adc_B3_homogeneity_3d

# pet_T1_contrast_3d    = pet_A1_contrast_3d    + pet_B1_contrast_3d
# pet_T2_contrast_3d    = pet_A2_contrast_3d    + pet_B2_contrast_3d
# pet_T3_contrast_3d    = pet_A3_contrast_3d    + pet_B3_contrast_3d
# pet_T1_correlation_3d = pet_A1_correlation_3d + pet_B1_correlation_3d
# pet_T2_correlation_3d = pet_A2_correlation_3d + pet_B2_correlation_3d
# pet_T3_correlation_3d = pet_A3_correlation_3d + pet_B3_correlation_3d
# pet_T1_energy_3d      = pet_A1_energy_3d      + pet_B1_energy_3d
# pet_T2_energy_3d      = pet_A2_energy_3d      + pet_B2_energy_3d
# pet_T3_energy_3d      = pet_A3_energy_3d      + pet_B3_energy_3d
# pet_T1_homogeneity_3d = pet_A1_homogeneity_3d + pet_B1_homogeneity_3d
# pet_T2_homogeneity_3d = pet_A2_homogeneity_3d + pet_B2_homogeneity_3d
# pet_T3_homogeneity_3d = pet_A3_homogeneity_3d + pet_B3_homogeneity_3d

# #%% Now save all the 3D stats

# # with open('adc_T1_contrast_3d.pkl', 'wb') as file:
# #     pickle.dump(adc_T1_contrast_3d, file)
# # with open('adc_T2_contrast_3d.pkl', 'wb') as file:
# #     pickle.dump(adc_T2_contrast_3d, file)
# # with open('adc_T3_contrast_3d.pkl', 'wb') as file:
# #     pickle.dump(adc_T3_contrast_3d, file)
# # with open('adc_T1_correlation_3d.pkl', 'wb') as file:
# #     pickle.dump(adc_T1_correlation_3d, file)
# # with open('adc_T2_correlation_3d.pkl', 'wb') as file:
# #     pickle.dump(adc_T2_correlation_3d, file)
# # with open('adc_T3_correlation_3d.pkl', 'wb') as file:
# #     pickle.dump(adc_T3_correlation_3d, file)
# # with open('adc_T1_energy_3d.pkl', 'wb') as file:
# #     pickle.dump(adc_T1_energy_3d, file)
# # with open('adc_T2_energy_3d.pkl', 'wb') as file:
# #     pickle.dump(adc_T2_energy_3d, file)
# # with open('adc_T3_energy_3d.pkl', 'wb') as file:
# #     pickle.dump(adc_T3_energy_3d, file)
# # with open('adc_T1_homogeneity_3d.pkl', 'wb') as file:
# #     pickle.dump(adc_T1_homogeneity_3d, file)
# # with open('adc_T2_homogeneity_3d.pkl', 'wb') as file:
# #     pickle.dump(adc_T2_homogeneity_3d, file)
# # with open('adc_T3_homogeneity_3d.pkl', 'wb') as file:
# #     pickle.dump(adc_T3_homogeneity_3d, file)
    
    

# # with open('dce_T1_contrast_3d.pkl', 'wb') as file:
# #     pickle.dump(dce_T1_contrast_3d, file)
# # with open('dce_T2_contrast_3d.pkl', 'wb') as file:
# #     pickle.dump(dce_T2_contrast_3d, file)
# # with open('dce_T3_contrast_3d.pkl', 'wb') as file:
# #     pickle.dump(dce_T3_contrast_3d, file)
# # with open('dce_T1_correlation_3d.pkl', 'wb') as file:
# #     pickle.dump(dce_T1_correlation_3d, file)
# # with open('dce_T2_correlation_3d.pkl', 'wb') as file:
# #     pickle.dump(dce_T2_correlation_3d, file)
# # with open('dce_T3_correlation_3d.pkl', 'wb') as file:
# #     pickle.dump(dce_T3_correlation_3d, file)
# # with open('dce_T1_energy_3d.pkl', 'wb') as file:
# #     pickle.dump(dce_T1_energy_3d, file)
# # with open('dce_T2_energy_3d.pkl', 'wb') as file:
# #     pickle.dump(dce_T2_energy_3d, file)
# # with open('dce_T3_energy_3d.pkl', 'wb') as file:
# #     pickle.dump(dce_T3_energy_3d, file)
# # with open('dce_T1_homogeneity_3d.pkl', 'wb') as file:
# #     pickle.dump(dce_T1_homogeneity_3d, file)
# # with open('dce_T2_homogeneity_3d.pkl', 'wb') as file:
# #     pickle.dump(dce_T2_homogeneity_3d, file)
# # with open('dce_T3_homogeneity_3d.pkl', 'wb') as file:
# #     pickle.dump(dce_T3_homogeneity_3d, file)  
    
      
    
# # with open('pet_T1_contrast_3d.pkl', 'wb') as file:
# #     pickle.dump(pet_T1_contrast_3d, file)
# # with open('pet_T2_contrast_3d.pkl', 'wb') as file:
# #     pickle.dump(pet_T2_contrast_3d, file)
# # with open('pet_T3_contrast_3d.pkl', 'wb') as file:
# #     pickle.dump(pet_T3_contrast_3d, file)
# # with open('pet_T1_correlation_3d.pkl', 'wb') as file:
# #     pickle.dump(pet_T1_correlation_3d, file)
# # with open('pet_T2_correlation_3d.pkl', 'wb') as file:
# #     pickle.dump(pet_T2_correlation_3d, file)
# # with open('pet_T3_correlation_3d.pkl', 'wb') as file:
# #     pickle.dump(pet_T3_correlation_3d, file)
# # with open('pet_T1_energy_3d.pkl', 'wb') as file:
# #     pickle.dump(pet_T1_energy_3d, file)
# # with open('pet_T2_energy_3d.pkl', 'wb') as file:
# #     pickle.dump(pet_T2_energy_3d, file)
# # with open('pet_T3_energy_3d.pkl', 'wb') as file:
# #     pickle.dump(pet_T3_energy_3d, file)
# # with open('pet_T1_homogeneity_3d.pkl', 'wb') as file:
# #     pickle.dump(pet_T1_homogeneity_3d, file)
# # with open('pet_T2_homogeneity_3d.pkl', 'wb') as file:
# #     pickle.dump(pet_T2_homogeneity_3d, file)
# # with open('pet_T3_homogeneity_3d.pkl', 'wb') as file:
# #     pickle.dump(pet_T3_homogeneity_3d, file)

#%% Step 2: Check generalization with k-fold cross-validation

# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.linear_model import LogisticRegression

# # Define the number of splits (k)
# k = 5

# # Initialize the KFold object
# kf = KFold(n_splits=k)

# # Initialize the model
# model = LogisticRegression()

# # Lists to store evaluation metrics
# accuracy_scores = []
# precision_scores = []
# recall_scores = []
# f1_scores = []

# # Perform k-fold cross-validation
# for train_index, test_index in kf.split(X):
#     X_train, X_test = selected_features[train_index], selected_features[test_index]
#     y_train, y_test = Y[train_index], Y[test_index]
    
#     # Fit the model on the training data
#     model.fit(X_train, y_train)
    
#     # Evaluate the model on the test data
#     y_pred = model.predict(X_test)
    
#     # Calculate evaluation metrics
#     accuracy  = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall    = recall_score(y_test, y_pred)
#     f1        = f1_score(y_test, y_pred)
    
#     accuracy_scores.append(accuracy)
#     precision_scores.append(precision)
#     recall_scores.append(recall)
#     f1_scores.append(f1)

# # Calculate average evaluation metrics
# average_accuracy = np.mean(accuracy_scores)
# average_precision = np.mean(precision_scores)
# average_recall = np.mean(recall_scores)
# average_f1 = np.mean(f1_scores)

# # Print results
# print(f'Average Accuracy: {average_accuracy}')
# print(f'Average Precision: {average_precision}')
# print(f'Average Recall: {average_recall}')
# print(f'Average F1-score: {average_f1}')

