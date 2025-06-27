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

#%% Load the PARTs and ROIs

### First p#: period (time) Second p#: part (p1: cervical, p2:tumor, c:combine)
os.chdir('/Users/hafeng/.spyder-py3/cervical_cancer_analysis/')

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

#%% Plot the contours for visualization

def draw_contour(img):
    
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.voxels(img)
    plt.title('3D Contour Visualization')
    
# draw_contour(pet_A1[0])

#%% Test Slice plot

img_test = dce_A1[0]

for i in range(32):
    plt.figure()
    plt.imshow(np.squeeze(img_test[:,:,i]), cmap=plt.cm.bone)
    plt.title('Image of SLICE: ' + str(i))
    plt.colorbar()

img_test = p1_p1_data[0]

for i in range(32):
    plt.figure()
    plt.imshow(np.squeeze(img_test[:,:,i]), cmap=plt.cm.bone)
    plt.title('Image of SLICE: ' + str(i))
    plt.colorbar()
    
img_test = dce_A1[0]*p1_p1_data[0]

for i in range(32):
    plt.figure()
    plt.imshow(np.squeeze(img_test[:,:,i]), cmap=plt.cm.bone)
    plt.title('Image of SLICE: ' + str(i))
    plt.colorbar()
    
### This result makes sense, move on to extract regional stats ###

#%% Next, try a more cleaned way to extract ROI

def ROI_extractor(matrix, mask, pid):
    
    mx, my, mz = mask[pid].shape
    result = []
    
    for i in range(mx):
        for j in range(my):
            for k in range(mz):
                
                if mask[pid,i,j,k] != 0 and np.var(matrix[pid,:,:,:]) !=0 :
                    result.append(matrix[pid,i,j,k])
                    
    result = np.array(result)
    
    return result

#%% Test the tumor ROI view

# img_test = pet_B1_roi[0]

# for i in range(32):
#     plt.figure()
#     plt.imshow(np.squeeze(img_test[:,:,i]), cmap=plt.cm.bone)
#     plt.title('Image of SLICE: ' + str(i))
#     plt.colorbar()

#%% Next, generate the box plots for different statistics, we first do some quick tests

### From the paper, the following needs to be computed: (A) mean (B) quantiles (C) coeff of variation (D) skewness (E) kurtosis
from scipy import stats

### Before that, let's try to make them into a flattened vector
roi_pixels = pet_A1

# Compute statistical measures
mean_value = np.mean(roi_pixels)
quantiles = np.percentile(roi_pixels, [25, 50, 75])
coefficient_of_variation = np.std(roi_pixels) / mean_value
skewness = stats.skew(roi_pixels)
kurtosis = stats.kurtosis(roi_pixels)

print(f"Mean: {mean_value}")
print(f"Quantiles (25%, 50%, 75%): {quantiles}")
print(f"Coefficient of Variation: {coefficient_of_variation}")
print(f"Skewness: {skewness}")
print(f"Kurtosis: {kurtosis}")

# Assuming you have multiple sets of data
data1 = [1, 2, 3, 4, 5]
data2 = [2, 3, 4, 5, 6]
data3 = [3, 4, 5, 6, 7]

# Create a box-and-whisker plot for multiple datasets
plt.figure()
plt.boxplot([data1, data2, data3])

# Add labels for each dataset
plt.xticks([1, 2, 3], ['Dataset 1', 'Dataset 2', 'Dataset 3'])

# Add a title and labels
plt.title('Box-and-Whisker Plot')
plt.xlabel('Datasets')
plt.ylabel('Values')

# Show the plot
plt.show()

#%% Now after the testing code, let's move on to the real data, first do ADC DWI

mean_value_adc_t1 = []
quantiles_adc_t1  = []
coeff_var_adc_t1  = []
skewness_adc_t1   = []
kurtosis_adc_t1   = []
max_value_adc_t1  = []

mean_value_adc_t2 = []
quantiles_adc_t2  = []
coeff_var_adc_t2  = []
skewness_adc_t2   = []
kurtosis_adc_t2   = []
max_value_adc_t2  = []

mean_value_adc_t3 = []
quantiles_adc_t3  = []
coeff_var_adc_t3  = []
skewness_adc_t3   = []
kurtosis_adc_t3   = []
max_value_adc_t3  = []

for i in range(23):
    
    if i < 12:
        PID = i
        if PID != 100 and PID !=5:
        
            adc_A1_roi = ROI_extractor(adc_A1, t1_p2_mask_A, PID)
            adc_A2_roi = ROI_extractor(adc_A2, t2_p2_mask_A, PID)
            adc_A3_roi = ROI_extractor(adc_A3, t3_p2_mask_A, PID)
            
            roi_pixels = adc_A1_roi
            if len(roi_pixels) == 0:
                print('Empty Cell at A3 PID: ', PID)
            else:
                print('Currently working on A1 PID: ', PID)
                mean_value_adc_t1.append(np.mean(roi_pixels))
                quantiles_adc_t1.append(np.percentile(roi_pixels, 10))
                coeff_var_adc_t1.append(np.std(roi_pixels) / np.mean(roi_pixels))
                skewness_adc_t1.append(stats.skew(roi_pixels))
                kurtosis_adc_t1.append(stats.kurtosis(roi_pixels))
                max_value_adc_t1.append(np.max(roi_pixels))
            
            roi_pixels = adc_A2_roi
            if len(roi_pixels) == 0:
                print('Empty Cell at A3 PID: ', PID)
            else:
                print('Currently working on A2 PID: ', PID)
                mean_value_adc_t2.append(np.mean(roi_pixels))
                quantiles_adc_t2.append(np.percentile(roi_pixels, 10))
                coeff_var_adc_t2.append(np.std(roi_pixels) / np.mean(roi_pixels))
                skewness_adc_t2.append(stats.skew(roi_pixels))
                kurtosis_adc_t2.append(stats.kurtosis(roi_pixels))
                max_value_adc_t2.append(np.max(roi_pixels))
            
            roi_pixels = adc_A3_roi
            if len(roi_pixels) == 0:
                print('Empty Cell at A3 PID: ', PID)
            else: 
                print('Currently working on A3 PID: ', PID)
                mean_value_adc_t3.append(np.mean(roi_pixels))
                quantiles_adc_t3.append(np.percentile(roi_pixels, 10))
                coeff_var_adc_t3.append(np.std(roi_pixels) / np.mean(roi_pixels))
                skewness_adc_t3.append(stats.skew(roi_pixels))
                kurtosis_adc_t3.append(stats.kurtosis(roi_pixels))
                max_value_adc_t3.append(np.max(roi_pixels))
        
    else:
        PID = i - 12
        
        if PID != 9 and PID !=10:
        
            adc_B1_roi = ROI_extractor(adc_B1, t1_p2_mask_B, PID)
            adc_B2_roi = ROI_extractor(adc_B2, t2_p2_mask_B, PID)
            adc_B3_roi = ROI_extractor(adc_B3, t3_p2_mask_B, PID)
            
            roi_pixels = adc_B1_roi
            if len(roi_pixels) == 0:
                print('Empty Cell at A3 PID: ', PID)
            else:
                print('Currently working on B1 PID: ', PID)
                mean_value_adc_t1.append(np.mean(roi_pixels))
                quantiles_adc_t1.append(np.percentile(roi_pixels, 10))
                coeff_var_adc_t1.append(np.std(roi_pixels) / np.mean(roi_pixels))
                skewness_adc_t1.append(stats.skew(roi_pixels))
                kurtosis_adc_t1.append(stats.kurtosis(roi_pixels))
                max_value_adc_t1.append(np.max(roi_pixels))
            
            roi_pixels = adc_B2_roi
            # if PID != 9 and not np.all(roi_pixels == 0):
            if not np.all(roi_pixels == 0) or len(roi_pixels)==0:
                print('Currently working on B2 PID: ', PID)
                mean_value_adc_t2.append(np.mean(roi_pixels))
                quantiles_adc_t2.append(np.percentile(roi_pixels, 10))
                coeff_var_adc_t2.append(np.std(roi_pixels) / np.mean(roi_pixels))
                skewness_adc_t2.append(stats.skew(roi_pixels))
                kurtosis_adc_t2.append(stats.kurtosis(roi_pixels))
                max_value_adc_t2.append(np.max(roi_pixels))
            elif np.all(roi_pixels == 0) and len(roi_pixels)!=0:
                print('Sample all zeros for B2 PID:' , PID)
                print(roi_pixels)
                
            else:
                print('Empty Cell at B2 PID: ', PID)
            
            roi_pixels = adc_B3_roi
            # if PID != 10:
            if not np.all(roi_pixels == 0) or len(roi_pixels)==0:
                print('Currently working on B3 PID: ', PID)
                mean_value_adc_t3.append(np.mean(roi_pixels))
                quantiles_adc_t3.append(np.percentile(roi_pixels, 10))
                coeff_var_adc_t3.append(np.std(roi_pixels) / np.mean(roi_pixels))
                skewness_adc_t3.append(stats.skew(roi_pixels))
                kurtosis_adc_t3.append(stats.kurtosis(roi_pixels))
                max_value_adc_t3.append(np.max(roi_pixels))
            else:
                print('Empty Cell at B3 PID: ', PID)

#%% Normalize the features

mean_value_adc_t1_norm = (mean_value_adc_t1 - np.min(mean_value_adc_t1)) / (np.max(mean_value_adc_t1) - np.min(mean_value_adc_t1))
mean_value_adc_t2_norm = (mean_value_adc_t2 - np.min(mean_value_adc_t2)) / (np.max(mean_value_adc_t2) - np.min(mean_value_adc_t2))
mean_value_adc_t3_norm = (mean_value_adc_t3 - np.min(mean_value_adc_t3)) / (np.max(mean_value_adc_t3) - np.min(mean_value_adc_t3))

quantiles_adc_t1_norm = (quantiles_adc_t1 - np.min(quantiles_adc_t1)) / (np.max(quantiles_adc_t1) - np.min(quantiles_adc_t1))
quantiles_adc_t2_norm = (quantiles_adc_t2 - np.min(quantiles_adc_t2)) / (np.max(quantiles_adc_t2) - np.min(quantiles_adc_t2))
quantiles_adc_t3_norm = (quantiles_adc_t3 - np.min(quantiles_adc_t3)) / (np.max(quantiles_adc_t3) - np.min(quantiles_adc_t3))
        
coeff_var_adc_t1_norm = (coeff_var_adc_t1 - np.min(coeff_var_adc_t1)) / (np.max(coeff_var_adc_t1) - np.min(coeff_var_adc_t1))
coeff_var_adc_t2_norm = (coeff_var_adc_t2 - np.min(coeff_var_adc_t2)) / (np.max(coeff_var_adc_t2) - np.min(coeff_var_adc_t2))
coeff_var_adc_t3_norm = (coeff_var_adc_t3 - np.min(coeff_var_adc_t3)) / (np.max(coeff_var_adc_t3) - np.min(coeff_var_adc_t3))

skewness_adc_t1_norm = (skewness_adc_t1 - np.min(skewness_adc_t1)) / (np.max(skewness_adc_t1) - np.min(skewness_adc_t1))
skewness_adc_t2_norm = (skewness_adc_t2 - np.min(skewness_adc_t2)) / (np.max(skewness_adc_t2) - np.min(skewness_adc_t2))
skewness_adc_t3_norm = (skewness_adc_t3 - np.min(skewness_adc_t3)) / (np.max(skewness_adc_t3) - np.min(skewness_adc_t3))

kurtosis_adc_t1_norm = (kurtosis_adc_t1 - np.min(kurtosis_adc_t1)) / (np.max(kurtosis_adc_t1) - np.min(kurtosis_adc_t1))
kurtosis_adc_t2_norm = (kurtosis_adc_t2 - np.min(kurtosis_adc_t2)) / (np.max(kurtosis_adc_t2) - np.min(kurtosis_adc_t2))
kurtosis_adc_t3_norm = (kurtosis_adc_t3 - np.min(kurtosis_adc_t3)) / (np.max(kurtosis_adc_t3) - np.min(kurtosis_adc_t3))

max_value_adc_t1_norm = (max_value_adc_t1 - np.min(max_value_adc_t1)) / (np.max(max_value_adc_t1) - np.min(max_value_adc_t1))
max_value_adc_t2_norm = (max_value_adc_t2 - np.min(max_value_adc_t2)) / (np.max(max_value_adc_t2) - np.min(max_value_adc_t2))
max_value_adc_t3_norm = (max_value_adc_t3 - np.min(max_value_adc_t3)) / (np.max(max_value_adc_t3) - np.min(max_value_adc_t3))

#%% Now generating the box plot

plt.figure()
plt.boxplot([mean_value_adc_t1, mean_value_adc_t2, mean_value_adc_t3])

# Add labels for each dataset
plt.xticks([1, 2, 3], ['Time 1', 'Time 2', 'Time 3'])

# Add a title and labels
plt.title('Box-and-Whisker Plot')
plt.xlabel('DWI ADC Datasets')
plt.ylabel('Mean Values')

# Show the plot
plt.show()

##############################################################################

plt.figure()
plt.boxplot([quantiles_adc_t1, quantiles_adc_t2, quantiles_adc_t3])

# Add labels for each dataset
plt.xticks([1, 2, 3], ['Time 1', 'Time 2', 'Time 3'])

# Add a title and labels
plt.title('Box-and-Whisker Plot')
plt.xlabel('DWI ADC Datasets')
plt.ylabel('Quantiles Values')

# Show the plot
plt.show()

##############################################################################

plt.figure()
plt.boxplot([coeff_var_adc_t1, coeff_var_adc_t2, coeff_var_adc_t3])

# Add labels for each dataset
plt.xticks([1, 2, 3], ['Time 1', 'Time 2', 'Time 3'])

# Add a title and labels
plt.title('Box-and-Whisker Plot')
plt.xlabel('DWI ADC Datasets')
plt.ylabel('Coeff Var Values')

# Show the plot
plt.show()

##############################################################################

plt.figure()
plt.boxplot([skewness_adc_t1, skewness_adc_t2, skewness_adc_t3])

# Add labels for each dataset
plt.xticks([1, 2, 3], ['Time 1', 'Time 2', 'Time 3'])

# Add a title and labels
plt.title('Box-and-Whisker Plot')
plt.xlabel('DWI ADC Datasets')
plt.ylabel('Skewness Values')

# Show the plot
plt.show()

##############################################################################

plt.figure()
plt.boxplot([kurtosis_adc_t1, kurtosis_adc_t2, kurtosis_adc_t3])

# Add labels for each dataset
plt.xticks([1, 2, 3], ['Time 1', 'Time 2', 'Time 3'])

# Add a title and labels
plt.title('Box-and-Whisker Plot')
plt.xlabel('DWI ADC Datasets')
plt.ylabel('Kurtosis Values')

# Show the plot
plt.show()

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

#%% Work on DCE Data

mean_value_dce_t1 = []
quantiles_dce_t1  = []
coeff_var_dce_t1  = []
skewness_dce_t1   = []
kurtosis_dce_t1   = []
max_value_dce_t1  = []

mean_value_dce_t2 = []
quantiles_dce_t2  = []
coeff_var_dce_t2  = []
skewness_dce_t2   = []
kurtosis_dce_t2   = []
max_value_dce_t2  = []

mean_value_dce_t3 = []
quantiles_dce_t3  = []
coeff_var_dce_t3  = []
skewness_dce_t3   = []
kurtosis_dce_t3   = []
max_value_dce_t3  = []

for i in range(23):
    
    if i < 12:
        PID = i
        
        if PID != 100 and PID != 5:
            dce_A1_roi = ROI_extractor(dce_A1, t1_p2_mask_A, PID)
            dce_A2_roi = ROI_extractor(dce_A2, t2_p2_mask_A, PID)
            dce_A3_roi = ROI_extractor(dce_A3, t3_p2_mask_A, PID)
            
            roi_pixels = dce_A1_roi
            print('Currently working on A1 PID: ', PID)
            mean_value_dce_t1.append(np.mean(roi_pixels))
            quantiles_dce_t1.append(np.percentile(roi_pixels, 10))
            coeff_var_dce_t1.append(np.std(roi_pixels) / np.mean(roi_pixels))
            skewness_dce_t1.append(stats.skew(roi_pixels))
            kurtosis_dce_t1.append(stats.kurtosis(roi_pixels))
            max_value_dce_t1.append(np.max(roi_pixels))
            
            roi_pixels = dce_A2_roi
            print('Currently working on A2 PID: ', PID)
            mean_value_dce_t2.append(np.mean(roi_pixels))
            quantiles_dce_t2.append(np.percentile(roi_pixels, 10))
            coeff_var_dce_t2.append(np.std(roi_pixels) / np.mean(roi_pixels))
            skewness_dce_t2.append(stats.skew(roi_pixels))
            kurtosis_dce_t2.append(stats.kurtosis(roi_pixels))
            max_value_dce_t2.append(np.max(roi_pixels))
            
            roi_pixels = dce_A3_roi
            if len(roi_pixels) == 0:
                print('Empty Cell at A3 PID: ', PID)
            elif PID != 5:
                print('Currently working on A3 PID: ', PID)
                mean_value_dce_t3.append(np.mean(roi_pixels))
                quantiles_dce_t3.append(np.percentile(roi_pixels, 10))
                coeff_var_dce_t3.append(np.std(roi_pixels) / np.mean(roi_pixels))
                skewness_dce_t3.append(stats.skew(roi_pixels))
                kurtosis_dce_t3.append(stats.kurtosis(roi_pixels))
                max_value_dce_t3.append(np.max(roi_pixels))
        
    else:
        PID = i - 12
        if PID !=9  and PID !=10:
            dce_B1_roi = ROI_extractor(dce_B1, t1_p2_mask_B, PID)
            dce_B2_roi = ROI_extractor(dce_B3, t2_p2_mask_B, PID)
            dce_B3_roi = ROI_extractor(dce_B3, t3_p2_mask_B, PID)
            
            roi_pixels = dce_B1_roi
            print('Currently working on B1 PID: ', PID)
            mean_value_dce_t1.append(np.mean(roi_pixels))
            quantiles_dce_t1.append(np.percentile(roi_pixels, 10))
            coeff_var_dce_t1.append(np.std(roi_pixels) / np.mean(roi_pixels))
            skewness_dce_t1.append(stats.skew(roi_pixels))
            kurtosis_dce_t1.append(stats.kurtosis(roi_pixels))
            max_value_dce_t1.append(np.max(roi_pixels))
            
            roi_pixels = dce_B2_roi
    
            print('Currently working on B2 PID: ', PID)
            mean_value_dce_t2.append(np.mean(roi_pixels))
            quantiles_dce_t2.append(np.percentile(roi_pixels, 10))
            coeff_var_dce_t2.append(np.std(roi_pixels) / np.mean(roi_pixels))
            skewness_dce_t2.append(stats.skew(roi_pixels))
            kurtosis_dce_t2.append(stats.kurtosis(roi_pixels))
            max_value_dce_t2.append(np.max(roi_pixels))
            
            roi_pixels = dce_B3_roi
    
            print('Currently working on B3 PID: ', PID)
            mean_value_dce_t3.append(np.mean(roi_pixels))
            quantiles_dce_t3.append(np.percentile(roi_pixels, 10))
            coeff_var_dce_t3.append(np.std(roi_pixels) / np.mean(roi_pixels))
            skewness_dce_t3.append(stats.skew(roi_pixels))
            kurtosis_dce_t3.append(stats.kurtosis(roi_pixels))
            max_value_dce_t3.append(np.max(roi_pixels))

###############################################################################

plt.figure()
plt.boxplot([mean_value_dce_t1, mean_value_dce_t2, mean_value_dce_t3])

# Add labels for each dataset
plt.xticks([1, 2, 3], ['Time 1', 'Time 2', 'Time 3'])

# Add a title and labels
plt.title('Box-and-Whisker Plot')
plt.xlabel('DCE Datasets')
plt.ylabel('Mean Values')

# Show the plot
plt.show()

##############################################################################

plt.figure()
plt.boxplot([quantiles_dce_t1, quantiles_dce_t2, quantiles_dce_t3])

# Add labels for each dataset
plt.xticks([1, 2, 3], ['Time 1', 'Time 2', 'Time 3'])

# Add a title and labels
plt.title('Box-and-Whisker Plot')
plt.xlabel('DCE Datasets')
plt.ylabel('Quantiles Values')

# Show the plot
plt.show()

##############################################################################

plt.figure()
plt.boxplot([coeff_var_dce_t1, coeff_var_dce_t2, coeff_var_dce_t3])

# Add labels for each dataset
plt.xticks([1, 2, 3], ['Time 1', 'Time 2', 'Time 3'])

# Add a title and labels
plt.title('Box-and-Whisker Plot')
plt.xlabel('DCE Datasets')
plt.ylabel('Coeff Var Values')

# Show the plot
plt.show()

##############################################################################

plt.figure()
plt.boxplot([skewness_dce_t1, skewness_dce_t2, skewness_dce_t3])

# Add labels for each dataset
plt.xticks([1, 2, 3], ['Time 1', 'Time 2', 'Time 3'])

# Add a title and labels
plt.title('Box-and-Whisker Plot')
plt.xlabel('DCE Datasets')
plt.ylabel('Skewness Values')

# Show the plot
plt.show()

##############################################################################

plt.figure()
plt.boxplot([kurtosis_dce_t1, kurtosis_dce_t2, kurtosis_dce_t3])

# Add labels for each dataset
plt.xticks([1, 2, 3], ['Time 1', 'Time 2', 'Time 3'])

# Add a title and labels
plt.title('Box-and-Whisker Plot')
plt.xlabel('DCE Datasets')
plt.ylabel('Kurtosis Values')

# Show the plot
plt.show()

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################

#%% Normalize the features

mean_value_dce_t1_norm = (mean_value_dce_t1 - np.min(mean_value_dce_t1)) / (np.max(mean_value_dce_t1) - np.min(mean_value_dce_t1))
mean_value_dce_t2_norm = (mean_value_dce_t2 - np.min(mean_value_dce_t2)) / (np.max(mean_value_dce_t2) - np.min(mean_value_dce_t2))
mean_value_dce_t3_norm = (mean_value_dce_t3 - np.min(mean_value_dce_t3)) / (np.max(mean_value_dce_t3) - np.min(mean_value_dce_t3))

quantiles_dce_t1_norm = (quantiles_dce_t1 - np.min(quantiles_dce_t1)) / (np.max(quantiles_dce_t1) - np.min(quantiles_dce_t1))
quantiles_dce_t2_norm = (quantiles_dce_t2 - np.min(quantiles_dce_t2)) / (np.max(quantiles_dce_t2) - np.min(quantiles_dce_t2))
quantiles_dce_t3_norm = (quantiles_dce_t3 - np.min(quantiles_dce_t3)) / (np.max(quantiles_dce_t3) - np.min(quantiles_dce_t3))
        
coeff_var_dce_t1_norm = (coeff_var_dce_t1 - np.min(coeff_var_dce_t1)) / (np.max(coeff_var_dce_t1) - np.min(coeff_var_dce_t1))
coeff_var_dce_t2_norm = (coeff_var_dce_t2 - np.min(coeff_var_dce_t2)) / (np.max(coeff_var_dce_t2) - np.min(coeff_var_dce_t2))
coeff_var_dce_t3_norm = (coeff_var_dce_t3 - np.min(coeff_var_dce_t3)) / (np.max(coeff_var_dce_t3) - np.min(coeff_var_dce_t3))

skewness_dce_t1_norm = (skewness_dce_t1 - np.min(skewness_dce_t1)) / (np.max(skewness_dce_t1) - np.min(skewness_dce_t1))
skewness_dce_t2_norm = (skewness_dce_t2 - np.min(skewness_dce_t2)) / (np.max(skewness_dce_t2) - np.min(skewness_dce_t2))
skewness_dce_t3_norm = (skewness_dce_t3 - np.min(skewness_dce_t3)) / (np.max(skewness_dce_t3) - np.min(skewness_dce_t3))

kurtosis_dce_t1_norm = (kurtosis_dce_t1 - np.min(kurtosis_dce_t1)) / (np.max(kurtosis_dce_t1) - np.min(kurtosis_dce_t1))
kurtosis_dce_t2_norm = (kurtosis_dce_t2 - np.min(kurtosis_dce_t2)) / (np.max(kurtosis_dce_t2) - np.min(kurtosis_dce_t2))
kurtosis_dce_t3_norm = (kurtosis_dce_t3 - np.min(kurtosis_dce_t3)) / (np.max(kurtosis_dce_t3) - np.min(kurtosis_dce_t3))

max_value_dce_t1_norm = (max_value_dce_t1 - np.min(max_value_dce_t1)) / (np.max(max_value_dce_t1) - np.min(max_value_dce_t1))
max_value_dce_t2_norm = (max_value_dce_t2 - np.min(max_value_dce_t2)) / (np.max(max_value_dce_t2) - np.min(max_value_dce_t2))
max_value_dce_t3_norm = (max_value_dce_t3 - np.min(max_value_dce_t3)) / (np.max(max_value_dce_t3) - np.min(max_value_dce_t3))

#%% Work on PET Data

mean_value_pet_t1 = []
quantiles_pet_t1  = []
coeff_var_pet_t1  = []
skewness_pet_t1   = []
kurtosis_pet_t1   = []
max_value_pet_t1  = []

mean_value_pet_t2 = []
quantiles_pet_t2  = []
coeff_var_pet_t2  = []
skewness_pet_t2   = []
kurtosis_pet_t2   = []
max_value_pet_t2  = []

mean_value_pet_t3 = []
quantiles_pet_t3  = []
coeff_var_pet_t3  = []
skewness_pet_t3   = []
kurtosis_pet_t3   = []
max_value_pet_t3  = []

for i in range(23):
    
    if i < 12:
        PID = i
        if PID != 100 and PID != 5:
            pet_A1_roi = ROI_extractor(pet_A1, t1_p2_mask_A, PID)
            pet_A2_roi = ROI_extractor(pet_A2, t2_p2_mask_A, PID)
            pet_A3_roi = ROI_extractor(pet_A3, t3_p2_mask_A, PID)
            
            roi_pixels = pet_A1_roi
            print('Currently working on A1 PID: ', PID)
            mean_value_pet_t1.append(np.mean(roi_pixels))
            quantiles_pet_t1.append(np.percentile(roi_pixels, 10))
            coeff_var_pet_t1.append(np.std(roi_pixels) / np.mean(roi_pixels))
            skewness_pet_t1.append(stats.skew(roi_pixels))
            kurtosis_pet_t1.append(stats.kurtosis(roi_pixels))
            max_value_pet_t1.append(np.max(roi_pixels))
            
            roi_pixels = pet_A2_roi
            print('Currently working on A2 PID: ', PID)
            mean_value_pet_t2.append(np.mean(roi_pixels))
            quantiles_pet_t2.append(np.percentile(roi_pixels, 10))
            coeff_var_pet_t2.append(np.std(roi_pixels) / np.mean(roi_pixels))
            skewness_pet_t2.append(stats.skew(roi_pixels))
            kurtosis_pet_t2.append(stats.kurtosis(roi_pixels))
            max_value_pet_t2.append(np.max(roi_pixels))
            
            roi_pixels = pet_A3_roi
            if len(roi_pixels) == 0:
                print('Empty Cell at A3 PID: ', PID)
            elif PID != 5:
                print('Currently working on A3 PID: ', PID)
                mean_value_pet_t3.append(np.mean(roi_pixels))
                quantiles_pet_t3.append(np.percentile(roi_pixels, 10))
                coeff_var_pet_t3.append(np.std(roi_pixels) / np.mean(roi_pixels))
                skewness_pet_t3.append(stats.skew(roi_pixels))
                kurtosis_pet_t3.append(stats.kurtosis(roi_pixels))
                max_value_pet_t3.append(np.max(roi_pixels))
        
    else:
        PID = i - 12
        if PID != 9  and PID!=10:
            pet_B1_roi = ROI_extractor(pet_B1, t1_p2_mask_B, PID)
            pet_B2_roi = ROI_extractor(pet_B3, t2_p2_mask_B, PID)
            pet_B3_roi = ROI_extractor(pet_B3, t3_p2_mask_B, PID)
            
            roi_pixels = pet_B1_roi
            print('Currently working on B1 PID: ', PID)
            mean_value_pet_t1.append(np.mean(roi_pixels))
            quantiles_pet_t1.append(np.percentile(roi_pixels, 10))
            coeff_var_pet_t1.append(np.std(roi_pixels) / np.mean(roi_pixels))
            skewness_pet_t1.append(stats.skew(roi_pixels))
            kurtosis_pet_t1.append(stats.kurtosis(roi_pixels))
            max_value_pet_t1.append(np.max(roi_pixels))
            
            roi_pixels = pet_B2_roi
    
            print('Currently working on B2 PID: ', PID)
            mean_value_pet_t2.append(np.mean(roi_pixels))
            quantiles_pet_t2.append(np.percentile(roi_pixels, 10))
            coeff_var_pet_t2.append(np.std(roi_pixels) / np.mean(roi_pixels))
            skewness_pet_t2.append(stats.skew(roi_pixels))
            kurtosis_pet_t2.append(stats.kurtosis(roi_pixels))
            max_value_pet_t2.append(np.max(roi_pixels))
    
            roi_pixels = pet_B3_roi
    
            print('Currently working on B3 PID: ', PID)
            mean_value_pet_t3.append(np.mean(roi_pixels))
            quantiles_pet_t3.append(np.percentile(roi_pixels, 10))
            coeff_var_pet_t3.append(np.std(roi_pixels) / np.mean(roi_pixels))
            skewness_pet_t3.append(stats.skew(roi_pixels))
            kurtosis_pet_t3.append(stats.kurtosis(roi_pixels))
            max_value_pet_t3.append(np.max(roi_pixels))

###############################################################################

plt.figure()
plt.boxplot([mean_value_pet_t1, mean_value_pet_t2, mean_value_pet_t3])

# Add labels for each dataset
plt.xticks([1, 2, 3], ['Time 1', 'Time 2', 'Time 3'])

# Add a title and labels
plt.title('Box-and-Whisker Plot')
plt.xlabel('PET Datasets')
plt.ylabel('Mean Values')

# Show the plot
plt.show()

##############################################################################

plt.figure()
plt.boxplot([quantiles_pet_t1, quantiles_pet_t2, quantiles_pet_t3])

# Add labels for each dataset
plt.xticks([1, 2, 3], ['Time 1', 'Time 2', 'Time 3'])

# Add a title and labels
plt.title('Box-and-Whisker Plot')
plt.xlabel('PET Datasets')
plt.ylabel('Quantiles Values')

# Show the plot
plt.show()

##############################################################################

plt.figure()
plt.boxplot([coeff_var_pet_t1, coeff_var_pet_t2, coeff_var_pet_t3])

# Add labels for each dataset
plt.xticks([1, 2, 3], ['Time 1', 'Time 2', 'Time 3'])

# Add a title and labels
plt.title('Box-and-Whisker Plot')
plt.xlabel('PET Datasets')
plt.ylabel('Coeff Var Values')

# Show the plot
plt.show()

##############################################################################

plt.figure()
plt.boxplot([skewness_pet_t1, skewness_pet_t2, skewness_pet_t3])

# Add labels for each dataset
plt.xticks([1, 2, 3], ['Time 1', 'Time 2', 'Time 3'])

# Add a title and labels
plt.title('Box-and-Whisker Plot')
plt.xlabel('PET Datasets')
plt.ylabel('Skewness Values')

# Show the plot
plt.show()

##############################################################################

plt.figure()
plt.boxplot([kurtosis_pet_t1, kurtosis_pet_t2, kurtosis_pet_t3])

# Add labels for each dataset
plt.xticks([1, 2, 3], ['Time 1', 'Time 2', 'Time 3'])

# Add a title and labels
plt.title('Box-and-Whisker Plot')
plt.xlabel('PET Datasets')
plt.ylabel('Kurtosis Values')

# Show the plot
plt.show()

###############################################################################


#%% Normalize the features

mean_value_pet_t1_norm = (mean_value_pet_t1 - np.min(mean_value_pet_t1)) / (np.max(mean_value_pet_t1) - np.min(mean_value_pet_t1))
mean_value_pet_t2_norm = (mean_value_pet_t2 - np.min(mean_value_pet_t2)) / (np.max(mean_value_pet_t2) - np.min(mean_value_pet_t2))
mean_value_pet_t3_norm = (mean_value_pet_t3 - np.min(mean_value_pet_t3)) / (np.max(mean_value_pet_t3) - np.min(mean_value_pet_t3))

quantiles_pet_t1_norm = (quantiles_pet_t1 - np.min(quantiles_pet_t1)) / (np.max(quantiles_pet_t1) - np.min(quantiles_pet_t1))
quantiles_pet_t2_norm = (quantiles_pet_t2 - np.min(quantiles_pet_t2)) / (np.max(quantiles_pet_t2) - np.min(quantiles_pet_t2))
quantiles_pet_t3_norm = (quantiles_pet_t3 - np.min(quantiles_pet_t3)) / (np.max(quantiles_pet_t3) - np.min(quantiles_pet_t3))
        
coeff_var_pet_t1_norm = (coeff_var_pet_t1 - np.min(coeff_var_pet_t1)) / (np.max(coeff_var_pet_t1) - np.min(coeff_var_pet_t1))
coeff_var_pet_t2_norm = (coeff_var_pet_t2 - np.min(coeff_var_pet_t2)) / (np.max(coeff_var_pet_t2) - np.min(coeff_var_pet_t2))
coeff_var_pet_t3_norm = (coeff_var_pet_t3 - np.min(coeff_var_pet_t3)) / (np.max(coeff_var_pet_t3) - np.min(coeff_var_pet_t3))

skewness_pet_t1_norm = (skewness_pet_t1 - np.min(skewness_pet_t1)) / (np.max(skewness_pet_t1) - np.min(skewness_pet_t1))
skewness_pet_t2_norm = (skewness_pet_t2 - np.min(skewness_pet_t2)) / (np.max(skewness_pet_t2) - np.min(skewness_pet_t2))
skewness_pet_t3_norm = (skewness_pet_t3 - np.min(skewness_pet_t3)) / (np.max(skewness_pet_t3) - np.min(skewness_pet_t3))

kurtosis_pet_t1_norm = (kurtosis_pet_t1 - np.min(kurtosis_pet_t1)) / (np.max(kurtosis_pet_t1) - np.min(kurtosis_pet_t1))
kurtosis_pet_t2_norm = (kurtosis_pet_t2 - np.min(kurtosis_pet_t2)) / (np.max(kurtosis_pet_t2) - np.min(kurtosis_pet_t2))
kurtosis_pet_t3_norm = (kurtosis_pet_t3 - np.min(kurtosis_pet_t3)) / (np.max(kurtosis_pet_t3) - np.min(kurtosis_pet_t3))

max_value_pet_t1_norm = (max_value_pet_t1 - np.min(max_value_pet_t1)) / (np.max(max_value_pet_t1) - np.min(max_value_pet_t1))
max_value_pet_t2_norm = (max_value_pet_t2 - np.min(max_value_pet_t2)) / (np.max(max_value_pet_t2) - np.min(max_value_pet_t2))
max_value_pet_t3_norm = (max_value_pet_t3 - np.min(max_value_pet_t3)) / (np.max(max_value_pet_t3) - np.min(max_value_pet_t3))

#%% Next try with Skillings-Mack test

# The scipy does not have skillings-mack module, so use friedman test method instead
# An example can be referred to this link: https://www.statology.org/friedman-test-python/

# Analyzing statistically significant difference between statistical features at different time mark
# Note: To conduct Friedman Test 

group1 = max_value_adc_t1
group2 = max_value_adc_t2
group3 = max_value_adc_t3

stats.friedmanchisquare(group1, group2, group3)

#%%
group1 = max_value_dce_t1
group2 = max_value_dce_t2
group3 = max_value_dce_t3

stats.friedmanchisquare(group1, group2, group3)

#%%
group1 = max_value_pet_t1
group2 = max_value_pet_t2
group3 = max_value_pet_t3

stats.friedmanchisquare(group1, group2, group3)

#%% Statistical Analysis

Responder = [1,1,0,1,1,1,0,0,1,1,0,0,0,0,1,0,1,1,0,0,1,1,1]
Responder_reduce = np.array([1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,1,1,0,0,1])

from scipy.stats import linregress

mean_adc_t1_r      = mean_value_adc_t1_norm
mean_adc_t2_r      = mean_value_adc_t2_norm
mean_adc_t3_r      = mean_value_adc_t3_norm
quantiles_adc_t1_r = quantiles_adc_t1_norm
quantiles_adc_t2_r = quantiles_adc_t2_norm
quantiles_adc_t3_r = quantiles_adc_t3_norm
coeff_var_adc_t1_r = coeff_var_adc_t1_norm
coeff_var_adc_t2_r = coeff_var_adc_t2_norm
coeff_var_adc_t3_r = coeff_var_adc_t3_norm
skewness_adc_t1_r  = skewness_adc_t1_norm
skewness_adc_t2_r  = skewness_adc_t2_norm
skewness_adc_t3_r  = skewness_adc_t3_norm
kurtosis_adc_t1_r  = kurtosis_adc_t1_norm
kurtosis_adc_t2_r  = kurtosis_adc_t2_norm
kurtosis_adc_t3_r  = kurtosis_adc_t3_norm

mean_dce_t1_r      = mean_value_dce_t1_norm
mean_dce_t2_r      = mean_value_dce_t2_norm
mean_dce_t3_r      = mean_value_dce_t3_norm
quantiles_dce_t1_r = quantiles_dce_t1_norm
quantiles_dce_t2_r = quantiles_dce_t2_norm
quantiles_dce_t3_r = quantiles_dce_t3_norm
coeff_var_dce_t1_r = coeff_var_dce_t1_norm
coeff_var_dce_t2_r = coeff_var_dce_t2_norm
coeff_var_dce_t3_r = coeff_var_dce_t3_norm
skewness_dce_t1_r  = skewness_dce_t1_norm
skewness_dce_t2_r  = skewness_dce_t2_norm
skewness_dce_t3_r  = skewness_dce_t3_norm
kurtosis_dce_t1_r  = kurtosis_dce_t1_norm
kurtosis_dce_t2_r  = kurtosis_dce_t2_norm
kurtosis_dce_t3_r  = kurtosis_dce_t3_norm

mean_pet_t1_r      = mean_value_pet_t1_norm
mean_pet_t2_r      = mean_value_pet_t2_norm
mean_pet_t3_r      = mean_value_pet_t3_norm
quantiles_pet_t1_r = quantiles_pet_t1_norm
quantiles_pet_t2_r = quantiles_pet_t2_norm
quantiles_pet_t3_r = quantiles_pet_t3_norm
coeff_var_pet_t1_r = coeff_var_pet_t1_norm
coeff_var_pet_t2_r = coeff_var_pet_t2_norm
coeff_var_pet_t3_r = coeff_var_pet_t3_norm
skewness_pet_t1_r  = skewness_pet_t1_norm
skewness_pet_t2_r  = skewness_pet_t2_norm
skewness_pet_t3_r  = skewness_pet_t3_norm
kurtosis_pet_t1_r  = kurtosis_pet_t1_norm
kurtosis_pet_t2_r  = kurtosis_pet_t2_norm
kurtosis_pet_t3_r  = kurtosis_pet_t3_norm

X_adc = np.column_stack((mean_adc_t1_r,mean_adc_t2_r,mean_adc_t3_r,quantiles_adc_t1_r,quantiles_adc_t2_r,quantiles_adc_t3_r,coeff_var_adc_t1_r,coeff_var_adc_t2_r,coeff_var_adc_t3_r,skewness_adc_t1_r,skewness_adc_t2_r,skewness_adc_t3_r,kurtosis_adc_t1_r,kurtosis_adc_t2_r,kurtosis_adc_t3_r,max_value_adc_t1,max_value_adc_t2,max_value_adc_t3))
X_dce = np.column_stack((mean_dce_t1_r,mean_dce_t2_r,mean_dce_t3_r,quantiles_dce_t1_r,quantiles_dce_t2_r,quantiles_dce_t3_r,coeff_var_dce_t1_r,coeff_var_dce_t2_r,coeff_var_dce_t3_r,skewness_dce_t1_r,skewness_dce_t2_r,skewness_dce_t3_r,kurtosis_dce_t1_r,kurtosis_dce_t2_r,kurtosis_dce_t3_r,max_value_dce_t1,max_value_dce_t2,max_value_dce_t3))
X_pet = np.column_stack((mean_pet_t1_r,mean_pet_t2_r,mean_pet_t3_r,quantiles_pet_t1_r,quantiles_pet_t2_r,quantiles_pet_t3_r,coeff_var_pet_t1_r,coeff_var_pet_t2_r,coeff_var_pet_t3_r,skewness_pet_t1_r,skewness_pet_t2_r,skewness_pet_t3_r,kurtosis_pet_t1_r,kurtosis_pet_t2_r,kurtosis_pet_t3_r,max_value_pet_t1,max_value_pet_t2,max_value_pet_t3))

#%%
X = np.column_stack((X_adc, X_dce, X_pet))

# X_adc = np.transpose(X_adc)
# X_dce = np.transpose(X_dce)
# X_pet = np.transpose(X_pet)

# X = np.column_stack((X_adc, X_dce[:,:19], X_pet[:,:19]))

### Try to randomly pick some samples
# np.random.seed(0)

# num_rows_to_select = 10

# indices = np.random.choice(19, num_rows_to_select, replace=False)
# select_rows = X[indices]

def rand_select(mtx, label, num):
    np.random.seed(0)
    num_rows_to_select = num
    indices = np.random.choice(19, num_rows_to_select, replace=False)
    select_rows  = mtx[indices]
    print(select_rows.shape)
    select_label = label[indices]
    print(select_label.shape)
    
    return select_rows, select_label

#%% Set up regression analysis

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

Y = Responder_reduce
num_features = 10

model = LinearRegression()
selector = RFE(model, n_features_to_select=num_features)
selector = selector.fit(X, Y)

selected_features = X[:,selector.support_]
model.fit(selected_features, Y)

#%% Get the results

# r_sq = model.score(X, Y)

# print(f"coefficient of determination: {r_sq}")

# print(f"intercept: {model.intercept_}")

# print(f"slope: {model.coef_}")

# y_pred = model.predict(X)
# print(f"predicted response:\n{y_pred}")

#%% Let's move to Matlab for a comprehensive analysis

# np.savetxt('parameters.csv', X, delimiter=',')
# np.savetxt('label.csv', Y, delimiter=',')

#%% Now let's try with packages giving the p-value

import statsmodels.api as sm

X_train = sm.add_constant(X)
y_train = Responder_reduce
# Fit the regression model
model = sm.OLS(y_train, X_train)
results = model.fit()

print(results.summary())

#%% Let's test for sample variance

group_1 = []
group_0 = []

for i in range(len(Y)):
    if Y[i] == 1:
        group_1.append(X[i,:])
    elif Y[i] == 0:
        group_0.append(X[i,:])
        
group_1 = np.array(group_1)
group_0 = np.array(group_0)     

## Compute Sample Variance
group_1_var = []
group_0_var = []

for j in range(45):
    group_1_var.append(np.var(group_1[:,j], ddof=1))
    group_0_var.append(np.var(group_0[:,j], ddof=1))

#%% Let's further test sample variance after min-max normalization

X_norm = np.zeros((X.shape[0],X.shape[1]))

for k in range(X.shape[1]):
    X_norm[:,k] = (X[:,k] - np.min(X[:,k])) / (np.max(X[:,k]) - np.min(X[:,k]))

group_1_norm = []
group_0_norm = []

for i in range(len(Y)):
    if Y[i] == 1:
        group_1_norm.append(X_norm[i,:])
    elif Y[i] == 0:
        group_0_norm.append(X_norm[i,:])
        
group_1_norm = np.array(group_1_norm)
group_0_norm = np.array(group_0_norm)     

## Compute Sample Variance
group_1_norm_var = []
group_0_norm_var = []

for j in range(45):
    group_1_norm_var.append(np.var(group_1_norm[:,j], ddof=1))
    group_0_norm_var.append(np.var(group_0_norm[:,j], ddof=1))
    
#%% Next, let's remove the linearly correslated parameters

X_adc_r = np.column_stack((mean_adc_t1_r,quantiles_adc_t1_r,quantiles_adc_t2_r,quantiles_adc_t3_r,coeff_var_adc_t1_r,coeff_var_adc_t2_r,coeff_var_adc_t3_r,skewness_adc_t1_r,skewness_adc_t2_r,skewness_adc_t3_r,kurtosis_adc_t1_r,kurtosis_adc_t2_r,kurtosis_adc_t3_r))
X_dce_r = np.column_stack((mean_dce_t1_r,quantiles_dce_t1_r,quantiles_dce_t2_r,quantiles_dce_t3_r,coeff_var_dce_t1_r,coeff_var_dce_t2_r,coeff_var_dce_t3_r,skewness_dce_t1_r,skewness_dce_t2_r,skewness_dce_t3_r,kurtosis_dce_t1_r,kurtosis_dce_t2_r,kurtosis_dce_t3_r))
X_pet_r = np.column_stack((mean_pet_t1_r,quantiles_pet_t1_r,quantiles_pet_t2_r,quantiles_pet_t3_r,coeff_var_pet_t1_r,coeff_var_pet_t2_r,coeff_var_pet_t3_r,skewness_pet_t1_r,skewness_pet_t2_r,skewness_pet_t3_r,kurtosis_pet_t1_r,kurtosis_pet_t2_r,kurtosis_pet_t3_r))

# X_adc_r = np.column_stack((quantiles_adc_t1_r,quantiles_adc_t2_r,quantiles_adc_t3_r,coeff_var_adc_t1_r,coeff_var_adc_t2_r,coeff_var_adc_t3_r,skewness_adc_t1_r,skewness_adc_t2_r,skewness_adc_t3_r,kurtosis_adc_t1_r,kurtosis_adc_t2_r,kurtosis_adc_t3_r))
# X_dce_r = np.column_stack((quantiles_dce_t1_r,quantiles_dce_t2_r,quantiles_dce_t3_r,coeff_var_dce_t1_r,coeff_var_dce_t2_r,coeff_var_dce_t3_r,skewness_dce_t1_r,skewness_dce_t2_r,skewness_dce_t3_r,kurtosis_dce_t1_r,kurtosis_dce_t2_r,kurtosis_dce_t3_r))
# X_pet_r = np.column_stack((quantiles_pet_t1_r,quantiles_pet_t2_r,quantiles_pet_t3_r,coeff_var_pet_t1_r,coeff_var_pet_t2_r,coeff_var_pet_t3_r,skewness_pet_t1_r,skewness_pet_t2_r,skewness_pet_t3_r,kurtosis_pet_t1_r,kurtosis_pet_t2_r,kurtosis_pet_t3_r))

# X_adc_r = np.column_stack((mean_adc_t1_r,mean_adc_t3_r,skewness_adc_t1_r,skewness_adc_t2_r,skewness_adc_t3_r))
# X_dce_r = np.column_stack((mean_dce_t1_r,skewness_dce_t1_r,skewness_dce_t3_r))
# X_pet_r = np.column_stack((mean_pet_t1_r,mean_pet_t2_r,skewness_pet_t1_r,skewness_pet_t2_r))

# X_adc_r = np.column_stack((mean_adc_t1_r,mean_adc_t2_r,mean_adc_t3_r,skewness_adc_t1_r,skewness_adc_t2_r,skewness_adc_t3_r,max_value_adc_t1,max_value_adc_t2,max_value_adc_t3))
# X_dce_r = np.column_stack((mean_dce_t1_r,mean_dce_t2_r,mean_dce_t3_r,skewness_dce_t1_r,skewness_dce_t2_r,skewness_dce_t3_r,max_value_dce_t1,max_value_dce_t2,max_value_dce_t3))
# X_pet_r = np.column_stack((mean_pet_t1_r,mean_pet_t2_r,mean_pet_t3_r,skewness_pet_t1_r,skewness_pet_t2_r,skewness_pet_t3_r,max_value_pet_t1,max_value_pet_t2,max_value_pet_t3))


# X_adc_r = np.column_stack((mean_adc_t1_r,mean_adc_t2_r,mean_adc_t3_r,quantiles_adc_t1_r,quantiles_adc_t2_r,quantiles_adc_t3_r,coeff_var_adc_t1_r,coeff_var_adc_t2_r,coeff_var_adc_t3_r,skewness_adc_t1_r,skewness_adc_t2_r,skewness_adc_t3_r,kurtosis_adc_t1_r,kurtosis_adc_t2_r,kurtosis_adc_t3_r))
# X_dce_r = np.column_stack((mean_dce_t1_r,mean_dce_t2_r,mean_dce_t3_r,quantiles_dce_t1_r,quantiles_dce_t2_r,quantiles_dce_t3_r,coeff_var_dce_t1_r,coeff_var_dce_t2_r,coeff_var_dce_t3_r,skewness_dce_t1_r,skewness_dce_t2_r,skewness_dce_t3_r,kurtosis_dce_t1_r,kurtosis_dce_t2_r,kurtosis_dce_t3_r))
# X_pet_r = np.column_stack((mean_pet_t1_r,mean_pet_t2_r,mean_pet_t3_r,quantiles_pet_t1_r,quantiles_pet_t2_r,quantiles_pet_t3_r,coeff_var_pet_t1_r,coeff_var_pet_t2_r,coeff_var_pet_t3_r,skewness_pet_t1_r,skewness_pet_t2_r,skewness_pet_t3_r,kurtosis_pet_t1_r,kurtosis_pet_t2_r,kurtosis_pet_t3_r))

# X_adc_r = np.column_stack((quantiles_adc_t1_r,quantiles_adc_t2_r,quantiles_adc_t3_r,coeff_var_adc_t1_r))
# X_dce_r = np.column_stack((quantiles_dce_t1_r,quantiles_dce_t2_r,quantiles_dce_t3_r,coeff_var_dce_t1_r,coeff_var_dce_t2_r,coeff_var_dce_t3_r))
# X_pet_r = np.column_stack((quantiles_pet_t1_r,quantiles_pet_t2_r,quantiles_pet_t3_r,coeff_var_pet_t1_r,coeff_var_pet_t2_r,coeff_var_pet_t3_r))

# X_adc_r = np.column_stack((mean_adc_t1_r,mean_adc_t2_r,mean_adc_t3_r,quantiles_adc_t1_r,quantiles_adc_t2_r,quantiles_adc_t3_r,skewness_adc_t1_r,skewness_adc_t2_r,skewness_adc_t3_r))
# X_dce_r = np.column_stack((mean_dce_t1_r,mean_dce_t2_r,mean_dce_t3_r,coeff_var_dce_t1_r,coeff_var_dce_t2_r,skewness_dce_t1_r,skewness_dce_t2_r,skewness_dce_t3_r))
# X_pet_r = np.column_stack((mean_pet_t1_r,mean_pet_t2_r,mean_pet_t3_r,skewness_pet_t1_r,skewness_pet_t2_r,skewness_pet_t3_r))

X_r = np.column_stack((X_adc_r, X_dce_r, X_pet_r))

X_r_norm = np.zeros((X_r.shape[0],X_r.shape[1]))

for k in range(X_r.shape[1]):
    X_r_norm[:,k] = (X_r[:,k] - np.min(X_r[:,k])) / (np.max(X_r[:,k]) - np.min(X_r[:,k]))

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

Y = Responder_reduce
num_features = 10

model = LinearRegression()
selector = RFE(model, n_features_to_select=num_features)
selector = selector.fit(X_r, Y)

selected_features = X_r[:,selector.support_]
model.fit(selected_features, Y)

#%%
import statsmodels.api as sm

X_train = sm.add_constant(X_r_norm)
y_train = Responder_reduce

X_train, y_train = rand_select(X_train, y_train, 14)

# Fit the regression model
model = sm.OLS(y_train, X_train)
results = model.fit()

print(results.summary())

#%% Feature Significance Analysis

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind

# Step 1: Compute P-values
p_values = []

for feature_idx in range(X_train.shape[1]):
    # Use t-test to compute p-value for the feature
    _, p_value = ttest_ind(X_train[y_train==0, feature_idx], X_train[y_train==1, feature_idx])
    p_values.append((feature_idx, p_value))

# Rank features based on p-values
p_values.sort(key=lambda x: x[1])

# Step 2: Compute AUC Scores
auc_scores = []

for feature_idx in range(X_train.shape[1]):
    # Train a logistic regression model on the current feature
    model = LogisticRegression()
    model.fit(X_train[:, feature_idx].reshape(-1, 1), y_train)
    
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_train[:, feature_idx].reshape(-1, 1))[:, 1]
    
    # Compute AUC
    auc = roc_auc_score(y_train, y_pred_proba)
    auc_scores.append((feature_idx, auc))

# Rank features based on AUC
auc_scores.sort(key=lambda x: x[1], reverse=True)

#%% Perform Mann-Whitney U test

import numpy as np
from scipy.stats import mannwhitneyu

# Generate two independent samples (replace these with your actual data)
sample1 = np.transpose(X_adc)
sample2 = np.transpose(X_dce)
sample3 = np.transpose(X_pet)

# Perform Mann-Whitney U test
statistic, p_value = mannwhitneyu(sample1[12,:], sample2[12,:], alternative='two-sided')

# Print the test statistic and p-value
print(f"Test Statistic: {statistic}")
print(f"P-value: {p_value}")

# Interpret the result
alpha = 0.05
if p_value < alpha:
    print("Reject null hypothesis: There is a significant difference between the samples.")
else:
    print("Fail to reject null hypothesis: There is no significant difference between the samples.")

#%% Compute the Results

mean_pet_1 = np.array(mean_value_pet_t1)
mean_pet_2 = np.array(mean_value_pet_t2)
mean_pet_3 = np.array(mean_value_pet_t3)

# print((mean_pet_2-mean_pet_1)/mean_pet_1 * 100)
# print((mean_pet_3-mean_pet_1)/mean_pet_1 * 100)

#%% Conduct Mann-Whitney U test among all features!

confusion_matrix = np.zeros((18,18))

for i in range(18):
    for j in range(18):
        
        statistic, p_value = mannwhitneyu(sample3[i,:], sample3[j,:], alternative='two-sided')
        confusion_matrix[i][j] = p_value
        
        
#%% Making an image of the 3d slices

# import cv2

# iidx = 0

# images = dce_A1[0]
# # pet_A1_new = pet_A1 * t1_p2_mask_A
# images_cut = dce_A1[0] * t1_p2_mask_A[0]

# import imageio
# volume = images

# # Assuming volume is a 3D numpy array
# images1 = [volume[:,:,i] for i in range(volume.shape[2])]

# # Save the list of images as a GIF
# imageio.mimsave('output_pet.gif', images1, duration=0.5)

# volume = images_cut

# # Assuming volume is a 3D numpy array
# images2 = [volume[:,:,i] for i in range(volume.shape[2])]

# # Save the list of images as a GIF
# imageio.mimsave('output_pet_cut.gif', images2, duration=0.5)

#%% Step 2: Check generalization with k-fold cross-validation

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

Y = Responder_reduce
num_features = 18

model = LogisticRegression()
selector = RFE(model, n_features_to_select=num_features)
selector = selector.fit(X_r, Y)

selected_features = X_r[:,selector.support_]
model.fit(selected_features, Y)

r_sq = model.score(selected_features, Y)

# print(f"coefficient of determination: {r_sq}")

# print(f"intercept: {model.intercept_}")

# print(f"slope: {model.coef_}")

y_pred = np.round(model.predict(selected_features))

print('True Label: ', Y)
print(f"predicted response:\n{y_pred}")

# Define the number of splits (k)
k = 5

# Initialize the KFold object
kf = KFold(n_splits=k)

# Initialize the model
model = LogisticRegression()

X = selected_features
Y = Responder_reduce

# Lists to store evaluation metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Perform k-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = selected_features[train_index], selected_features[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Evaluate the model on the test data
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

# Calculate average evaluation metrics
average_accuracy  = np.mean(accuracy_scores)
average_precision = np.mean(precision_scores)
average_recall    = np.mean(recall_scores)
average_f1        = np.mean(f1_scores)

# Print results
print(f'Average Accuracy: {average_accuracy}')
print(f'Average Precision: {average_precision}')
print(f'Average Recall: {average_recall}')
print(f'Average F1-score: {average_f1}')

my_list = [average_accuracy, average_precision, average_recall, average_f1]

