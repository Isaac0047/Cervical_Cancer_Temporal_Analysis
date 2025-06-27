#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 22:21:53 2023

@author: Haotian Feng, Radiation Oncology, UCSF
"""

# This code loads the cervical cancer image data and do preprocessing to automatically extract the ROI

#%% Load important modules

import SimpleITK as sitk
import os
import tensorflow as tf
import pandas as pd
import time
import numpy as np

from matplotlib import pyplot as plt
from dltk.io.augmentation import *
from dltk.io.preprocessing import *

import pydicom as di
from rt_utils import RTStructBuilder

#%% Load the data (read the dicom files)

fp = '/Users/hafeng/Documents/Research_Data/Cervical_Cancer/manifest-1655581046477/CC-Tumor-Heterogeneity/CCTH-A01/03-06-2005-NA-MR1-64986/4.000000-SAG T2-37825/'
fn = fp + '1-10.dcm'

ds = di.filereader.dcmread(fn)

im = di.dcmread(fn)
plt.imshow(im.pixel_array, cmap=plt.cm.bone)

#%% Load the CT Scan data

ct_pth   = '/Users/hafeng/Documents/Research_Data/Cervical_Cancer/manifest-1655581046477/CC-Tumor-Heterogeneity/CCTH-A01/03-06-2005-NA-MR1-64986/4.000000-SAG T2-37825/'

for i in range(30):
    
    if i+1<10:
        idd = '0' + str(i+1)
    else:
        idd = str(i+1)
    
    nm = '1-' + idd + '.dcm'
    fn = ct_pth + nm

    ds = di.filereader.dcmread(fn)

    im = di.dcmread(fn)
    
    plt.figure()
    plt.imshow(im.pixel_array, cmap=plt.cm.bone)
    plt.title('Image Name of Sample ' + idd + ' file')
    
#%% Load data with rt_utils

cont_pth = '/Users/hafeng/Documents/Research_Data/Cervical_Cancer/manifest-1655581046477/CC-Tumor-Heterogeneity/CCTH-A01/03-06-2005-NA-MR1-64986/1.000000-ROI-34265/1-1.dcm'
rtstruct = RTStructBuilder.create_from(dicom_series_path=ct_pth, rt_struct_path=cont_pth)

rois = rtstruct.get_roi_names()

mask_3ds = []
for i in range(len(rois)):
    mask_3d = rtstruct.get_roi_mask_by_name(rois[i])
    mask_3ds.append(np.flip(mask_3d, axis=-1))
    
#%% Plotting the data

fig = plt.figure()
# ax  = fig.add_subplot(projection='3d')

volume = mask_3ds[0]

# for x in range(len(volume[:, 0, 0])):
#     for y in range(len(volume[0, :, 0])):
#         for z in range(len(volume[0, 0, :])):
            
#             ax.scatter(x, y, z, c = tuple([volume[x, y, z], volume[x, y, z], volume[x, y, z], 1]))

# ax.set(xlabel='r', ylabel='g', zlabel='b')
# ax.set_aspect('equal')
# plt.show()

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region')

#%% Search for all potential files (Directory stored as final_dir1, dir2 and dir3)

import glob

home_pth = '/Users/hafeng/Documents/Research_Data/Cervical_Cancer/manifest-1655581046477/CC-Tumor-Heterogeneity/'

final_dir1 = []
final_dir2 = []
final_dir3 = []

check_pth  = []

for ii in range(23):
    
    if ii+1 < 6:
        curr_pth = home_pth + 'CCTH-A0' + str(ii+1) + '/'
    elif ii+1 > 6  and ii+1 < 10:
        curr_pth = home_pth + 'CCTH-A0' + str(ii+1) + '/'
    elif ii+1>=10 and ii+1<13:
        curr_pth = home_pth + 'CCTH-A'  + str(ii+1) + '/'
    elif ii+1>=13 and ii+1<22:
        curr_pth = home_pth + 'CCTH-B0' + str(ii+1-12) + '/'
    elif ii+1>=22:
        curr_pth = home_pth + 'CCTH-B'  + str(ii+1-12) + '/'
    else:
        continue
    
    # Method 1: By listing all subfolders (works)
    directory = os.listdir(curr_pth)
    check_pth.append(curr_pth)
    
    #for foname in directory:
    #    print(foname)
    
    # Method 2: By matching exact words (works)
    os.chdir(curr_pth)
    foldname1 = glob.glob('*MR1*')
    foldname2 = glob.glob('*MR2*')    
    foldname3 = glob.glob('*MR3*')    
    
    # Check through each folders to see if ROI folder exists
    
    print('Checking the subfolder name')
    for subfold in foldname1:
        print(subfold)
        os.chdir(curr_pth + '/' + subfold + '/')
        
        filecheck = glob.glob('*ROI*')
        print(filecheck)
        
        if len(filecheck) != 0:
            final_dir1.append(curr_pth + '/' + subfold + '/')
        else:
            continue
    
    print('Checking the subfolder name')
    for subfold in foldname2:
        print(subfold)
        os.chdir(curr_pth + '/' + subfold + '/')
        
        filecheck = glob.glob('*ROI*')
        print(filecheck)
        
        if len(filecheck) != 0:
            final_dir2.append(curr_pth + '/' + subfold + '/')
        else:
            continue
    
    print('Checking the subfolder name')
    for subfold in foldname3:
        print(subfold)
        os.chdir(curr_pth + '/' + subfold + '/')
        
        filecheck = glob.glob('*ROI*')
        print(filecheck)
        
        if len(filecheck) != 0:
            final_dir3.append(curr_pth + '/' + subfold + '/')
        else:
            continue
        
print(len(final_dir1))
print(len(final_dir2))
print(len(final_dir3))

#%% Store the model matrix for 3 MRI periods

# The first period
mask_3ds_p1 = []
mask_3ds_p1_c  = []
mask_3ds_p1_p1 = []
mask_3ds_p1_p2 = []

for hm_pth in final_dir1:
    os.chdir(hm_pth)
    cont_fold = glob.glob('*ROI*')
    cont_pth  = hm_pth + cont_fold[0] + '/1-1.dcm'
    
    if glob.glob('*SAG T2*'):
        print('SAG T2!')
        ct_fold = glob.glob('*SAG T2*')
        ct_pth  = hm_pth + ct_fold[0] + '/'
    elif glob.glob('*Sag T2*'):
        print('Sag T2!')
        ct_fold = glob.glob('*Sag T2*')
        ct_pth  = hm_pth + ct_fold[0] + '/'
    else:
        print(hm_pth + ' cannot locate the ct files')
        break
    
    rtstruct = RTStructBuilder.create_from(dicom_series_path=ct_pth, rt_struct_path=cont_pth)

    rois = rtstruct.get_roi_names()
    
    for i in range(len(rois)):
        mask_3d = rtstruct.get_roi_mask_by_name(rois[i])
        mask_3ds_p1.append(mask_3d)
        #mask_3ds_p1.append(np.flip(mask_3d, axis=-1))
        
        if i==0:
            mask_3ds_p1_p1.append(mask_3d)
        elif i==1:
            mask_3ds_p1_p2.append(mask_3d)
            
        if i%2==0:
            tmp = mask_3d
        elif i%2==1:
            tmp = tmp + mask_3d
            mask_3ds_p1_c.append(tmp)
                    
# The second period 
mask_3ds_p2 = []
mask_3ds_p2_c  = []
mask_3ds_p2_p1 = []
mask_3ds_p2_p2 = []

for hm_pth in final_dir2:
    os.chdir(hm_pth)
    cont_fold = glob.glob('*ROI*')
    cont_pth  = hm_pth + cont_fold[0] + '/1-1.dcm'
    
    if glob.glob('*SAG T2*'):
        print('SAG T2!')
        ct_fold = glob.glob('*SAG T2*')
        ct_pth  = hm_pth + ct_fold[0] + '/'
    elif glob.glob('*Sag T2*'):
        print('Sag T2!')
        ct_fold = glob.glob('*Sag T2*')
        ct_pth  = hm_pth + ct_fold[0] + '/'
    else:
        print(hm_pth + 'cannot locate the ct files')
        break
    
    rtstruct = RTStructBuilder.create_from(dicom_series_path=ct_pth, rt_struct_path=cont_pth)

    rois = rtstruct.get_roi_names()
    
    for i in range(len(rois)):
        mask_3d = rtstruct.get_roi_mask_by_name(rois[i])
        mask_3ds_p2.append(mask_3d)
        #mask_3ds_p1.append(np.flip(mask_3d, axis=-1))
        
        if i==0:
            mask_3ds_p2_p1.append(mask_3d)
        elif i==1:
            mask_3ds_p2_p2.append(mask_3d)
            
        if i%2==0:
            tmp = mask_3d
        elif i%2==1:
            tmp = tmp + mask_3d
            mask_3ds_p2_c.append(tmp)
 
# The third period            
mask_3ds_p3 = []
mask_3ds_p3_c  = []
mask_3ds_p3_p1 = []
mask_3ds_p3_p2 = []

for hm_pth in final_dir3:
    os.chdir(hm_pth)
    cont_fold = glob.glob('*ROI*')
    cont_pth  = hm_pth + cont_fold[0] + '/1-1.dcm'
    
    if glob.glob('*SAG T2*'):
        print('SAG T2!')
        ct_fold = glob.glob('*SAG T2*')
        ct_pth  = hm_pth + ct_fold[0] + '/'
    elif glob.glob('*Sag T2*'):
        print('Sag T2!')
        ct_fold = glob.glob('*Sag T2*')
        ct_pth  = hm_pth + ct_fold[0] + '/'
    else:
        if glob.glob('5.000000-Ax T2 TSER*'):
            print('Ax T2 TSER!')
            ct_fold = glob.glob('5.000000-Ax T2 TSER*')
            ct_pth  = hm_pth + ct_fold[0] + '/'
            
        else:
            print(hm_pth + ' cannot locate the ct files')
            break
    
    rtstruct = RTStructBuilder.create_from(dicom_series_path=ct_pth, rt_struct_path=cont_pth)

    rois = rtstruct.get_roi_names()
    
    for i in range(len(rois)):
        mask_3d = rtstruct.get_roi_mask_by_name(rois[i])
        mask_3ds_p3.append(mask_3d)
        
        if i==0:
            mask_3ds_p3_p1.append(mask_3d)
        elif i==1:
            mask_3ds_p3_p2.append(mask_3d)
        #mask_3ds_p1.append(np.flip(mask_3d, axis=-1))
        
        if i%2==0:
            tmp = mask_3d
        elif i%2==1:
            tmp = tmp + mask_3d
            mask_3ds_p3_c.append(tmp)

#%% Plotting some data for verification

fig = plt.figure()
volume = mask_3ds_p1_p1[0]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region 1 at Time 1')

fig = plt.figure()
volume = mask_3ds_p1_p2[0]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region 2 at Time 1')

fig = plt.figure()
volume = mask_3ds_p2_p1[0]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region 1 at Time 2')

fig = plt.figure()
volume = mask_3ds_p2_p2[0]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region 2 at Time 2')

fig = plt.figure()
volume = mask_3ds_p3_p1[0]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region 1 at Time 3')

fig = plt.figure()
volume = mask_3ds_p3_p2[0]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region 2 at Time 3')

#%% Plot the combined model Section 1

fig = plt.figure()
volume = mask_3ds_p1_c[5]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region #5 at Time 1')

fig = plt.figure()
volume = mask_3ds_p2_c[5]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region #5 at Time 2')

fig = plt.figure()
volume = mask_3ds_p3_c[5]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region #5 at Time 3')


fig = plt.figure()
volume = mask_3ds_p1_c[10]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region #10 at Time 1')

fig = plt.figure()
volume = mask_3ds_p2_c[10]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region #10 at Time 2')

fig = plt.figure()
volume = mask_3ds_p3_c[10]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region #10 at Time 3')

#%% Plot the combined model Section 2

fig = plt.figure()
volume = mask_3ds_p1_c[15]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region #15 at Time 1')

fig = plt.figure()
volume = mask_3ds_p2_c[15]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region #15 at Time 2')

fig = plt.figure()
volume = mask_3ds_p3_c[15]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region #15 at Time 3')


fig = plt.figure()
volume = mask_3ds_p1_c[20]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region #20 at Time 1')

fig = plt.figure()
volume = mask_3ds_p2_c[20]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region #20 at Time 2')

fig = plt.figure()
volume = mask_3ds_p3_c[20]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region #20 at Time 3')

#%% Check and Print out the size of each element

print('Shapes of the first period')
for jj in range(len(mask_3ds_p1)):
    print(mask_3ds_p1[jj].shape)
    
print('Shapes of the second period')
for jj in range(len(mask_3ds_p2)):
    print(mask_3ds_p2[jj].shape)
    
print('Shapes of the third period')
for jj in range(len(mask_3ds_p3)):
    print(mask_3ds_p3[jj].shape)
    
#%% Reshape all matrix into the same size (test code)

import scipy

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
new_img = resize(mask_3ds_p1[0],shape)

fig = plt.figure()
volume = mask_3ds_p1[0]

ax = plt.axes(projection='3d')
ax.voxels(volume)
plt.title('Extracted Region')

fig = plt.figure()

ax = plt.axes(projection='3d')
ax.voxels(new_img)
plt.title('Test Resized Region')

#%% Reshaping all matrices

mask_3ds_p1_resize = []
mask_3ds_p2_resize = []
mask_3ds_p3_resize = []

shape = [256,256,32]

for mask in mask_3ds_p1:
    mask_new = resize(mask, shape)
    mask_3ds_p1_resize.append(mask_new)
    
for mask in mask_3ds_p2:
    mask_new = resize(mask, shape)
    mask_3ds_p2_resize.append(mask_new)
    
for mask in mask_3ds_p3:
    mask_new = resize(mask, shape)
    mask_3ds_p3_resize.append(mask_new)
    

#%% Save the variables

np.save('mask_3ds_p1_resize.npy', mask_3ds_p1_resize)
np.save('mask_3ds_p2_resize.npy', mask_3ds_p2_resize)
np.save('mask_3ds_p3_resize.npy', mask_3ds_p3_resize)

#%% Reshaping the combined matrices

mask_3ds_p1_c_resize = []
mask_3ds_p2_c_resize = []
mask_3ds_p3_c_resize = []

shape = [256,256,32]

for mask in mask_3ds_p1_c:
    mask_new = resize(mask, shape)
    mask_3ds_p1_c_resize.append(mask_new)
    
for mask in mask_3ds_p2_c:
    mask_new = resize(mask, shape)
    mask_3ds_p2_c_resize.append(mask_new)
    
for mask in mask_3ds_p3_c:
    mask_new = resize(mask, shape)
    mask_3ds_p3_c_resize.append(mask_new)

#%% Save the combined variables

np.save('mask_3ds_p1_c_resize.npy', mask_3ds_p1_c_resize)
np.save('mask_3ds_p2_c_resize.npy', mask_3ds_p2_c_resize)
np.save('mask_3ds_p3_c_resize.npy', mask_3ds_p3_c_resize)



