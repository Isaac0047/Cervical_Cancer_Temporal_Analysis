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

#%% Load the data for analysis

### ADC FEATURES
with open('adc_T1_contrast_origin.pkl', 'rb') as file:
    adc_T1_contrast = pickle.load(file)
with open('adc_T2_contrast_origin.pkl', 'rb') as file:
    adc_T2_contrast = pickle.load(file)
with open('adc_T3_contrast_origin.pkl', 'rb') as file:
    adc_T3_contrast = pickle.load(file)
    
with open('adc_T1_energy_origin.pkl', 'rb') as file:
    adc_T1_energy = pickle.load(file)
with open('adc_T2_energy_origin.pkl', 'rb') as file:
    adc_T2_energy = pickle.load(file)
with open('adc_T3_energy_origin.pkl', 'rb') as file:
    adc_T3_energy = pickle.load(file)
    
with open('adc_T1_correlation_origin.pkl', 'rb') as file:
    adc_T1_correlation = pickle.load(file)
with open('adc_T2_correlation_origin.pkl', 'rb') as file:
    adc_T2_correlation = pickle.load(file)
with open('adc_T3_correlation_origin.pkl', 'rb') as file:
    adc_T3_correlation = pickle.load(file)
    
with open('adc_T1_homogeneity_origin.pkl', 'rb') as file:
    adc_T1_homogeneity = pickle.load(file)
with open('adc_T2_homogeneity_origin.pkl', 'rb') as file:
    adc_T2_homogeneity = pickle.load(file)
with open('adc_T3_homogeneity_origin.pkl', 'rb') as file:
    adc_T3_homogeneity = pickle.load(file)
    
    
### DCE FEATURES
with open('dce_T1_contrast_origin.pkl', 'rb') as file:
    dce_T1_contrast = pickle.load(file)
with open('dce_T2_contrast_origin.pkl', 'rb') as file:
    dce_T2_contrast = pickle.load(file)
with open('dce_T3_contrast_origin.pkl', 'rb') as file:
    dce_T3_contrast = pickle.load(file)
    
with open('dce_T1_energy_origin.pkl', 'rb') as file:
    dce_T1_energy = pickle.load(file)
with open('dce_T2_energy_origin.pkl', 'rb') as file:
    dce_T2_energy = pickle.load(file)
with open('dce_T3_energy_origin.pkl', 'rb') as file:
    dce_T3_energy = pickle.load(file)
    
with open('dce_T1_correlation_origin.pkl', 'rb') as file:
    dce_T1_correlation = pickle.load(file)
with open('dce_T2_correlation_origin.pkl', 'rb') as file:
    dce_T2_correlation = pickle.load(file)
with open('dce_T3_correlation_origin.pkl', 'rb') as file:
    dce_T3_correlation = pickle.load(file)
    
with open('dce_T1_homogeneity_origin.pkl', 'rb') as file:
    dce_T1_homogeneity = pickle.load(file)
with open('dce_T2_homogeneity_origin.pkl', 'rb') as file:
    dce_T2_homogeneity = pickle.load(file)
with open('dce_T3_homogeneity_origin.pkl', 'rb') as file:
    dce_T3_homogeneity = pickle.load(file)
    
    
### PET FEATURES
with open('pet_T1_contrast_origin.pkl', 'rb') as file:
    pet_T1_contrast = pickle.load(file)
with open('pet_T2_contrast_origin.pkl', 'rb') as file:
    pet_T2_contrast = pickle.load(file)
with open('pet_T3_contrast_origin.pkl', 'rb') as file:
    pet_T3_contrast = pickle.load(file)
    
with open('pet_T1_energy_origin.pkl', 'rb') as file:
    pet_T1_energy = pickle.load(file)
with open('pet_T2_energy_origin.pkl', 'rb') as file:
    pet_T2_energy = pickle.load(file)
with open('pet_T3_energy_origin.pkl', 'rb') as file:
    pet_T3_energy = pickle.load(file)
    
with open('pet_T1_correlation_origin.pkl', 'rb') as file:
    pet_T1_correlation = pickle.load(file)
with open('pet_T2_correlation_origin.pkl', 'rb') as file:
    pet_T2_correlation = pickle.load(file)
with open('pet_T3_correlation_origin.pkl', 'rb') as file:
    pet_T3_correlation = pickle.load(file)
    
with open('pet_T1_homogeneity_origin.pkl', 'rb') as file:
    pet_T1_homogeneity = pickle.load(file)
with open('pet_T2_homogeneity_origin.pkl', 'rb') as file:
    pet_T2_homogeneity = pickle.load(file)
with open('pet_T3_homogeneity_origin.pkl', 'rb') as file:
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

#%% Step 0.45: Combine all values

List_contrast = []
List_energy   = []
List_correlation = []
List_homogeneity = []

for i in range(L):
    lists1 = [flat_adc_T1_energy[i],flat_adc_T2_energy[i],flat_adc_T3_energy[i],flat_dce_T1_energy[i],flat_dce_T2_energy[i],flat_dce_T3_energy[i],flat_pet_T1_energy[i],flat_pet_T2_energy[i],flat_pet_T3_energy[i]]
    List_energy.append(np.concatenate(lists1))
    
    lists2 = [flat_adc_T1_contrast[i],flat_adc_T2_contrast[i],flat_adc_T3_contrast[i],flat_dce_T1_contrast[i],flat_dce_T2_contrast[i],flat_dce_T3_contrast[i],flat_pet_T1_contrast[i],flat_pet_T2_contrast[i],flat_pet_T3_contrast[i]]
    List_contrast.append(np.concatenate(lists2))
    
    lists3 = [flat_adc_T1_correlation[i],flat_adc_T2_correlation[i],flat_adc_T3_correlation[i],flat_dce_T1_correlation[i],flat_dce_T2_correlation[i],flat_dce_T3_correlation[i],flat_pet_T1_correlation[i],flat_pet_T2_correlation[i],flat_pet_T3_correlation[i]]
    List_correlation.append(np.concatenate(lists3))
    
    lists4 = [flat_adc_T1_homogeneity[i],flat_adc_T2_homogeneity[i],flat_adc_T3_homogeneity[i],flat_dce_T1_homogeneity[i],flat_dce_T2_homogeneity[i],flat_dce_T3_homogeneity[i],flat_pet_T1_homogeneity[i],flat_pet_T2_homogeneity[i],flat_pet_T3_homogeneity[i]]
    List_homogeneity.append(np.concatenate(lists4))
    
List_final = []

for i in range(L):
    # List = [List_energy[i], List_contrast[i], List_correlation[i], List_homogeneity[i]]
    List = [List_contrast[i]]
    List_final.append(np.concatenate(List))
    
    
# X1 = np.concatenate(list1, axis=1)
# X2 = np.concatenate(

#%% Step 0.5 Randomly pick some features

import random   
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

num_values_to_select = 200

# Lists to store evaluation metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

model = LinearRegression()
num_features = 18 # Adjust this to the desired number of features               

for kkk in range(20):
    
    sampled_values = []

    for arr in List_final:
        sampled_indices = random.sample(range(len(arr)), min(len(arr), num_values_to_select))
        sampled_values.append(arr[sampled_indices])
    
    X = np.vstack(sampled_values)
    
    Y = [1,1,0,1,1,1,0,0,1,1,0,0,0,0,1,0,1,1,0,0,1,1,1]
    Y = np.array([1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,1,1,0,0,1])
    
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.linear_model import LogisticRegression
    
    # Define the number of splits (k)
    k = 5
    
    # Initialize the KFold object
    kf = KFold(n_splits=k)
    
    # Initialize the model
    model = LogisticRegression()
    
    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        rfe = RFE(model, n_features_to_select = num_features)
        selector = rfe.fit_transform(X_train, y_train)
        
        X_train  = X_train[:, rfe.support_]
        X_test   = X_test[:,  rfe.support_]
        
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



#%% All Label Code ##########################################

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
print('The Energy length should be: ',      np.min(len_list_energy))
print('The Contrast length should be: ',    np.min(len_list_contrast))
print('The Correlation length should be: ', np.min(len_list_correlation))
print('The Homogeneity length should be: ', np.min(len_list_homogeneity))

#%% Generate New Arrays

List_energy_new   = []
List_contrast_new = []
List_correlation_new = []
List_homogeneity_new = []

for i in range(20):
    List_energy_new.append(List_energy[i][0:np.min(len_list_energy)])
    List_contrast_new.append(List_contrast[i][0:np.min(len_list_contrast)])
    List_correlation_new.append(List_correlation[i][0:np.min(len_list_correlation)])
    List_homogeneity_new.append(List_homogeneity[i][0:np.min(len_list_homogeneity)])
    
#%% Step 0.5 Randomly pick some features

import random   
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.linear_model import LogisticRegression

# Define the number of splits (k)
k = 5

# Initialize the KFold object
kf = KFold(n_splits=k)

num_values_to_select = min_len 

# Lists to store evaluation metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

model = LogisticRegression(max_iter=1000)
num_features = 18
 # Adjust this to the desired number of features               

for kkk in range(3):
    
    # X = np.vstack(sampled_values)
    X1 = np.array(List_energy_new)
    X2 = np.array(List_contrast_new)
    X3 = np.array(List_correlation_new)
    X4 = np.array(List_homogeneity_new)
    
    X = np.concatenate((X1,X2,X3,X4), axis=1)
    
    Y = [1,1,0,1,1,1,0,0,1,1,0,0,0,0,1,0,1,1,0,0,1,1,1]
    Y = np.array([1,1,0,1,1,0,0,1,1,0,0,0,0,1,0,1,1,0,0,1])
    
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.linear_model import LogisticRegression
    
    # Define the number of splits (k)
    k = 5
    
    # Initialize the KFold object
    kf = KFold(n_splits=k)
    
    # Initialize the model
    # model = LogisticRegression()
    
    # Perform k-fold cross-validation
    for train_index, test_index in kf.split(X): 
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        rfe = RFE(model, n_features_to_select = num_features)
        selector = rfe.fit_transform(X_train, y_train)
        
        X_train  = X_train[:, rfe.support_]
        X_test   = X_test[:,  rfe.support_]
        
        # Fit the model on the training data
        model.fit(X_train, y_train)
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
average_accuracy = np.mean(accuracy_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_f1 = np.mean(f1_scores)

# Print results
print(f'Average Accuracy: {average_accuracy}')
print(f'Average Precision: {average_precision}')
print(f'Average Recall: {average_recall}')
print(f'Average F1-score: {average_f1}')

#%% Print out location

for i in range(len(rfe.support_)):
    if rfe.support_[i] == True:
        print('True Label at Location: ', i)













