#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:08:26 2024

@author: jespfra
"""

import pandas as pd 
import numpy as np
import os

# Initialize an empty DataFrame to store results
columns = [
    'model', 'layers', 'neurons', 'MAE_training', 'MSE_training', 'NMSE_training',
    'loss_training', 'RMSE_training', 'MAE_test_binary', 'MSE_test_binary',
    'NMSE_test_binary', 'loss_test_binary', 'MAE_test', 'MSE_test',
    'NMSE_test', 'loss_test', 'RMSE_test', 'convergence'
]

# Create an empty DataFrame with the specified columns
results = pd.DataFrame(columns=columns)

setups = [0,1,3,4,8,10,11,12,13,14,15,16,17]

# Loop through the folders and process each CSV file
for i in range(0,18):
    # Construct the file path
    file_path = f'models/model_structure_{i}/metrics_train_avg.csv'  # Adjust the folder and file naming as needed
    print(i)
    # Open the CSV file
    if os.path.exists(file_path):
        df_train = pd.read_csv(file_path)
        
        # Find the row with the lowest MSE_validation
        best_row_validation = df_train.loc[df_train['MSE_test'].idxmin()]
        
        # find the row number 
        best_row_number = df_train[df_train['MSE_test']==df_train['MSE_test'].min()].idxmin()[0]
        
        convergence = pd.read_csv(f'models/model_structure_{i}/neurons_{int(best_row_validation["neurons"])}/convergence.csv')
    
        
        # Save the results
        best_row_df = pd.DataFrame({
           'model': [i],
           'layers': 1,
           'neurons': [best_row_validation['neurons']],
           'MAE_training': df_train['MAE_training'][best_row_number],
           'MSE_training': df_train['MSE_training'][best_row_number],
           'NMSE_training': df_train['NMSE_training'][best_row_number],
           'loss_training': df_train['loss_training'][best_row_number],
           'RMSE_training': df_train['RMSE_training'][best_row_number],
           'MAE_test_binary': df_train['MAE_test_binary'][best_row_number],
           'MSE_test_binary': df_train['MSE_test_binary'][best_row_number],
           'NMSE_test_binary': df_train['NMSE_test_binary'][best_row_number],
           'loss_test_binary': df_train['loss_test_binary'][best_row_number],
           'RMSE_test_binary': df_train['RMSE_test_binary'][best_row_number],
           'MAE_test': df_train['MAE_test'][best_row_number],
           'MSE_test': df_train['MSE_test'][best_row_number],
           'NMSE_test': df_train['NMSE_test'][best_row_number],
           'loss_test': df_train['loss_test'][best_row_number],
           'RMSE_test': df_train['RMSE_test'][best_row_number],
           'convergence': convergence['loss'].iloc[-1]
       })
       
        # Concatenate the best row DataFrame to the results DataFrame
        results = pd.concat([results, best_row_df], ignore_index=True)
    else:
        best_row_df = pd.DataFrame({
           'model': [i],
           'layers': 1,
           'neurons': 0,
           'MAE_training': 0,
           'MSE_training': 0,
           'NMSE_training': 0,
           'loss_training': 0,
           'RMSE_training': 0,
           'MAE_test_binary': 0,
           'MSE_test_binary': 0,
           'NMSE_test_binary': 0,
           'loss_test_binary': 0,
           'RMSE_test_binary': 0,
           'MAE_test': 0,
           'MSE_test': 0,
           'NMSE_test': 0,
           'loss_test': 0,
           'RMSE_test': 0,
           'convergence': 0
       })
       
        # Concatenate the best row DataFrame to the results DataFrame
        results = pd.concat([results, best_row_df], ignore_index=True)
    
    
    # Evaluate for 2 layers
    file_path = f'models/model_structure_{i}/metrics_train_avg_2.csv'  # Adjust the folder and file naming as needed
    # Open the CSV file
    if os.path.exists(file_path):
        df_validation = pd.read_csv(file_path)
        
        # Open the CSV file
        df_train = pd.read_csv(file_path)
        
        # Find the row with the lowest MSE_validation
        best_row_validation = df_train.loc[df_train['MSE_test'].idxmin()]
        
        # find the row number 
        best_row_number = df_train[df_train['MSE_test']==df_train['MSE_test'].min()].idxmin()[0]
        
        convergence = pd.read_csv(f'models/model_structure_{i}/neurons_{int(best_row_validation["neurons"])}_2/convergence_2.csv')
        
        # Save the results
        best_row_df = pd.DataFrame({
           'model': [i],
           'layers': 2,
           'neurons': [best_row_validation['neurons']],
           'MAE_training': df_train['MAE_training'][best_row_number],
           'MSE_training': df_train['MSE_training'][best_row_number],
           'NMSE_training': df_train['NMSE_training'][best_row_number],
           'loss_training': df_train['loss_training'][best_row_number],
           'RMSE_training': df_train['RMSE_training'][best_row_number],
           'MAE_test_binary': df_train['MAE_test_binary'][best_row_number],
           'MSE_test_binary': df_train['MSE_test_binary'][best_row_number],
           'NMSE_test_binary': df_train['NMSE_test_binary'][best_row_number],
           'loss_test_binary': df_train['loss_test_binary'][best_row_number],
           'RMSE_test_binary': df_train['RMSE_test_binary'][best_row_number],
           'MAE_test': df_train['MAE_test'][best_row_number],
           'MSE_test': df_train['MSE_test'][best_row_number],
           'NMSE_test': df_train['NMSE_test'][best_row_number],
           'loss_test': df_train['loss_test'][best_row_number],
           'RMSE_test': df_train['RMSE_test'][best_row_number],
           'convergence': convergence['loss'].iloc[-1]
       })
       
        # Concatenate the best row DataFrame to the results DataFrame
        results = pd.concat([results, best_row_df], ignore_index=True)
        
    else:
        best_row_df = pd.DataFrame({
           'model': [i],
           'layers': 2,
           'neurons': 0,
           'MAE_training': 0,
           'MSE_training': 0,
           'NMSE_training': 0,
           'loss_training': 0,
           'RMSE_training': 0,
           'MAE_test_binary': 0,
           'MSE_test_binary': 0,
           'NMSE_test_binary': 0,
           'loss_test_binary': 0,
           'RMSE_test_binary': 0,
           'MAE_test': 0,
           'MSE_test': 0,
           'NMSE_test': 0,
           'loss_test': 0,
           'RMSE_test': 0,
           'convergence': 0
       })
       
        # Concatenate the best row DataFrame to the results DataFrame
        results = pd.concat([results, best_row_df], ignore_index=True)

# Display the results
results.to_csv('results.csv')



#%% Same but for the pretrained models only 


columns = [
    'model', 'layers', 'neurons', 'MAE_training', 'MSE_training', 'NMSE_training',
    'loss_training', 'RMSE_training', 'MAE_test_binary', 'MSE_test_binary',
    'NMSE_test_binary', 'loss_test_binary', 'MAE_test', 'MSE_test',
    'NMSE_test', 'loss_test', 'RMSE_test', 'convergence'
]

# Create an empty DataFrame with the specified columns
results = pd.DataFrame(columns=columns)

# Loop through the folders and process each CSV file
for i in range(0,18):
    # Construct the file path
    file_path = f'models/model_structure_{i}/metrics_train_avg_pretrain.csv'  # Adjust the folder and file naming as needed
    print(i)
    # Open the CSV file
    if os.path.exists(file_path):
        df_train = pd.read_csv(file_path)
        
        # Find the row with the lowest MSE_validation
        best_row_validation = df_train.loc[df_train['MSE_test'].idxmin()]
        
        # find the row number 
        best_row_number = df_train[df_train['MSE_test']==df_train['MSE_test'].min()].idxmin()[0]
        
        # convergence = pd.read_csv(f'models/model_structure_{i}/neurons_{int(best_row_validation["neurons"])}/convergence.csv')
    
        
        # Save the results
        best_row_df = pd.DataFrame({
           'model': [i],
           'layers': 1,
           'neurons': [best_row_validation['neurons']],
           'MAE_training': df_train['MAE_training'][best_row_number],
           'MSE_training': df_train['MSE_training'][best_row_number],
           'NMSE_training': df_train['NMSE_training'][best_row_number],
           'loss_training': df_train['loss_training'][best_row_number],
           'RMSE_training': df_train['RMSE_training'][best_row_number],
           'MAE_test_binary': df_train['MAE_test_binary'][best_row_number],
           'MSE_test_binary': df_train['MSE_test_binary'][best_row_number],
           'NMSE_test_binary': df_train['NMSE_test_binary'][best_row_number],
           'loss_test_binary': df_train['loss_test_binary'][best_row_number],
           'RMSE_test_binary': df_train['RMSE_test_binary'][best_row_number],
           'MAE_test': df_train['MAE_test'][best_row_number],
           'MSE_test': df_train['MSE_test'][best_row_number],
           'NMSE_test': df_train['NMSE_test'][best_row_number],
           'loss_test': df_train['loss_test'][best_row_number],
           'RMSE_test': df_train['RMSE_test'][best_row_number],
           'convergence': 0
       })
       
        # Concatenate the best row DataFrame to the results DataFrame
        results = pd.concat([results, best_row_df], ignore_index=True)
    else:
        best_row_df = pd.DataFrame({
           'model': [i],
           'layers': 1,
           'neurons': 0,
           'MAE_training': 0,
           'MSE_training': 0,
           'NMSE_training': 0,
           'loss_training': 0,
           'RMSE_training': 0,
           'MAE_test_binary': 0,
           'MSE_test_binary': 0,
           'NMSE_test_binary': 0,
           'loss_test_binary': 0,
           'RMSE_test_binary': 0,
           'MAE_test': 0,
           'MSE_test': 0,
           'NMSE_test': 0,
           'loss_test': 0,
           'RMSE_test': 0,
           'convergence': 0
       })
       
        # Concatenate the best row DataFrame to the results DataFrame
        results = pd.concat([results, best_row_df], ignore_index=True)
    
    
    # Evaluate for 2 layers
    file_path = f'models/model_structure_{i}/metrics_train_avg_2_pretrain.csv'  # Adjust the folder and file naming as needed
    # Open the CSV file
    if os.path.exists(file_path):
        df_validation = pd.read_csv(file_path)
        
        # Open the CSV file
        df_train = pd.read_csv(file_path)
        
        # Find the row with the lowest MSE_validation
        best_row_validation = df_train.loc[df_train['MSE_test'].idxmin()]
        
        # find the row number 
        best_row_number = df_train[df_train['MSE_test']==df_train['MSE_test'].min()].idxmin()[0]
        
        # convergence = pd.read_csv(f'models/model_structure_{i}/neurons_{int(best_row_validation["neurons"])}_2/convergence_2.csv')
        
        # Save the results
        best_row_df = pd.DataFrame({
           'model': [i],
           'layers': 2,
           'neurons': [best_row_validation['neurons']],
           'MAE_training': df_train['MAE_training'][best_row_number],
           'MSE_training': df_train['MSE_training'][best_row_number],
           'NMSE_training': df_train['NMSE_training'][best_row_number],
           'loss_training': df_train['loss_training'][best_row_number],
           'RMSE_training': df_train['RMSE_training'][best_row_number],
           'MAE_test_binary': df_train['MAE_test_binary'][best_row_number],
           'MSE_test_binary': df_train['MSE_test_binary'][best_row_number],
           'NMSE_test_binary': df_train['NMSE_test_binary'][best_row_number],
           'loss_test_binary': df_train['loss_test_binary'][best_row_number],
           'RMSE_test_binary': df_train['RMSE_test_binary'][best_row_number],
           'MAE_test': df_train['MAE_test'][best_row_number],
           'MSE_test': df_train['MSE_test'][best_row_number],
           'NMSE_test': df_train['NMSE_test'][best_row_number],
           'loss_test': df_train['loss_test'][best_row_number],
           'RMSE_test': df_train['RMSE_test'][best_row_number],
           'convergence': 0
       })
       
        # Concatenate the best row DataFrame to the results DataFrame
        results = pd.concat([results, best_row_df], ignore_index=True)
        
    else:
        best_row_df = pd.DataFrame({
           'model': [i],
           'layers': 2,
           'neurons': 0,
           'MAE_training': 0,
           'MSE_training': 0,
           'NMSE_training': 0,
           'loss_training': 0,
           'RMSE_training': 0,
           'MAE_test_binary': 0,
           'MSE_test_binary': 0,
           'NMSE_test_binary': 0,
           'loss_test_binary': 0,
           'RMSE_test_binary': 0,
           'MAE_test': 0,
           'MSE_test': 0,
           'NMSE_test': 0,
           'loss_test': 0,
           'RMSE_test': 0,
           'convergence': 0
       })
       
        # Concatenate the best row DataFrame to the results DataFrame
        results = pd.concat([results, best_row_df], ignore_index=True)

# Display the results
results.to_csv('results_pretrain.csv')







