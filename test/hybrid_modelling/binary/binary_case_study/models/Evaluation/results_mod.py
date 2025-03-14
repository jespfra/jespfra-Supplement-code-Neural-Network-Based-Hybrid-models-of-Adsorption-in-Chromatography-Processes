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
    'loss_training', 'RMSE_training', 'MAE_test', 'MSE_test',
    'NMSE_test', 'loss_test', 'RMSE_test', 'convergence'
]



# Display the results
results = pd.read_csv('../../results.csv')

# translation_table = [[1, 0], [2, 1], [3, 3], [4, 16], [5, 4], [6, 10], [7, 11], [8, 8], [9, 12], [10, 14], [11, 13], [12, 15]]
translation_table = [[1, 0], [2, 1], [3, 3], [4, 16], [5, 4], [6, 8], [7, 12], [8, 14], [9, 13], [10, 15]]
# Create an empty DataFrame with the specified column
results_mod = pd.DataFrame(columns=columns)


results_mm = pd.read_csv('../mechanistic_model/metrics_train_avg.csv')
df = pd.DataFrame({
    'model': [0],
    'layers': [0],
    'neurons': [0],
    'MAE_training': [results_mm['MAE_training'][0]],
    'MSE_training': [results_mm['MSE_training'][0]],
    'NMSE_training': [results_mm['NMSE_training'][0]],
    'loss_training': [results_mm['loss_training'][0]],
    'RMSE_training': [results_mm['RMSE_training'][0]],
    'MAE_test': [results_mm['MAE_test'][0]],
    'MSE_test': [results_mm['MSE_test'][0]],
    'NMSE_test': [results_mm['NMSE_test'][0]],
    'loss_test': [results_mm['loss_test'][0]],
    'RMSE_test': [results_mm['RMSE_test'][0]],
    'convergence': [0]
})
results_mod = pd.concat([results_mod, df], ignore_index=True)

# Populate the results_mod DataFrame
for i in range(len(translation_table)):
    model_value = translation_table[i][1]
    row = results[results['model'] == model_value].copy()
    row['model'] = translation_table[i][0]
    results_mod = pd.concat([results_mod, row], ignore_index=True)



results_mod.to_csv('results_mod.csv')




import matplotlib.pyplot as plt 


# Sample data
IT = results_mod['MSE_training']
ECE = results_mod['MSE_test']

barWidth = 0.4
fig, ax1 = plt.subplots(figsize=(12, 8))

br1 = np.arange(len(IT))
br2 = [x + barWidth for x in br1]

ax1.bar(br1, IT, color='r', width=barWidth, edgecolor='grey', label='Training MSE')
ax1.bar(br2, ECE, color='g', width=barWidth, edgecolor='grey', label='Test MSE')

ax1.set_xlabel('Layers', fontweight='bold', fontsize=20)
ax1.set_ylabel('MSE (mol/L)$^2$', fontweight='bold', fontsize=20)

# Draw horizontal dashed line at height ECE[0]
ax1.axhline(y=ECE[0], color='black', linestyle='--', linewidth=2, label=f'MM Test MSE = {ECE[0]:.3f} (mol/L)$^2$')

# Set primary x-ticks
primary_xticks = ['-'] + [str(i % 2 + 1) for i in range(len(IT)-1)]
xticks = list([r + barWidth/2 for r in range(1,len(primary_xticks))])
xticks.insert(0, barWidth/2)
ax1.set_xticks(xticks)
ax1.set_xticklabels(primary_xticks)

# Set secondary x-ticks
secondary_xticks = [''] + ['MM'] + [f'HM{i}' for i in range(1, len(IT) // 2 + 1)]+[""]
ax2 = ax1.twiny()
xticks2 = list(r + barWidth*3/2 for r in range(1,len(secondary_xticks)-1))
xticks2.insert(0, barWidth*2)
xticks2.insert(0, 0)
ax2.set_xticks(xticks2)
ax2.set_xticklabels(secondary_xticks)
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward', 60))
ax2.set_xlabel('Models', fontweight='bold', fontsize=20)



# Adjust font size of ticks
ax1.tick_params(axis='both', which='major', labelsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)
ax1.legend(fontsize=20)
ax1.grid(True)
plt.tight_layout()
plt.savefig("results_mod.svg")
plt.show()




#%% plot 2d matrix with models neurons and test mse 
# Downwards it should have models 1-2 using 1-2 layers 
# sideways should have 2-10 neurons 
matrix_mse = pd.DataFrame(columns=['HM', 'layers', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10'])


for i in range(len(translation_table)):
    
    # Read the metrics_train_avg.csv file for 1 layer 
    df_1_layer = pd.read_csv(f'../model_structure_{translation_table[i][1]}/metrics_train_avg.csv')
    
    # Extract neurons and test MSE for 1 layer
    test_mse_1_layer = df_1_layer['MSE_test'].values
    
    # Store the values in matrix_mse for 1 layer
    # row value: HM = HM{translation_table[i][1]}, layers = 1, N2..N10 = test_mse_1_layer
    df = {'HM': f'HM{translation_table[i][0]}',
        'layers': 1,
        'N2': [test_mse_1_layer[0]],
        'N3': [test_mse_1_layer[1]],
        'N4': [test_mse_1_layer[2]],
        'N5': [test_mse_1_layer[3]],
        'N6': [test_mse_1_layer[4]],
        'N7': [test_mse_1_layer[5]],
        'N8': [test_mse_1_layer[6]],
        'N9': [test_mse_1_layer[7]],
        'N10': [test_mse_1_layer[8]] }
    matrix_mse = pd.concat([matrix_mse, pd.DataFrame(df)], ignore_index=True)
    
    # Read the metrics_train_avg_2.csv file for 2 layer 
    df_2_layer = pd.read_csv(f'../model_structure_{translation_table[i][1]}/metrics_train_avg_2.csv')
    
    # Extract neurons and test MSE for 2 layer
    test_mse_2_layer = df_2_layer['MSE_test'].values
    
    # Store the values in matrix_mse for 2 layer
    # row value: HM = HM{translation_table[i][1]}, layers = 2, N2..N10 = test_mse_1_layer
    df = {
        'HM': f'HM{translation_table[i][0]}',
        'layers': 2,
        'N2': [test_mse_2_layer[0]],
        'N3': [test_mse_2_layer[1]],
        'N4': [test_mse_2_layer[2]],
        'N5': [test_mse_2_layer[3]],
        'N6': [test_mse_2_layer[4]],
        'N7': [test_mse_2_layer[5]],
        'N8': [test_mse_2_layer[6]],
        'N9': [test_mse_2_layer[7]],
        'N10': [test_mse_2_layer[8]]}
    matrix_mse = pd.concat([matrix_mse, pd.DataFrame(df)], ignore_index=True)
    

matrix_mse_val = np.array([
    matrix_mse['N2'].values, 
    matrix_mse['N3'].values,
    matrix_mse['N4'].values,
    matrix_mse['N5'].values,
    matrix_mse['N6'].values,
    matrix_mse['N7'].values,
    matrix_mse['N8'].values,
    matrix_mse['N9'].values,
    matrix_mse['N10'].values]).T


# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 12))

# Create a heatmap using imshow
cax = ax.imshow(matrix_mse_val, cmap='RdYlGn_r', aspect='auto', vmax=1.5)

# Add color bar
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Test MSE (mol/L)$^2$', fontsize=20)

# Set the font size of the color bar values
cbar.ax.tick_params(labelsize=18)

# Set the x-axis labels
ax.set_xticks(np.arange(len(matrix_mse.columns[2:])))
ax.set_xticklabels(matrix_mse.columns[2:], fontsize=20)

# Set the y-axis labels
y_labels = [f"{row['HM']}, {row['layers']}" for _, row in matrix_mse.iterrows()]
ax.set_yticks(np.arange(len(y_labels)))
ax.set_yticklabels(y_labels, fontsize=20)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Annotate the heatmap with the test MSE values and mark the lowest value in each row with bold text
for i in range(len(y_labels)):
    for j in range(len(matrix_mse.columns[2:])):
        value = matrix_mse_val[i, j]
        if value < 0.1:
            formatted_value = f"{value:.4f}"
        elif value < 1:
            formatted_value = f"{value:.3f}"
        else:
            formatted_value = f"{value:.2f}"
        
        if j == np.argmin(matrix_mse_val[i]):
            text = ax.text(j, i, formatted_value,
                           ha="center", va="center", color="black", fontsize=16, fontweight='bold')
        else:
            text = ax.text(j, i, formatted_value,
                           ha="center", va="center", color="black", fontsize=15)


# Set the labels and title
ax.set_xlabel('Neurons', fontsize=20)
# ax.set_ylabel('HM and Layers', fontsize=20)
# ax.set_title('Test MSE for Different HM, Layers, and Neurons')

# Show the plot
plt.tight_layout()
plt.savefig("results_mod_matrix.svg", dpi=400)
plt.show()


#%% Same plot but with pretrained models only
vmin = matrix_mse_val.min()
matrix_mse = pd.DataFrame(columns=['HM', 'layers', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10'])


for i in range(len(translation_table)):
    if i == 0 :
        continue
    
    # Read the metrics_train_avg.csv file for 1 layer 
    df_1_layer = pd.read_csv(f'../model_structure_{translation_table[i][1]}/metrics_train_avg.csv')
    
    # Extract neurons and test MSE for 1 layer
    test_mse_1_layer = df_1_layer['MSE_test_pretrain'].values
    
    # Store the values in matrix_mse for 1 layer
    # row value: HM = HM{translation_table[i][1]}, layers = 1, N2..N10 = test_mse_1_layer
    df = {'HM': f'HM{translation_table[i][0]}',
        'layers': 1,
        'N2': [test_mse_1_layer[0]],
        'N3': [test_mse_1_layer[1]],
        'N4': [test_mse_1_layer[2]],
        'N5': [test_mse_1_layer[3]],
        'N6': [test_mse_1_layer[4]],
        'N7': [test_mse_1_layer[5]],
        'N8': [test_mse_1_layer[6]],
        'N9': [test_mse_1_layer[7]],
        'N10': [test_mse_1_layer[8]] }
    matrix_mse = pd.concat([matrix_mse, pd.DataFrame(df)], ignore_index=True)
    
    # Read the metrics_train_avg_2.csv file for 2 layer 
    df_2_layer = pd.read_csv(f'../model_structure_{translation_table[i][1]}/metrics_train_avg_2.csv')
    
    # Extract neurons and test MSE for 2 layer
    test_mse_2_layer = df_2_layer['MSE_test_pretrain'].values
    
    # Store the values in matrix_mse for 2 layer
    # row value: HM = HM{translation_table[i][1]}, layers = 2, N2..N10 = test_mse_1_layer
    df = {
        'HM': f'HM{translation_table[i][0]}',
        'layers': 2,
        'N2': [test_mse_2_layer[0]],
        'N3': [test_mse_2_layer[1]],
        'N4': [test_mse_2_layer[2]],
        'N5': [test_mse_2_layer[3]],
        'N6': [test_mse_2_layer[4]],
        'N7': [test_mse_2_layer[5]],
        'N8': [test_mse_2_layer[6]],
        'N9': [test_mse_2_layer[7]],
        'N10': [test_mse_2_layer[8]]}
    matrix_mse = pd.concat([matrix_mse, pd.DataFrame(df)], ignore_index=True)
    

matrix_mse_val = np.array([
    matrix_mse['N2'].values, 
    matrix_mse['N3'].values,
    matrix_mse['N4'].values,
    matrix_mse['N5'].values,
    matrix_mse['N6'].values,
    matrix_mse['N7'].values,
    matrix_mse['N8'].values,
    matrix_mse['N9'].values,
    matrix_mse['N10'].values]).T


# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 12))

# Create a heatmap using imshow
cax = ax.imshow(matrix_mse_val, cmap='RdYlGn_r', aspect='auto', vmax=1.5, vmin = vmin)

# Add color bar
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Test MSE (mol/L)$^2$', fontsize=20)

# Set the font size of the color bar values
cbar.ax.tick_params(labelsize=18)

# Set the x-axis labels
ax.set_xticks(np.arange(len(matrix_mse.columns[2:])))
ax.set_xticklabels(matrix_mse.columns[2:], fontsize=20)

# Set the y-axis labels
y_labels = [f"{row['HM']}, {row['layers']}" for _, row in matrix_mse.iterrows()]
ax.set_yticks(np.arange(len(y_labels)))
ax.set_yticklabels(y_labels, fontsize=20)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Annotate the heatmap with the test MSE values and mark the lowest value in each row with bold text
for i in range(len(y_labels)):
    for j in range(len(matrix_mse.columns[2:])):
        value = matrix_mse_val[i, j]
        if value < 0.1:
            formatted_value = f"{value:.4f}"
        elif value < 1:
            formatted_value = f"{value:.3f}"
        else:
            formatted_value = f"{value:.2f}"
        
        if j == np.argmin(matrix_mse_val[i]):
            text = ax.text(j, i, formatted_value,
                           ha="center", va="center", color="black", fontsize=16, fontweight='bold')
        else:
            text = ax.text(j, i, formatted_value,
                           ha="center", va="center", color="black", fontsize=15)


# Set the labels and title
ax.set_xlabel('Neurons', fontsize=20)
# ax.set_ylabel('HM and Layers', fontsize=20)
# ax.set_title('Test MSE for Different HM, Layers, and Neurons')

# Show the plot
plt.tight_layout()
plt.savefig("results_mod_matrix_pretrain.svg", dpi=400)
plt.show()
