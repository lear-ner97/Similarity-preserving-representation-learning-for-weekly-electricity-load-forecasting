# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 21:26:15 2026

@author: umroot
"""


import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset
from functions import *
from itertools import product
import json 
import time
import matplotlib.dates as mdates
from feature_engineering import *
from visualization import *
from baselines import *




torch.cuda.memory.reset_peak_memory_stats() 
num_runs=5

#historical length and future horizon
T=7*24
H=7*24

total_pretraining_epochs=50
batch_size=64


search_space = {
    #"total_pretraining_epochs": [30],
    "hidden_sizes": [64],
    #"linear_probing_epochs": [20],
    "learning_rate": [1e-4],
    "dropout":[0.3],
    "patience":[7],
    "weight_decay":[1e-4],
    "maxnorm":[2]
    #"batch_size": [64],
}

training_mode='ssl-sup' #supervised or ssl or tgt-only
forecasting_type='multivariate' #multivariate
save_dir = f"./checkpoints/{forecasting_type}/history_{T}_horizon_{H}"
os.makedirs(save_dir, exist_ok=True)


load_path=r"data\archive (1)\electricity_cleaned.csv" #put your data directory
df=pd.read_csv(load_path)
weather_path=r"data\archive (1)\weather.csv"
weather=pd.read_csv(weather_path)

#source building
src_site_id='Hog' #location
src_building_type='office' #building type 
src_building_name='Mary' #
src_building=f'{src_site_id}_{src_building_type}_{src_building_name}'
src_load=df[src_building]
#target building
tgt_site_id='Hog' #location
tgt_building_type='office' #building type
tgt_building_name='Miriam'
tgt_building=f'{tgt_site_id}_{tgt_building_type}_{tgt_building_name}'
tgt_load=df[tgt_building]


#feature engineering
full_range = pd.date_range(start='2016-01-01 00:00:00',end='2017-12-31 23:00:00',freq='H')
#imputation of temperature missing values
src_airTemp=missing_temperature_imputation(weather,src_site_id,full_range)
tgt_airTemp=missing_temperature_imputation(weather,tgt_site_id,full_range)
    

#temperature visualization
temperature_visualization(tgt_airTemp)
# load visualization
load_visualization(src_airTemp[:int(len(src_airTemp)*15/24)],src_load[:int(len(src_airTemp)*15/24)],'blue')
load_visualization(tgt_airTemp[-int(len(tgt_airTemp)*3/24):],tgt_load[-int(len(tgt_airTemp)*3/24):],'orange')
#acf plot: detection of seasonality
acf_load(src_load)


#calendar features
#if tgt_site_id=='Robin' or tgt_site_id=='Lamb' :
time_series_weekend,src_time_series_holiday,tgt_time_series_holiday,time_series_hour_sin,\
time_series_hour_cos,time_series_day,time_series_day_sin,\
time_series_day_cos=calendar_features(full_range,src_site_id,tgt_site_id)



#creation of source and target data samples using moving window
src_time_series_load=np.array(src_load)
tgt_time_series_load=np.array(tgt_load)
src_time_series_airTemp=np.array(src_airTemp)
tgt_time_series_airTemp=np.array(tgt_airTemp)

src_data = np.column_stack((src_time_series_load, src_time_series_airTemp, time_series_hour_sin, time_series_hour_cos,time_series_day_sin, time_series_day_cos, time_series_weekend, src_time_series_holiday))#
tgt_data = np.column_stack((tgt_time_series_load, tgt_time_series_airTemp, time_series_hour_sin, time_series_hour_cos,time_series_day_sin, time_series_day_cos, time_series_weekend, tgt_time_series_holiday))#


#split data into source pretrain, target finetune (train), target validation, target test

#source pretrain data end index
pretrain_end = int(len(src_data)*15/24)#len(src_X_tensor)//2 #only 1 year of source data #int(train_end * n)

valid_src_start = len(src_data)-int(len(src_data)*9/24) #and ends after three months at test_start
#test data start index
test_start=len(src_data)-int(len(src_data)*6/24) #the last 6 months


# Pretraining 
src_pretrain = src_data[:pretrain_end]
tgt_train= tgt_data[valid_src_start:valid_src_start+7*24] #only 1 week to satisfy the model's 
#one week historical data assumption

#data normalization using mean and std of train data, only load and temperature are normalized
num_numeric_features=2 #load and temp

src_numeric_features = src_pretrain[:, :num_numeric_features]
src_binary_features = src_pretrain[:, num_numeric_features:]
src_numeric_features_complete = src_data[:, :num_numeric_features]
src_binary_features_complete = src_data[:, num_numeric_features:]
src_mean = np.mean(src_numeric_features, axis=0)
src_std = np.std(src_numeric_features, axis=0)
src_numeric_features_complete = (src_numeric_features_complete - src_mean) / src_std
src_normalized = np.concatenate([src_numeric_features_complete, src_binary_features_complete], axis=-1)


tgt_numeric_features = tgt_train[:, :num_numeric_features]
tgt_binary_features = tgt_train[:, num_numeric_features:]
tgt_numeric_features_complete = tgt_data[:, :num_numeric_features]
tgt_binary_features_complete = tgt_data[:, num_numeric_features:]
tgt_mean = np.mean(tgt_numeric_features, axis=0)
tgt_std = np.std(tgt_numeric_features, axis=0)
tgt_numeric_features_complete = (tgt_numeric_features_complete - tgt_mean) / tgt_std
tgt_normalized = np.concatenate([tgt_numeric_features_complete, tgt_binary_features_complete], axis=-1)


#normalized source  
src_pretrain = src_normalized[:pretrain_end]
src_valid = src_normalized[valid_src_start:test_start]

# Test
tgt_test = tgt_normalized[test_start:]

n_features=tgt_train.shape[-1]

print(src_pretrain.shape, tgt_test.shape)

# generation of normalized input-output samples using moving window 
src_X_pretrain,src_y_pretrain=create_windows(src_pretrain, T, H)
src_X_valid,src_y_valid=create_windows(src_valid, T, H)
tgt_X_test,tgt_y_test=create_windows(tgt_test, T, H)


print(src_X_pretrain.shape, src_X_valid.shape, tgt_X_test.shape, tgt_y_test.shape)

src_pretrain_dataset=TensorDataset(src_X_pretrain, src_y_pretrain)
src_valid_dataset=TensorDataset(src_X_valid, src_y_valid)
tgt_test_dataset=TensorDataset(tgt_X_test, tgt_y_test)


#### start training

best_metrics_runs=[]
best_config_runs=[]
#n_trials = 20
keys = search_space.keys()
#unique configurations
all_configs = [
    dict(zip(search_space.keys(), values))
    for values in product(*search_space.values())
]

    
rmse_runs = []
mae_runs = []
smape_runs = []

base_seed = 2026


for i in range(num_runs):
    print(f"runtime {i+1}/{num_runs}")

    best_val_loss = float("inf")
    best_config=None
    best_model=None
    seed = base_seed + i
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for config in all_configs:
        hidden_sizes=config['hidden_sizes']
        learning_rate=config['learning_rate']
        dropout=config['dropout']
        patience=config['patience']
        weight_decay=config['weight_decay']
        maxnorm=config['maxnorm']
        start_time = time.time()
        pretraining_losses,model=pre_training_zeroshot(src_pretrain_dataset,src_valid_dataset,total_pretraining_epochs,T,n_features,H,hidden_sizes,learning_rate,batch_size,tgt_building,dropout,weight_decay,maxnorm,seed,patience)
        end_time = time.time()
        pretraining_time = end_time - start_time
        print(f"preTraining time: {pretraining_time:.2f} seconds")

    save_path_model=None #,save_path_model
    best_config=config
    y_true,y_pred,metrics=test(tgt_test_dataset,best_config,T,n_features,H,torch.tensor(tgt_mean)[0].to('cuda'),torch.tensor(tgt_std)[0].to('cuda'),batch_size,model)
    print('best config: '+str(best_config))
    print('validation metrics of the best model: \n'+str(metrics))
    mae_runs.append(metrics['MAE'])
    rmse_runs.append(metrics['RMSE'])
    smape_runs.append(metrics['sMAPE (%)'])
    


print("=" * 50)
print(f"Source building : {src_building}")
print(f"Target building : {tgt_building}")
print(f"Training mode   : {training_mode}")
print(f"Horizon         : {H}")
print(f"MAE:  {np.mean(mae_runs):.4f} ± {np.std(mae_runs):.4f}")
print(f"RMSE: {np.mean(rmse_runs):.4f} ± {np.std(rmse_runs):.4f}")
print(f"sMAPE: {np.mean(smape_runs):.4f} ± {np.std(smape_runs):.4f}")
print("-" * 50)

########################### model XAI: integrated gradient
 #uncomment (it takes some time to finish)
#integrated_gradient(model,tgt_X_test)

############################### end xAI
