# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:04:44 2026

@author: umroot
"""
import os
import random
from captum.attr import IntegratedGradients #pip install captum
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



torch.cuda.memory.reset_peak_memory_stats() #Reset the “peak” stats tracked by the CUDA memory allocator

#number of runs
num_runs=5
#historical length and future horizon
T=7*24
H=7*24


total_pretraining_epochs=30
linear_probing_epochs=20
batch_size=64


search_space = {
    #"total_pretraining_epochs": [30],
    "hidden_sizes": [128],
    #"linear_probing_epochs": [20],
    "learning_rate": [1e-3],
    "dropout":[0.3],
    "patience":[7],
    "weight_decay":[1e-4],
    "maxnorm":[2]
    #"batch_size": [64],
}

training_mode='ssl-sup' #ssl-sup or tgt-only
forecasting_type='multivariate' #multivariate
save_dir = f"./checkpoints/{forecasting_type}/history_{T}_horizon_{H}"
os.makedirs(save_dir, exist_ok=True)
#we are interested in these site ids: 'Robin', 'Lamb', 'Hog'
#robin: london, lamb: cardiff (UK),  hog: US central

load_path=r"data\archive (1)\electricity_cleaned.csv" ####change load path to your specific path
df=pd.read_csv(load_path)
weather_path=r"data\archive (1)\weather.csv" #change this path as well
weather=pd.read_csv(weather_path)





#source building
src_site_id='Robin' #location
src_building_type='lodging' #building type 
src_building_name='Dorthy' #
src_building=f'{src_site_id}_{src_building_type}_{src_building_name}'
src_load=df[src_building]
#target building
tgt_site_id='Robin' #location
tgt_building_type='lodging' #building type
tgt_building_name='Celia'
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
load_visualization(tgt_airTemp,tgt_load)
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

# Create dataset
tgt_X = []
tgt_y = []
src_X = []
src_y = []


num_samples=len(src_time_series_load)-T-H 

for i in range(0,num_samples):
    src_x_load = src_time_series_load[i : i + T] 
    tgt_x_load = tgt_time_series_load[i : i + T]

    src_x_temp = src_time_series_airTemp[i : i + T]
    tgt_x_temp = tgt_time_series_airTemp[i : i + T]

    x_weekend=time_series_weekend[i:i+T]

    x_hour_sin=time_series_hour_sin[i:i+T]
    x_hour_cos=time_series_hour_cos[i:i+T]
    x_day_sin=time_series_day_sin[i:i+T]
    x_day_cos=time_series_day_cos[i:i+T]
    
    src_x_holiday=src_time_series_holiday[i:i+T]
    tgt_x_holiday=tgt_time_series_holiday[i:i+T]
    src_x = np.column_stack((src_x_load, src_x_temp, x_hour_sin, x_hour_cos, x_day_sin, x_day_cos, x_weekend, src_x_holiday))#
    tgt_x = np.column_stack((tgt_x_load, tgt_x_temp, x_hour_sin, x_hour_cos, x_day_sin, x_day_cos, x_weekend, tgt_x_holiday))#
 
    src_X.append(src_x)
    src_y.append(src_time_series_load[i + T : i + T + H])
    tgt_X.append(tgt_x)
    tgt_y.append(tgt_time_series_load[i + T : i + T + H])

src_X = np.array(src_X)
src_y = np.array(src_y)
tgt_X = np.array(tgt_X)
tgt_y = np.array(tgt_y)


src_X_tensor = torch.tensor(src_X, dtype=torch.float32)
src_y_tensor = torch.tensor(src_y, dtype=torch.float32)
tgt_X_tensor = torch.tensor(tgt_X, dtype=torch.float32)
tgt_y_tensor = torch.tensor(tgt_y, dtype=torch.float32)



n = src_X_tensor.shape[0]


#split data into source pretrain, target finetune (train), target validation, target test

#source pretrain data end index
pretrain_end = int(len(src_X_tensor)*15/24)#len(src_X_tensor)//2 #only 1 year of source data #int(train_end * n)
#finetune data start index
train_start=pretrain_end
#finetune data end index
train_end = train_start+10*24
#valid data end index
valid_end = train_end+4*24
#valid source data
valid_src_start = len(src_X_tensor)-int(len(src_X_tensor)*9/24) #and ends after three months at test_start
#test data start index
test_start=len(src_X_tensor)-int(len(src_X_tensor)*6/24) #the last 6 months


# Pretraining 
src_X_pretrain = src_X_tensor[:pretrain_end]
src_y_pretrain = src_y_tensor[:pretrain_end]  # used in linear probing, not pretraining

tgt_X_train = tgt_X_tensor[train_start:train_end]
tgt_y_train = tgt_y_tensor[train_start:train_end] 




#data normalization using mean and std of train data, only load and temperature are normalized
num_numeric_features=2 #load and temp

src_numeric_features = src_X_pretrain[:, :, :num_numeric_features]
src_binary_features = src_X_pretrain[:, :, num_numeric_features:]
src_numeric_features_complete = src_X_tensor[:, :, :num_numeric_features]
src_binary_features_complete = src_X_tensor[:, :, num_numeric_features:]
src_mean = src_numeric_features.mean(dim=(0,1), keepdim=True)
src_std = src_numeric_features.std(dim=(0,1), keepdim=True)
src_numeric_features_complete = (src_numeric_features_complete - src_mean) / src_std
src_X_tensor = torch.cat([src_numeric_features_complete, src_binary_features_complete], dim=-1)
src_y_tensor=(src_y_tensor - src_mean[:,:,0]) / src_std[:,:,0]

tgt_numeric_features = tgt_X_train[:, :, :num_numeric_features]
tgt_binary_features = tgt_X_train[:, :, num_numeric_features:]
tgt_numeric_features_complete = tgt_X_tensor[:, :, :num_numeric_features]
tgt_binary_features_complete = tgt_X_tensor[:, :, num_numeric_features:]
tgt_mean = tgt_numeric_features.mean(dim=(0,1), keepdim=True)
tgt_std = tgt_numeric_features.std(dim=(0,1), keepdim=True)
tgt_numeric_features_complete = (tgt_numeric_features_complete - tgt_mean) / tgt_std
tgt_X_tensor = torch.cat([tgt_numeric_features_complete, tgt_binary_features_complete], dim=-1)
tgt_y_tensor=(tgt_y_tensor - tgt_mean[:,:,0]) / tgt_std[:,:,0]


src_X_pretrain = src_X_tensor[:pretrain_end]
src_y_pretrain= src_y_tensor[:pretrain_end]
src_pretrain_dataset = TensorDataset(src_X_pretrain, src_y_pretrain)

tgt_X_train = tgt_X_tensor[train_start:train_end]
tgt_y_train= tgt_y_tensor[train_start:train_end]
tgt_train_dataset = TensorDataset(tgt_X_train, tgt_y_train)


n_features=src_X_pretrain.shape[-1] 



# Validation
tgt_X_valid = tgt_X_tensor[train_end:valid_end]
tgt_y_valid = tgt_y_tensor[train_end:valid_end]
src_X_valid = src_X_tensor[valid_src_start:test_start]
src_y_valid = src_y_tensor[valid_src_start:test_start]

if training_mode=='ssl-sup':
    tgt_valid_dataset = TensorDataset(torch.cat((tgt_X_valid, src_X_valid), dim=0), 
                                      torch.cat((tgt_y_valid, src_y_valid), dim=0))
else: #target only
    tgt_valid_dataset = TensorDataset(tgt_X_valid,tgt_y_valid)

# Test
tgt_X_test = tgt_X_tensor[test_start:]
tgt_y_test = tgt_y_tensor[test_start:]

tgt_test_dataset = TensorDataset(tgt_X_test, tgt_y_test)

print(src_X_pretrain.shape, tgt_X_train.shape, tgt_X_valid.shape, tgt_X_test.shape)







############ start training


best_metrics_runs=[]
best_config_runs=[]

keys = search_space.keys()
#unique configurations
all_configs = [
    dict(zip(search_space.keys(), values))
    for values in product(*search_space.values())
]

    
rmse_runs = []
mae_runs = []
mape_runs = []

base_seed = 2026

if training_mode=='tgt-only':
    for i in range(num_runs):
        seed = base_seed + i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        #hyperparameter tuning 
        best_config_encoder = None
        best_config_head=None
        best_val_loss = float("inf")
        best_config=None
        best_checkpointt=0
        index=1
        for config in all_configs:
            print(f'config {index}/{len(all_configs)}: {config} \n')
            index+=1

            hidden_sizes=config['hidden_sizes']
            learning_rate=config['learning_rate']
            dropout=config['dropout']
            patience=config['patience']
            weight_decay=config['weight_decay']
            maxnorm=config['maxnorm']

            best_model=tgt_only_training(hidden_sizes,linear_probing_epochs,tgt_train_dataset,tgt_valid_dataset,total_pretraining_epochs,T,n_features,H,learning_rate,batch_size,tgt_building,tgt_mean[:,:,0].to('cuda'),tgt_std[:,:,0].to('cuda'),dropout,patience,weight_decay,maxnorm,seed)

        y_true,y_pred,metrics=test(tgt_test_dataset,config,T,n_features,H,tgt_mean[:,:,0].to('cuda'),tgt_std[:,:,0].to('cuda'),batch_size,best_model)
       
        print('validation metrics of the best model: \n'+str(metrics))
        mae_runs.append(metrics['MAE'])
        rmse_runs.append(metrics['RMSE'])
        mape_runs.append(metrics['sMAPE (%)'])



elif training_mode=='ssl-sup':
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
        for config in all_configs: #if you want to perform hyperparameter tuning then at 
        # the end you have to compute validation metrics not test metrics

            hidden_sizes=config['hidden_sizes']
            learning_rate=config['learning_rate']
            dropout=config['dropout']
            patience=config['patience']
            weight_decay=config['weight_decay']
            maxnorm=config['maxnorm']

            start_time = time.time()
            pretraining_losses=pre_training_fewshot(src_pretrain_dataset,total_pretraining_epochs,T,n_features,H,hidden_sizes,learning_rate,batch_size,tgt_building,dropout,weight_decay,maxnorm,seed)
            end_time = time.time()
            pretraining_time = end_time - start_time
            print(f"preTraining time: {pretraining_time:.2f} seconds")
            start_time = time.time()
            lowest_validation_loss,best_checkpoint,best_model=linear_probing(hidden_sizes,linear_probing_epochs,tgt_train_dataset,tgt_valid_dataset,total_pretraining_epochs,T,n_features,H,learning_rate,batch_size,tgt_building,tgt_mean[:,:,0].to('cuda'),tgt_std[:,:,0].to('cuda'),dropout,patience,weight_decay,maxnorm,seed)
            end_time = time.time()
            finetuning_time = end_time - start_time
            print(f"finetuning time: {finetuning_time:.2f} seconds")

         
        
        
        ########################### model XAI: integrated gradient
        #uncomment (it takes some time to finish)

        # feature_names=['Load','Temperature','Hour sine encoding','Hour cosine encoding',
        #                'Day sine encoding','Day cosine encoding','Weekend','Holidays']
        # best_model.eval()
        # ig = IntegratedGradients(best_model)

        # start_time = time.time()
        # all_attr = []

        # for h in range(168):
        #     attr_h = ig.attribute(
        #         inputs=tgt_X_test.to('cuda'),#[:10,...] it may take time on full test data
        #         target=h, #target class (regrssion in our case) index
        #         n_steps=25
        #     ) #baseline x' = 0 by default, which makes sense because inputs are normalized 
        #     all_attr.append(attr_h)
        
        # all_attr = torch.stack(all_attr)

        # # average over future steps and samples
        # importance = (
        #     all_attr.abs()
        #     .mean(dim=(0,1))
        # ) #(input_size, features)
        
        # fig, ax = plt.subplots(figsize=(12, 5))
        
        # im = ax.imshow(
        #     importance.T.cpu(),
        #     aspect='auto',
        #     cmap='viridis'
        # )
        
        # plt.colorbar(im, ax=ax, label="Importance")
        
        # ax.set_xlabel("Historical time step")
        # ax.set_ylabel("Feature")
        # #ax.set_title("Overall importance across forecast horizon")
        
        # # Replace integer labels with feature names
        # ax.set_yticks(np.arange(len(feature_names)))
        # ax.set_yticklabels(feature_names)
        
        # plt.tight_layout()
        # plt.show()
       ############################### end xAI
        
        
        #computational complexity
        # peak_mem = torch.cuda.memory.max_memory_allocated(torch.device) / 1e9
        # print(f"Max peak memory (GB): {peak_mem:.2f}")

        best_config=config
        #test performance
        y_true,y_pred,metrics=test(tgt_test_dataset,best_config,T,n_features,H,tgt_mean[:,:,0].to('cuda'),tgt_std[:,:,0].to('cuda'),batch_size,best_model)
        print('best config: '+str(best_config))
        print('validation metrics of the best model: \n'+str(metrics))
        mae_runs.append(metrics['MAE'])
        rmse_runs.append(metrics['RMSE'])
        mape_runs.append(metrics['sMAPE (%)'])
        
print("=" * 50)
print(f"Source building : {src_building}")
print(f"Target building : {tgt_building}")
print(f"Training mode   : {training_mode}")
print(f"Horizon         : {H}")
print(f"MAE:  {np.mean(mae_runs):.4f} ± {np.std(mae_runs):.4f}")
print(f"RMSE: {np.mean(rmse_runs):.4f} ± {np.std(rmse_runs):.4f}")
print(f"sMAPE: {np.mean(mape_runs):.4f} ± {np.std(mape_runs):.4f}")
print("-" * 50)


