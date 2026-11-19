# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:26:42 2026

@author: umroot
"""
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from training import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, median_absolute_error
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from baselines import *
from ptflops import get_model_complexity_info
import matplotlib.dates as mdates
from captum.attr import IntegratedGradients #pip install captum

    




def pre_training_zeroshot(train_dataset,valid_dataset, total_epoch,T,n_features,H,hidden_sizes,lr,bs,building,dropout,w,maxnorm,seed,patience):#config, 
    #encoder=MLPEncoder(T,n_features,hidden_sizes,dropout).to('cuda')
    encoder=TSMixerEncoder(1,T,hidden_sizes,0.1,n_features).to('cuda')
    #benchmarking: comment mlpencoder and activate one of the following encoders
    #encoder=TransformerEncoder(n_features,hidden_sizes,T,8,2,256,0.1).to('cuda')#(T,n_features,hidden_sizes,dropout).to('cuda')
    #encoder=TCNEncoder(n_features,hidden_sizes,T,5,2).to('cuda')
    #encoder=LSTMEncoder(n_features, hidden_sizes*2, T, 2).to('cuda')
    #encoder=GRUEncoder(n_features, hidden_sizes, T).to('cuda')
    #encoder=TimesNetEncoder(T, 2, hidden_sizes, hidden_sizes, 6, n_features, 2).to('cuda') #d_ff=2nd hidden sizes,1=layers,k=2 and kernel=6 by default
    #encoder=PatchTSTEncoder(T,hidden_sizes,hidden_sizes,8,4).to('cuda') #d_model,d_ff,n_heads,n_layers
    head=nn.Linear(n_features*T,H).to('cuda') # #*4            8,4 patchtst
    model = nn.Sequential(encoder,head).to('cuda')
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, pin_memory=True,generator=g)#config['batch_size']

    valid_loader = DataLoader(dataset=valid_dataset, batch_size=bs, shuffle=False, pin_memory=True)


    pretraing_losses,model = ssl_sup_pretrainer_zeroshot(model,total_epoch,train_loader,valid_loader,optimizer,T,H,building,maxnorm,patience)
    
    #complexity: i you want to assess complexity uncommment this
    # macs, params = get_model_complexity_info(model,
    #                                         (7*24,8),   
    #                                         as_strings=True,
    #                                         print_per_layer_stat=False,
    #                                         verbose=False
    #                                         )

    # print("MACs:", macs) #     MMac (M million)
    # print("Params:", params) # 

    #zeroshot: add model
    return pretraing_losses,model


def test(test_dataset,best_config,T,n_features,H,mean,std,bs,best_model): #building,mean_y,std_y 

    model=best_model
    model.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, pin_memory=True)
    all_preds = []
    all_targets = []
    embeddings = []
    with torch.no_grad():

       for i, batch in enumerate(test_loader):
           X, y = batch ##X:bs,T,n_feat
           X, y = X.to('cuda'), y.to('cuda')

           embed=model[0](X)
           forecast=model[1](embed)
           embeddings.append(embed.cpu())
           all_preds.append(forecast.cpu())
           all_targets.append(y.cpu())
    
    y_pred = torch.cat(all_preds, dim=0).numpy()*std.cpu().numpy()+mean.cpu().numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()*std.cpu().numpy()+mean.cpu().numpy() 
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print(y_pred.shape)
    print(y_true.shape)
    # Compute metrics
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    epsilon=1e-8
    #mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)
    smape = float(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100)
    #r2 = r2_score(y_true.flatten(), y_pred.flatten())
    #medae = median_absolute_error(y_true.flatten(), y_pred.flatten())
    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "sMAPE (%)": smape
        # "MedAE": medae,
        # "R2": r2,
        # "MAPE (%)": mape
    }
    ############################### uncomment this block if you want to plot pred vs true
    # y_pred=y_pred[::168,:].reshape(-1)
    # y_true=y_true[::168,:].reshape(-1)
    # print(y_pred.shape)
    # print(y_true.shape)
    
    
    # plt.style.use('seaborn-v0_8-whitegrid')

    # fig, ax = plt.subplots(figsize=(8, 3.8))
    
    # # x-axis positions
    # x = np.arange(len(y_true))
    
    # ax.plot(x, y_true,
    #         color='black',
    #         linewidth=2,
    #         label='Ground Truth')
    
    # ax.plot(x, y_pred,
    #         color='tab:orange',
    #         linewidth=2,
    #         linestyle='--',
    #         label='Forecast')
    
    # # ---- Put this HERE, after plotting and before labels ----
    # month_pos = np.linspace(0, len(y_true)-1, 7)
    
    # ax.set_xticks(month_pos)
    # ax.set_xticklabels(['Jul', 'Aug', 'Sep', 'Oct',
    #                     'Nov', 'Dec', 'Jan'])
    # # ---------------------------------------------------------
    
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Load (kWh)')
    # ax.legend(frameon=False)
    
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.grid(alpha=0.3)
    
    # plt.tight_layout()
    # plt.show()
    # np.savez(
    # "tgt_sim_pred.npz",
    # y_true=y_true,
    # y_pred=y_pred
    # )
    # print(y_pred.shape)
    ##################################################
    
    
    ############################ uncomment this block if you want to visualize tsne plots
    # tsne = TSNE(
    # n_components=2,
    # perplexity=30,
    # random_state=42
    # )
    # z_2d = tsne.fit_transform(embeddings)
    # # # #####
    # # # # plt.scatter(
    # # # # z_2d[:, 0],
    # # # # z_2d[:, 1],
    # # # # c=y_true.mean(axis=1),#y_true[:, 23],#(y[:, -1] - y[:, 0]).cpu().numpy(),#clusters,#
    # # # # cmap="viridis",# "tab10",
    # # # # s=3#10 marker size
    # # # # )
    # # # # plt.colorbar()#(label="target")
    # # # #####
    # fig, ax = plt.subplots(figsize=(6,5), dpi=100)
    
    # sc = ax.scatter(
    #     z_2d[:,0],
    #     z_2d[:,1],
    #     c=y_true.mean(axis=1),
    #     cmap="viridis",
    #     s=8,
    #     alpha=0.75,
    #     edgecolors="none"
    # )
    
    # cbar = plt.colorbar(sc, ax=ax)
    # cbar.set_label("Mean target load (kWh)")
    
    # ax.set_xlabel("t-SNE dimension 1")
    # ax.set_ylabel("t-SNE dimension 2")
    # ax.set_xticks([])
    # ax.set_yticks([])
    
    # for spine in ax.spines.values():
    #     spine.set_visible(False)
    
    # plt.tight_layout()
    
    ############################################

    return y_true,y_pred,metrics



def integrated_gradient(model,tgt_X_test):
    feature_names=['Load','Temperature','Hour sine encoding','Hour cosine encoding',
                    'Day sine encoding','Day cosine encoding','Weekend','Holidays']
    model.eval()
    ig = IntegratedGradients(model)

    all_attr = []

    for h in range(168):
        attr_h = ig.attribute(
            inputs=tgt_X_test.to('cuda'),#[:10,...] it may take time on full test data
            target=h, #target class (regrssion in our case) index
            n_steps=25
        ) #baseline x' = 0 by default, which makes sense because inputs are normalized 
        all_attr.append(attr_h)
     
    all_attr = torch.stack(all_attr)

    # average over future steps and samples
    importance = (
        all_attr.abs()
        .mean(dim=(0,1))
    ) #(input_size, features)

    fig, ax = plt.subplots(figsize=(12, 5))

    im = ax.imshow(
        importance.T.cpu(),
        aspect='auto',
        cmap='viridis'
    )
     
    plt.colorbar(im, ax=ax, label="Importance")

    ax.set_xlabel("Historical time step")
    ax.set_ylabel("Feature")
    #ax.set_title("Overall importance across forecast horizon")

    # Replace integer labels with feature names
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(feature_names)

    plt.tight_layout()
    plt.show()