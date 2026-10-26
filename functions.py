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

    

class MLPEncoder(nn.Module):
    def __init__(self, input_size=168, n_features=1,hidden_sizes=128,dropout=0.2):#, embedding_dim=64
        super().__init__()
        
        layers_t = []
        layers_f = []
        #temporal embedding
        layers_t.append(nn.Linear(input_size, hidden_sizes))
        layers_t.append(nn.GELU())
        layers_t.append(nn.Dropout(dropout))
        layers_t.append(nn.Linear(hidden_sizes,input_size))
        #feature embedding
        layers_f.append(nn.Linear(n_features, hidden_sizes))
        layers_f.append(nn.GELU())
        layers_f.append(nn.Dropout(dropout))
        layers_f.append(nn.Linear(hidden_sizes,n_features))


        self.embed_layer_time = nn.Sequential(*layers_t)
        self.embed_layer_feature = nn.Sequential(*layers_f)

        self.norm = nn.LayerNorm(n_features)
        self.norm1 = nn.LayerNorm(n_features*input_size)
        self.norm2=nn.LayerNorm(input_size)

    
    def forward(self, x):


        x=x.permute(0,2,1)
        
        out= self.embed_layer_time(x)##bs,n_feat,T


        Res=self.norm2(out)+x #bs,n_feat,T
        Rep_out=Res.permute(0,2,1) #bs,T,n_feat
        Rep_out=self.embed_layer_feature(Rep_out) #bs,T,n_feat   
        out=Res.permute(0,2,1)+self.norm(Rep_out) #bs,T,n_feat self.gate(

        Rep_out=out.reshape(out.shape[0],-1)
        out=self.norm1(Rep_out)#+Rep_out #bs,n_Feat*T


        return out


#fewshot: no valid_dataset & no patience
def pre_training_fewshot(train_dataset,total_epoch,T,n_features,H,hidden_sizes,lr,bs,building,dropout,w,maxnorm,seed):#config, 

    encoder=MLPEncoder(T,n_features,hidden_sizes,dropout).to('cuda')#(T,n_features,hidden_sizes,dropout).to('cuda')
    #benchmarking: comment mlpencoder and activate one of the following encoders
    #encoder=TransformerEncoder(n_features,hidden_sizes,T,8,2,256,0.1).to('cuda')#(T,n_features,hidden_sizes,dropout).to('cuda')
    #encoder=TCNEncoder(n_features,hidden_sizes,T,5,2).to('cuda')
    #encoder=LSTMEncoder(n_features, hidden_sizes*2, T, 2).to('cuda')
    #encoder=GRUEncoder(n_features, hidden_sizes, T).to('cuda')
    #encoder=TimesNetEncoder(T, 2, hidden_sizes, hidden_sizes, 6, n_features, 2).to('cuda') #d_ff=2nd hidden sizes,1=layers,k=2 and kernel=6 by default
    #encoder=TSMixerEncoder(1,T,hidden_sizes,0.1,n_features).to('cuda')
    #8,1 for zeroshot
    #encoder=PatchTSTEncoder(T,hidden_sizes,hidden_sizes,8,4).to('cuda') #d_model,d_ff,n_heads,n_layers
    head=nn.Linear(n_features*T,H).to('cuda') # #*4            8,4 patchtst
    model = nn.Sequential(encoder,head).to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, pin_memory=True,generator=g)#config['batch_size']

    pretraing_losses = ssl_sup_pretrainer_fewshot(model,total_epoch,train_loader,optimizer,T,H,building,maxnorm)
    
    #complexity: i you want to assess complexity uncommment this
    # macs, params = get_model_complexity_info(model,
    #                                         (7*24,8),   
    #                                         as_strings=True,
    #                                         print_per_layer_stat=False,
    #                                         verbose=False
    #                                         )

    # print("MACs:", macs) #     MMac (M million)
    # print("Params:", params) # 

    return pretraing_losses

#zeroshot: ,valid_dataset & ,patience
def pre_training_zeroshot(train_dataset,valid_dataset, total_epoch,T,n_features,H,hidden_sizes,lr,bs,building,dropout,w,maxnorm,seed,patience):#config, 

    encoder=TSMixerEncoder(1,T,hidden_sizes,0.1,n_features).to('cuda')
    #benchmarking: comment mlpencoder and activate one of the following encoders
    #encoder=TransformerEncoder(n_features,hidden_sizes,T,8,2,256,0.1).to('cuda')#(T,n_features,hidden_sizes,dropout).to('cuda')
    #encoder=TCNEncoder(n_features,hidden_sizes,T,5,2).to('cuda')
    #encoder=LSTMEncoder(n_features, hidden_sizes*2, T, 2).to('cuda')
    #encoder=GRUEncoder(n_features, hidden_sizes, T).to('cuda')
    #encoder=TimesNetEncoder(T, 2, hidden_sizes, hidden_sizes, 6, n_features, 2).to('cuda') #d_ff=2nd hidden sizes,1=layers,k=2 and kernel=6 by default
    #8,1 for zeroshot
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


def linear_probing(hidden_sizes,linear_probing_epochs,train_dataset,valid_dataset,total_pretraining_epochs,T,n_features,H,lr,bs,building,mean,std,dropout,patience,w,maxnorm,seed):
    lowest_loss = float('inf')
    best_checkpoint = None
    save_dir = f"./checkpoints/history_{T}_horizon_{H}_{building}"
    valid_regression_losses=[]
    
    for epoch_num in range(5, total_pretraining_epochs+1, 5):
        encoder=MLPEncoder(T,n_features,hidden_sizes,dropout).to('cuda')#(T,n_features,hidden_sizes,dropout).to('cuda')
        #benchmarking: if you want to compare with other benchmark encoders
        #encoder=TransformerEncoder(n_features,hidden_sizes,T,8,2,256,0.1).to('cuda')
        #encoder=TCNEncoder(n_features,hidden_sizes,T,5,2).to('cuda')
        #encoder=LSTMEncoder(n_features,  hidden_sizes*2, T, 2).to('cuda') #h,T,1
        #encoder=GRUEncoder(n_features, hidden_sizes, T).to('cuda')
        #encoder=TimesNetEncoder(T, 2, hidden_sizes, hidden_sizes, 6, n_features, 2).to('cuda')
        #encoder=TSMixerEncoder(1,T,hidden_sizes,0.1,n_features).to('cuda')
        #encoder=PatchTSTEncoder(T,hidden_sizes,hidden_sizes,8,4).to('cuda') #d_model,d_ff,n_heads,n_layers
        head = nn.Linear(n_features*T,H).to('cuda')
        model=nn.Sequential(encoder, head)
        save_path = os.path.join(save_dir, f"encoder_epoch_{epoch_num}.pth")
        model.load_state_dict(torch.load(save_path))
        valid_min_loss,model = linear_probing_one_checkpoint(model, hidden_sizes, linear_probing_epochs,train_dataset,valid_dataset,H,lr,bs,mean,std,patience,w,maxnorm,T,n_features,seed)

        if valid_min_loss<lowest_loss:
            lowest_loss = valid_min_loss
            best_checkpoint = epoch_num

            #finetuning
            best_model=model

        valid_regression_losses.append(valid_min_loss)

    print(f'best finetuning achieved at pretraining epoch: {best_checkpoint}')
    plt.plot(range(5, total_pretraining_epochs+1, 5), valid_regression_losses, marker='o', linestyle='None')
    plt.xlabel("pretraining epoch")
    plt.ylabel("validation regression Loss")
    plt.title("validation regression loss vs Epoch")
    plt.show()
    return lowest_loss,best_checkpoint,best_model
    






#zeroshot: ,model
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
    #only for visualization of true vs pred
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
    # "lstm_pred.npz",
    # y_true=y_true,
    # y_pred=y_pred
    # )
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


    

    
    
def tgt_only_training(hidden_sizes,linear_probing_epochs,train_dataset,valid_dataset,total_pretraining_epochs,T,n_features,H,lr,bs,building,mean,std,dropout,patience,w,maxnorm,seed):

    save_dir = f"./checkpoints/history_{T}_horizon_{H}_{building}"

    encoder=MLPEncoder(T,n_features,hidden_sizes,dropout).to('cuda')#(T,n_features,hidden_sizes,dropout).to('cuda')
    head=nn.Linear(T*n_features,H).to('cuda') # #
    model = nn.Sequential(encoder,head)

    valid_min_loss,model = finetuning_one_checkpoint(model, hidden_sizes, total_pretraining_epochs,train_dataset,valid_dataset,H,lr,bs,mean,std,patience,w,maxnorm,T,n_features,seed)

    return model