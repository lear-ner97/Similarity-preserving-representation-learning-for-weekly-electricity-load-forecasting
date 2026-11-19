# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:09:06 2026

@author: umroot
"""
import torch
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from itertools import combinations
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sktime.distances import erp_distance
import random
import time
from torch.optim import lr_scheduler
from scipy.stats import spearmanr


def Distance_normalizer(distance):

    if len(distance) == 1:
        Normal_distance = distance/distance
    else:
        min_val = torch.min(distance)
        max_val = torch.max(distance)

        # Normalize the distances between 0 and 1
        Normal_distance = (distance - min_val) / (max_val - min_val)
    return Normal_distance


def ssl_sup_pretrainer_zeroshot(model,total_epoch,train_loader,valid_loader,optimizer,T,H,building,maxnorm,patience):
    
    
    pretraining_losses=[]

    save_dir = f"./checkpoints/history_{T}_horizon_{H}_{building}"
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    best_mae=float('inf')

    for epoch_num in range(total_epoch):
        epoch_loss = 0  # total loss of epoch
        model = model.train()
        total_batches = 0  # total batches in epoch
        for i, batch in enumerate(train_loader):
            X, y = batch  #bs,T,n_feat
            X,y=X.to('cuda'), y.to('cuda')
            optimizer.zero_grad() #clear gradients

            rep_out_x= model[0](X) #encoder representations

            #pairwise distance between representations
            Distance_out = torch.cdist(rep_out_x, rep_out_x)**2 
            mask = torch.tril(torch.ones_like(Distance_out), diagonal=-1).bool()
            Distance_out = torch.masked_select(Distance_out, mask) #shape 2016
            Distance_out = Distance_normalizer(Distance_out)
            
            #pairwise distance between: targets if use y, inputs if use X.flatten
            
            Distance_out_y = torch.cdist(y,y)####(X.flatten(-2),X.flatten(-2))#####
            mask = torch.tril(torch.ones_like(Distance_out_y), diagonal=-1).bool()
            Distance_out_y = torch.masked_select(Distance_out_y, mask) #shape 2016
            Distance_out_y = Distance_normalizer(Distance_out_y)

            #generate forecast
            forecast= model[1](rep_out_x)
            
            #compute loss function: forecast loss+similarity loss
            total_loss = nn.SmoothL1Loss()(forecast, y)+nn.SmoothL1Loss()(Distance_out, Distance_out_y)#
            
            
            total_loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=maxnorm) 

            optimizer.step() 
            
            total_batches += 1
            epoch_loss += total_loss.item()
            
        epoch_loss = epoch_loss / total_batches  # average loss per sample for whole epoch
        
        pretraining_losses.append(epoch_loss)
        
        #zeroshot
        model.eval()
        with torch.no_grad():
            #valid_loss=0
            total_mae=0
            batches=0
            for i, batch in enumerate(valid_loader):
                X, y = batch  #bs,T,n_feat
                X,y=X.to('cuda'), y.to('cuda')
                yhat=model(X)
                mae = torch.abs(y - yhat)
                total_mae += mae.sum().item()
                batches+=1
            total_mae/=batches
            if total_mae<best_mae:
                best_mae=total_mae
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered")
                    break

    #if you want to compute spearman correlation between embedding distance and target distance
    # latent_distances = Distance_out.detach().cpu().numpy()
    # target_distances = Distance_out_y.detach().cpu().numpy()
    
    # rho, p_value = spearmanr(latent_distances, target_distances)
    
    # print(f"Spearman correlation: {rho:.4f}")
    # print(f"P-value: {p_value:.6e}")
    
    plt.figure()
    plt.plot(range(epoch_num+1), pretraining_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Pretrain Loss")
    plt.title("Pretraining Loss vs Epoch")
    plt.show()
    
    return pretraining_losses,model





