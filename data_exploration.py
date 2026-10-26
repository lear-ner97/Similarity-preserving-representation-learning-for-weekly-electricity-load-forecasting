import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import random
#read data
path=r"data\archive (1)\electricity_cleaned.csv"
path1=r"data\archive (1)\metadata.csv"
df=pd.read_csv(path)
df1=pd.read_csv(path1)



############### source domain selection

code='Hog_office' #'Robin_lodging'#
buildings=['Valda','Bill','Mary','Bessie','Sherrie','Miriam']#['Elmer','Celia','Dorthy','Oliva']###['Casandra','Donnie','Rachael','Madge','Jewel','Jordan']
data = {m: df[f'{code}_{m}'] for m in buildings}
#data1 = {m: df1[f'{code}_{m}'] for m in buildings}
data1=df1[df1['building_id'].isin([f'{code}_{m}' for m in buildings])][['sqm']]
y=np.array(data1).astype(float)
features = []
i=0 #index for metadata (eui)
for col in buildings:
    #few-shot
    x = data[col][:(366+31+28+31)*24]# till march 31, 2017
    if col=='Celia':
         x = data[col][(366+31+28+31)*24:(366+31+28+31+28)*24] # half may to end may 2017 target data

    
    features.append({
        'meter': col,
        'mean': x.mean(), #1-distribution 
        'std': x.std(),
        'min': x.min(),
        'p10': x.quantile(0.10),
        'p50': x.quantile(0.50),
        'p90': x.quantile(0.90),
        'mean/peak': x.mean() / x.max(), 
        'sqm': y[i]

        
    })
    i+=1

features = pd.DataFrame(features).set_index('meter')

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(features)

from scipy.spatial.distance import cdist
D = cdist(X, X, metric='euclidean')
#D computes all distances between buildings
dist_sum = D.sum(axis=1)
#1-compute the medoid building
medoid_idx = dist_sum.argmin() #the medoid building has the lowest sum of distances to other buildings

medoid_meter = features.index[medoid_idx]
medoid_meter




#robin=Univ. College London (UCL), education
#weekly: ssl-sup is consistently better, daily not 50-50
# 1- robin education julius, re derick, re terrance, re jasper, re lizbeth, re takako, re billi, re kristopher
#takako =src

#moose=Ottawa - City Buildings (municipal government buildings), education
#moose education ricardo

# hog=anonymous location but 6A climate (cold humid), so probably in north america, education
# 2- hog education casandra, hog education donnie, hog education rachael, hog education madge, he jewel, hog education jordan,
#jordan=src
#sup is better than ssl-sup

#robin, lodging ssl-sup slightly better but i think with tuning of weights we can still improve the results
# 3- robin lodging dorthy, rl elmer, rl oliva, rl celia, rl janie, rl renea
#oliva=src

#hog, lodging
# 4- hog lodging shanti, hl nikki, hl ora, hl francisco, hl hal
#hal=src ssl-sup consistently better

#robin, office
# 5- robin office addie, ro serena, maryann, antonina, 
#maryann=src ssl-sup better for weekly

#hog, office
# 6- hog office  shawna, hog office sydney, hog office valda, hog office shawnna,hog office merilyn, hog office almeda, hog office bill, 
#hog office bessie, ho napoleon, ho miriam, ho shon, ho mary, ho betsy, ho lizzie, ho myles, ho sherrie, ho denita, ho mike
#bessie=src so so, done, go to lamb for more locations

#hog, other (other is another category different from education, lodging, etc but not specified, see p3 of the paper, barchart)
#hog other tobias, hog other noma

#hog, public services
#hog servies adrianna, hog services kerrie, 

#lamb, Cardiff - City Buildings, UK
# 7- lamb assembly rosa, lamb assembly librada, lamb assembly cherie, 
#librada=src so so

#hog, public assembly
# 8- hog assembly colette, ha dona, ha jasmine
#dona=src ok

#hog, public services
# 9- hog public kevin, hp octavia, hp gerard, hp crystal
#crystal=src done so so

#hog, industrial
# 10- hog industrial quentin, hi jeremy, hi mariah, hi joanne, 
#jeremy=src ok

#hog, food service (probably restaurant)
#hog food morgan, 

#hog, parking
#hog parking jean, hp shannon, 







##### uncomment below if you want to plot pred vs true for different encoders 




# Load predictions
# data1 = np.load("mlp_pred.npz")
# data2 = np.load("patchtst_pred.npz")
# data3 = np.load("lstm_pred.npz")


# #because plots will be largely overlapped, we'll plot zooms
# start =  168*11#168*40        # example start hour 43 weeks in total in the test period
# end = 168*14#168*43-1  # 14 days

# # Ground truth (same for both files)
# y_true = data1["y_true"]#[start:end]

# # Predictions
# y_pred1 = data1["y_pred"]#[start:end]
# y_pred2 = data2["y_pred"]#[start:end]
# y_pred3 = data3["y_pred"]#[start:end]

# plt.style.use('seaborn-v0_8-whitegrid')

# fig, ax = plt.subplots(figsize=(8, 3.8))

# # x-axis positions
# x = np.arange(len(y_true))

# # Ground truth
# ax.plot(x, y_true,
#         color='black',
#         linewidth=2.5,
#         label='True')

# # Model 1
# ax.plot(x, y_pred1,
#         color='tab:blue',
#         linewidth=2,
#         linestyle='--',
#         label='MLP Pred')

# # Model 2
# ax.plot(x, y_pred2,
#         color='tab:orange',
#         linewidth=2,
#         linestyle='-.',
#         label='PatchTST Pred')

# ax.plot(x, y_pred3,
#         color='tab:green',
#         linewidth=2,
#         linestyle='-.',
#         label='LSTM Pred')

# # Month labels
# month_pos = np.linspace(0, len(y_true)-1, 7)#10) 8

# ax.set_xticks(month_pos)
# ax.set_xticklabels(['Jul','Aug', 'Sep', 'Oct','Nov','Dec','Jan'])#'Apr', 'May', 'Jun',
#                     #'Jun', 'Jul'])#'Jul', 'Aug', 'Sep', 'Oct','Nov'#, 
#                     #'Jun', 'Jul'])

# ax.set_xlabel('Time')
# ax.set_ylabel('Load (kWh)')

# ax.legend(loc="best", frameon=False, fontsize=10)

# # Paper-ready formatting
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.grid(alpha=0.3)

# plt.tight_layout()
# plt.savefig("zeroshot_pred.png", dpi=100, bbox_inches='tight')
# plt.show()