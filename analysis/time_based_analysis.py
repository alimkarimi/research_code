import pandas as pd
import numpy as np
from model_output_db import read_metrics_df
from matplotlib import pyplot as plt

df = read_metrics_df(fp = 'LSTM_w_cv.pkl')

# filter to the last epoch:
epoch_filtered_df = df[df['epoch'] == 2]
epoch_filtered_df.reset_index(drop=True, inplace=True)
len_obs = len(epoch_filtered_df.loc[0,'LAI_GTs'])
total_folds = epoch_filtered_df['k_fold'].max()

# to do:
# for each experiment (HIPS 21, 22, f54)
#       for each model (LSTM, Transformer)
            # compute RMSE, R2, rRMSE by date for test fold
            # compute overall RMSE, R2, rRMSE for test fold
            # compute variance of RMSE, R2, rRMSE by date for test fold
            # compute variance of RMSE, R2, rRMSE for test fold


# build plots for each k-fold:
for fold in range(total_folds + 1):
    # filter for the correct fold:
    fold_epoch_filtered_df = epoch_filtered_df[epoch_filtered_df['k_fold'] == fold]
    fold_epoch_filtered_df.reset_index(drop=True, inplace=True)

    # initialize timeseries dict:
    pred_dict = {}
    GT_dict = {}
    date_dict = {}

    # build keys for GT dict and preds dict. Build key/value pair for date dictionary
    for x, i in enumerate(range(len_obs)):
        pred_dict[x] = []
        GT_dict[x] = []
        date_dict[x] = fold_epoch_filtered_df.loc[0, 'dates'][x]

    # loop below appends observations to the correct key in the GT and pred dictionaries.
    for timeseries_GT, timeseries_pred in zip(fold_epoch_filtered_df['LAI_GTs'], fold_epoch_filtered_df['LAI_preds']):
        for n, (obs_GT, obs_pred) in enumerate(zip(timeseries_GT, timeseries_pred)):
            pred_dict[n].append(obs_pred)
            GT_dict[n].append(obs_GT)
            

    # get 1 to 1 plot with dates:
    plt.figure(figsize=(8,8))
    for j in range(len_obs):
        plt.scatter(GT_dict[j], pred_dict[j], label=date_dict[j])


    plt.plot([0,7], [0,7], label='One to one', color = 'black')
    #plt.title('Ground Truth vs Predicted LAI on by ' + str(field_dict[int(field_id)]) + ' After Epoch ' + str(epoch), fontsize=14)
    plt.xlabel('Ground Truth LAI', fontsize=14)
    plt.ylabel('Predicted LAI', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0,7)
    plt.ylim(0,7)
    plt.legend()
    plt.savefig(str(fold) + 'onetoonetest.jpg')
    plt.clf()
        