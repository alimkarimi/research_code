import pandas as pd
import numpy as np
from argparse import ArgumentParser
import sys
sys.path.append('..')
from model_output_db import read_metrics_df, aggregate_for_r2_rmse

from dataloading_scripts.dataloading_utils import field_dict
from matplotlib import pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error

parser = ArgumentParser(description="Builds a 1 to 1 plot of a given model")
parser.add_argument('-epoch', type=int, help='Epoch to pull 1 to 1 plots from', default=19)
parser.add_argument('-model', type=str, help='Model name - to print out on plot', default="no model specified")
parser.add_argument('-metrics_db', help='DB with data used to compute metrics', default= 'LSTM_w_cv.pkl')

args = parser.parse_args()



def init_timebased_derived_metrics_df():
    timebased_derived_metrics_df = pd.DataFrame() # derived metrics database holds r2, rmse, relative rmse for each 
    # timestep and at each fold. It can quickly be used to compute averages, variances within a fold or across all folds.
    # it can also be used to find mins and maxes.
    timebased_derived_metrics_df['fold'] = None
    timebased_derived_metrics_df['timestep'] = None
    timebased_derived_metrics_df['r2'] = None
    timebased_derived_metrics_df['rmse'] = None
    timebased_derived_metrics_df['rel_rmse'] = None

    return timebased_derived_metrics_df

def update_timebased_derived_metrics_df(timebased_derived_metrics_df, fold, timestep, r2, rmse, rel_rmse):
    # get current shape of df:
    num_rows = timebased_derived_metrics_df.shape[0]

    timebased_derived_metrics_df.loc[num_rows, 'fold'] = fold
    timebased_derived_metrics_df.loc[num_rows,'timestep'] = timestep
    timebased_derived_metrics_df.loc[num_rows,'r2'] = r2
    timebased_derived_metrics_df.loc[num_rows,'rmse'] = rmse
    timebased_derived_metrics_df.loc[num_rows,'rel_rmse'] = rel_rmse

    return timebased_derived_metrics_df


def compute_timewise_uncertainty(y_true_tn, y_pred_tn):
    """
    y_true_tn and y_pred_tn represent objects that contain the true and estimated predictions for each timestep, tn
    """
    # compute r2
    r2_tn = r2_score(y_true_tn, y_pred_tn)
    print(r2_tn, 'this is the r2 score for time', n, 'and fold', fold)

    # compute rmse
    rmse_tn = mean_squared_error(y_true_tn, y_pred_tn, squared=False)
    print(rmse_tn, 'this is the RMSE for time', n, 'and fold', fold)

    # compute relative rmse:
    max_y_true = np.max(np.array(y_pred_tn))
    min_y_true = np.min(np.array(y_pred_tn))
    relative_rmse_tn = rmse_tn / (max_y_true - min_y_true) # relative RMSE def from Purnima's revised RMSE paper (on my iPad)
    print(relative_rmse_tn, 'this is the rRMSE for time', n, 'and fold', fold)

    return r2_tn, rmse_tn, relative_rmse_tn

df = read_metrics_df(fp = args.metrics_db)
print(df.shape)
timebased_derived_metrics_df = init_timebased_derived_metrics_df()

# filter to the last epoch:
epoch_filtered_df = df[df['epoch'] == args.epoch]
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

    field_id = fold_epoch_filtered_df['field_id'][0]

    # initialize timeseries dict:
    pred_dict = {}
    GT_dict = {}
    date_dict = {}

    # build keys for GT dict and preds dict. Build key/value pair for date dictionary
    for x, i in enumerate(range(len_obs)): # len obs is usually 3, 4, or 5. Refers to number of observations in a timeseries.
        pred_dict[x] = []
        GT_dict[x] = []
        date_dict[x] = fold_epoch_filtered_df.loc[0, 'dates'][x]

    # loop below appends observations to the correct key in the GT and pred dictionaries.
    for a, (timeseries_GT, timeseries_pred) in enumerate(zip(fold_epoch_filtered_df['LAI_GTs'], fold_epoch_filtered_df['LAI_preds'])):
        num_tn_obs_in_fold = fold_epoch_filtered_df['LAI_GTs'].shape[0] # this will be around 8, 9, or 10 in a 90/10 split.
        for n, (obs_GT, obs_pred) in enumerate(zip(timeseries_GT, timeseries_pred)):
            pred_dict[n].append(obs_pred) # this appends another estimate from t0, if n is 0
            GT_dict[n].append(obs_GT) # this appends another ground truth value from t0, if n is 0
            if (num_tn_obs_in_fold - 1 == a):
                metrics = compute_timewise_uncertainty(y_true_tn=GT_dict[n], y_pred_tn=pred_dict[n])
                r2_tn, rmse_tn, relative_rmse_tn = metrics # unpack
                update_timebased_derived_metrics_df(timebased_derived_metrics_df=timebased_derived_metrics_df,
                                                    fold=fold, timestep=n, r2=r2_tn, rmse=rmse_tn,
                                                    rel_rmse=relative_rmse_tn)
                

    # compute r2, rmse, relative rmse for entire testing set in a given fold/epoch so we can put those on the plots
    # note that time is not factored in here. r2, rmse, rRmse is computed over all time points in the func below.
    r_2, rmse, relative_rmse = aggregate_for_r2_rmse(fold_epoch_filtered_df, fold = fold, epoch = args.epoch)

    # get 1 to 1 plot with dates and metrics for each timestep:
    # plt.figure(figsize=(8,8))
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,12), gridspec_kw={'height_ratios': [3, 1], 'hspace' : 0.25})
    ax1.set_aspect('equal', adjustable='box')

    init_y_plt = 0.95
    for j in range(len_obs):
        ax1.scatter(GT_dict[j], pred_dict[j], label=date_dict[j])
        # retrieve r2, rmse, and rRmse per timestep:
        temp_df = timebased_derived_metrics_df[(timebased_derived_metrics_df['fold'] == fold) & (timebased_derived_metrics_df['timestep'] == j)]
        print('r2 to plot:', *temp_df['r2'].values)
        print('rmse to plot:', *temp_df['rmse'].values)
        print('Rrmse to plot:', *temp_df['rel_rmse'].values)
        r2_tn_plotting = temp_df['r2'].values
        rmse_tn_plotting = temp_df['rmse'].values
        rel_rmse_tn_plotting = temp_df['rel_rmse'].values
        
        # Annotation outside of the plot
        ax1.text(1.02, init_y_plt, f'R^2 for t{str(int(j))}: {float(r2_tn_plotting):.2f}', fontsize=14, transform=plt.gca().transAxes)
        ax1.text(1.3, init_y_plt, f'RMSE for t{str(int(j))}: {float(rmse_tn_plotting):.2f}', fontsize=14, transform=plt.gca().transAxes)
        ax1.text(1.62, init_y_plt, f'rRMSE for t{str(int(j))}: {float(rel_rmse_tn_plotting):.2f}', fontsize=14, transform=plt.gca().transAxes)
        init_y_plt -= 0.3


    ax1.plot([0,7], [0,7], label='One to one', color = 'black')
    ax1.set_title('Ground Truth vs Predicted LAI for ' + args.model + ' ' + str(field_dict[int(field_id)]), fontsize=14)
    ax1.set_xlabel('Ground Truth LAI', fontsize=14)
    ax1.set_ylabel('Predicted LAI', fontsize=14)
    # ax1.set_xticks(fontsize=14)
    # ax1.set_yticks(fontsize=14)
    ax1.annotate(f'R^2 = {r_2:.3f}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=14)
    ax1.annotate(f'RMSE = {rmse:.3f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=14)
    ax1.annotate(f'rRMSE = {relative_rmse:.3f}', xy=(0.05, 0.7), xycoords='axes fraction', fontsize=14)
    ax1.set_xlim(0,7)
    ax1.set_ylim(0,7)
    ax1.legend(loc='lower right')
    base_path = '/Users/alim/Documents/prototyping/research_lab/research_code/analysis/1t1/'

    # second plot:
    ax2.set_title('R2, RMSE, rRMSE by Time')
    ax2.set_xlabel('Observation Index')
    ax2.set_ylabel('Metric Value')
    time_ax_for_ax2 = np.arange(0, len(timebased_derived_metrics_df[timebased_derived_metrics_df['fold'] == fold]['r2']))
    ax2.plot(time_ax_for_ax2, timebased_derived_metrics_df[timebased_derived_metrics_df['fold'] == fold]['r2'], label="R2")
    ax2.plot(time_ax_for_ax2, timebased_derived_metrics_df[timebased_derived_metrics_df['fold'] == fold]['rmse'], label="RMSE")
    ax2.plot(time_ax_for_ax2, timebased_derived_metrics_df[timebased_derived_metrics_df['fold'] == fold]['rel_rmse'], label="rRMSE")
    ax2.set_xticks(time_ax_for_ax2)
    ax2.set_xticklabels(time_ax_for_ax2)
    ax2.set_ylim(0,1)
    ax2.legend( loc='upper right')


    fig.savefig(base_path + str(fold) + '_' + str(args.model) + '_1t1_f' + str(field_id) + '.jpg', bbox_inches='tight', dpi=1000)
    fig.clf()    