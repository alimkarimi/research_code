from argparse import ArgumentParser
from models import RNN

import torch
import torch.nn as nn
from torch import optim

import sys
sys.path.append('..')
from dataloading_scripts.feature_dataloader import FeaturesDataset
from dataloading_scripts.read_purnima_features import get_svr_features
from analysis.model_output_db import update_metrics_df, init_metrics_df
from dataloading_scripts.dataloading_utils import get_field_metadata, get_subplot_metadata, field_dict


from matplotlib import pyplot as plt

import numpy as np

from sklearn.metrics import r2_score, mean_squared_error

parser = ArgumentParser(description="Configure model hyperparameters")
parser.add_argument('-epochs', type=int, help='Epoch to pull 1 to 1 plots from', default=20)
parser.add_argument('-folds', type=int, help='Number of folds for cross validation', default=10)
parser.add_argument('-metrics_db_name', help='Name for Metrics DB', default= None)
parser.add_argument('-field', help="Field of observations - for Genotype or Management. Choices are"
                    "2022_f54, hips_2021, or hips_2022", default=None)

args = parser.parse_args()


epochs = args.epochs
criterion = nn.MSELoss()
batch_size = 1
num_folds = args.folds # number of k-fold validations to run

torch.manual_seed(0)
np.random.seed(0)
df_name = args.metrics_db_name + '_' + args.field
print('this is DF name!!', df_name)

metrics_df = init_metrics_df(df_name=df_name) # was 'LSTM_w_cv.pkl for LSTM'

y_pred_all_folds = [] # to compute r2, rmse, and rel rmse for ALL testing data
y_true_all_folds = [] # to compute r2, rmse, and rel rmse for ALL testing data

for k_fold in range(num_folds):

    # instantiate new model for each fold
    rnn  = RNN(batch_size = batch_size, concat_based_LSTM = True, addition_based_LSTM = False,
            hidden_size = 100, cell_size = 100) # lstm gets instantiated inside RNN class.
    rnn = rnn.to(torch.float32)

    # instantiate optimizer
    optimizer = torch.optim.Adam(rnn.parameters(), lr = 0.0005, betas = (0.9, 0.99))

    cpu_override = False

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        rnn = rnn.to(device)
        print('USING DEVICE:', device)
    else:
        device = torch.device("cpu")
        rnn = rnn.to(device)
        print('USING DEVICE:', device)

    if cpu_override:
        print('CUSTOM OVERRIDE TO CPU EVEN THOUGH GPU IS AVAILABLE')
        device = torch.device("cpu")
        rnn = rnn.to("cpu")

    # instantiate dataset
    training_data = FeaturesDataset(field = args.field, train=True, test=False, return_split=k_fold)
    testing_data     = FeaturesDataset(field = args.field, train=False, test=True, return_split=k_fold)

    # instantiate dataloaders for train/test
    training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, num_workers = 0, drop_last=False, shuffle=True)
    testing_datalodaer  = torch.utils.data.DataLoader(testing_data,  batch_size=1, num_workers = 0, drop_last=False)

    running_loss = []
    r_2_list = [] # used in function test_after_epoch
    rmse_list = [] # used in function test_after_epoch
    relative_rmse_list = [] # used in function test_after_epoch
    total_loss = 0


    def test_after_epoch(epoch, field_id, plot_t_preds=True, plot_1to1=True, metrics_df = metrics_df):
        df = get_field_metadata(field_id) # load the right dataframe so we can get the pedigree/genotype data when plotting.
        
        """
        Currently, this only tests where batch size = 1. Update to accomodate larger batch size. 
        """
        y_pred = [] # we reset this to empty for every epoch
        y_true = [] # we reset this to empty for every epoch

        if plot_t_preds:
            num_testing_samples = len(testing_datalodaer)
            print('length of test', num_testing_samples)
            rows = 2
            cols = 5
            fig, ax = plt.subplots(rows, cols, figsize=(30,12))
            fig.suptitle('Prediction vs GT for Testing Split in : Epoch ' + str(epoch), fontsize=20)
            fig.subplots_adjust(hspace=0.3, wspace=0.3)


        for n, testing_sample in enumerate(testing_datalodaer):
            features, GT, plot_id, field_id = testing_sample

            if device == torch.device("mps") or device == torch.device("cuda"): # handle input to GPU
                features = features.to(torch.float32) 
                features = features.to(device)

                # if input and network are in float32, we want to move the GT to float 32 as well
                # because we want precision of input/target in loss computation to be the same:
                GT = GT.to(torch.float32)
                GT = GT.to(device) # move GT to GPU

            features = torch.squeeze(features, 0)

            GT = torch.squeeze(GT, 0)
            
            out = rnn(features)
            _, _, final_pred, all_pred = out
            #print( all_pred.shape, GT.shape)
            # generate plot of prediction vs GT:
            

            for x in range(GT.shape[0]):
                y_true.append(float(GT[x].cpu().detach().numpy())) # append y_true to list.
                y_pred.append(float(all_pred[x].cpu().detach().numpy())) # append y_pred to list

                if epoch == (epochs - 1): # save predictions and GT for last epoch in training - this computes
                    # the rmse, rRMSE, and r**2 for all testing data.
                    y_true_all_folds.append(float(GT[x].cpu().detach().numpy())) 
                    y_pred_all_folds.append(float(all_pred[x].cpu().detach().numpy()))

            if plot_t_preds:
                # get the right pedigree and hybrid/inbred data for the plot:
                metadata = get_subplot_metadata(df, plot_id)
                pedigree, hybrid_inbred, nitrogen_treatment, dates = metadata
                dates_xaxis = [date[4:] for date in dates]
                # the lists above will be needed to compute r_2_avgs and rmse_avgs after the entire testing sample is iterated through.
                # plot GT vs Predicted:
                row_idx= n // cols  # Calculate the row index
                col_idx = n % cols   # Calculate the column index
                #print(row_idx, col_idx)
                ax[row_idx, col_idx].plot(dates_xaxis, GT.cpu().detach().numpy(), label='Ground Truth')
                ax[row_idx, col_idx].plot(dates_xaxis, all_pred.cpu().detach().numpy(), label = 'Prediction')
                ax[row_idx, col_idx].legend(fontsize=14)
                if field_id <= 3: # make sure the plot title makes logical sense for the evaluation task
                    ax[row_idx, col_idx].set_title('Predictions for ' + pedigree + ': ' + dates[0][0:4])
                if field_id > 3: # make sure the plot title makes logical sense for the evaluation task
                    ax[row_idx, col_idx].set_title('Predictions for N Treatment ' + str(nitrogen_treatment) + ': ' + dates[0][0:4])
                ax[row_idx, col_idx].set_xlabel('Date', fontsize=16)
                ax[row_idx, col_idx].set_ylabel('LAI', fontsize=16)
                ax[row_idx, col_idx].tick_params(axis='x', labelsize=16)
                ax[row_idx, col_idx].tick_params(axis='y', labelsize=16)

            # update metrics db:
            lookback = GT.shape[0]
            # print(plot_id.shape)
            metrics_df = update_metrics_df(df = metrics_df, model_name=df_name,
                                        LAI_preds=y_pred[-lookback:], LAI_GTs=y_true[-lookback:], dates=dates, k_fold=k_fold, r2 = None, rmse = None, r_rmse=None,
                                        epoch=epoch, plot_id = plot_id[0, 0].item(), field_id = field_id.item())


        r_2 = r2_score(y_true, y_pred) # compute r_2 
        rmse = mean_squared_error(y_true, y_pred, squared=False) 
        max_y_true = np.max(np.array(y_true))
        min_y_true = np.min(np.array(y_true)) 
        relative_rmse = rmse / (max_y_true - min_y_true) # relative RMSE def from Purnima's revised RMSE paper (on my iPad)

        r_2_list.append(r_2) # save r_2 for each epoch. List isn't currently used anywhere
        rmse_list.append(rmse) # save RMSE for each epoch. List isn't currently used anywhere
        relative_rmse_list.append(relative_rmse) # save relative RMSE for each epoch. List isn't currently used anywhere
        print('r_2, rmse, relative_rmse testing after epoch' , epoch, ': ', r_2, rmse, relative_rmse)


        if plot_t_preds:
            fig.savefig('/Users/alim/Documents/prototyping/research_lab/research_code/models/GT_vs_pred_over_time/GT_vs_pred_sample_epoch' + str(epoch) + '_field' + str(int(field_id)) + '_fold' + str(k_fold) + '.jpg', 
            dpi=500)
            fig.clf() # clear figure for future plots
        if plot_1to1:
            # plot the 1:1 line of predictions, GT values:
            plt.figure(figsize=(8,8))
            plt.scatter(y_true, y_pred, label='Pred vs GT')
            plt.plot([0,7], [0,7], label='One to one', color = 'black')
            plt.title('Ground Truth vs Predicted LAI on ' + str(field_dict[int(field_id)]) + ' After Epoch ' + str(epoch), fontsize=14)
            plt.xlabel('Ground Truth LAI', fontsize=14)
            plt.ylabel('Predicted LAI', fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlim(0,7)
            plt.ylim(0,7)
            plt.annotate(f'R^2 = {r_2:.2f}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=14)
            plt.annotate(f'RMSE = {rmse:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=14)
            plt.annotate(f'rRMSE = {relative_rmse:.2f}', xy=(0.05, 0.7), xycoords='axes fraction', fontsize=14)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend(loc='lower right', fontsize=14)
            # save after creating plot:
            plt.savefig('/Users/alim/Documents/prototyping/research_lab/research_code/models/one_to_one_plots/one_to_one_epoch' + str(epoch) + '_field' + str(int(field_id)) + '_fold' + str(k_fold) + '.jpg',
                        dpi=300)
            plt.clf() # clear plot/figure

            


    # training loop below. Note, not possible to do batched gradient descent. Should implement this, so that we can find a better optimized 
    # function.
    for epoch in range(epochs):
        field_id = None
        for i, timeseries in enumerate(training_dataloader):
            optimizer.zero_grad()
            idx_to_compute_avg_loss = len(training_dataloader)

            features, GT, plot_id, field_id_holder = timeseries
            field_id = field_id_holder # save field id for bringing up metadata in testing.
            if device == torch.device("mps") or device == torch.device("cuda"): # handle input to GPU
                features = features.to(torch.float32) 
                features = features.to(device)

                # if input and network are in float32, we want to move the GT to float 32 as well
                # because we want precision of input/target in loss computation to be the same:
                GT = GT.to(torch.float32)
                GT = GT.to(device) # move GT to GPU
            
            features = torch.squeeze(features, 0)
            GT = torch.squeeze(GT, 0)
            
            
            
            out = rnn(features)
            
            _, _, final_pred, all_pred = out

            loss = criterion(all_pred, GT) # compute loss - should be on same device
            #print('loss is', loss)

            loss.backward() # compute gradient of loss wrt each parameter
            optimizer.step() # take a step based on optimizer learning rate and hyper-parameters.
            total_loss += loss.item()

            if (i + 1) % (idx_to_compute_avg_loss - 1) == 0:
                avg_loss = total_loss / idx_to_compute_avg_loss
                running_loss.append(avg_loss)
                print("Loss in epoch", epoch, " is ", avg_loss)
                total_loss = 0

        print('KICKING OFF TEST AFTER EPOCH')
        test_after_epoch(epoch = epoch, field_id = field_id)


    # print training loss curve after training run:
    plt.plot(running_loss)
    plt.xlabel('Epoch') # we can say epoch here becaue we are computing the average loss at the end of the 
    # training for the epoch.
    plt.ylabel('Loss')
    plt.title('Loss over training for ' + testing_data.field + '_fold ' + str(k_fold))
    plt.savefig('LSTM_loss_plots/training_running_loss_' + testing_data.field + '_fold' + str(k_fold) + '.jpg')
    plt.clf() # close figure so we can save r_2, rmse values later.

    # print rmse and r_2 after each epoch:
    plt.plot(r_2_list, label='R^2 values')
    plt.plot(rmse_list, label = 'RMSE values')
    plt.title('R^2 and RMSE on Test Data - ' + testing_data.field)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE or R_2')
    plt.legend()
    plt.savefig('r_2_rmses/r_2_and_rmse_over_training_' + testing_data.field + '_fold' + str(k_fold) + '.jpg')


    # save the model:
    torch.save(rnn.state_dict(), 'LSTM_models/trained_rnn_model_'+ args.field + 'fold_' + str(k_fold) + '.pth')

    print('done with fold', k_fold)

# compute overall r2, rmse, rel rmse:
r_2 = r2_score(y_true_all_folds, y_pred_all_folds) # compute r_2 
rmse = mean_squared_error(y_true_all_folds, y_pred_all_folds, squared=False) 
max_y_true = np.max(np.array(y_true_all_folds))
min_y_true = np.min(np.array(y_true_all_folds)) 
relative_rmse = rmse / (max_y_true - min_y_true) # relative RMSE def from Purnima's revised RMSE paper (on my iPad)
print(len(y_pred_all_folds), len(y_true_all_folds))
print('FINAL STATISTICS:')
print('R**2:', r_2)
print('RMSE:', rmse)
print('rRMSE:', relative_rmse)
