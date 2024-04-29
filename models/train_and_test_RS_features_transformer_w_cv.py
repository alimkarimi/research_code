import torch
import torch.nn as nn
from torch import optim

from argparse import ArgumentParser
import sys
sys.path.append('..')

from dataloading_scripts.feature_dataloader import FeaturesDataset
from dataloading_scripts.dataloading_utils import get_field_metadata, get_subplot_metadata, field_dict

from models import Transformer
from hyperspectral_lidar_processing.hyperspectral_plot_extraction import get_visual_hyperspectral_rgb

from matplotlib import pyplot as plt
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error
from analysis.model_output_db import update_metrics_df, init_metrics_df

parser = ArgumentParser(description="Configure model hyperparameters")
parser.add_argument('-epochs', type=int, help='Epoch to pull 1 to 1 plots from', default=20)
parser.add_argument('-folds', type=int, help='Number of folds for cross validation', default=10)
parser.add_argument('-metrics_db_name', help='Name for Metrics DB. For example, transformer_w_cv', default= None)
parser.add_argument('-field', help="Field of observations - for Genotype or Management. Choices are"
                    "2022_f54, hips_2021, or hips_2022", default=None)

args = parser.parse_args()

torch.manual_seed(0) # to replicate results
epochs = args.epochs
cpu_override=False
num_folds = args.folds
field = args.field

df_name = args.metrics_db_name + '_' + args.field
print('this is DF name!!', df_name)

metrics_df = init_metrics_df(df_name=df_name) # was 'LSTM_w_cv.pkl for LSTM'
if field == 'hips_2021':
    timepoints=4
if field == 'hips_2022':
    timepoints=3
if field == '2022_f54':
    timepoints=5

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
        fig.suptitle('Prediction vs GT for Testing Split: Epoch ' + str(epoch))

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
        
        out = model(features)
        all_pred = out
        #print(final_pred.shape, all_pred.shape, GT.shape)
        # generate plot of prediction vs GT:
        

        for x in range(GT.shape[0]):
            y_true.append(float(GT[x].cpu().detach().numpy())) # append y_true to list.
            y_pred.append(float(all_pred[x].cpu().detach().numpy())) # append y_pred to list

        if plot_t_preds:
            # get the right pedigree and hybrid/inbred data for the plot:
            metadata = get_subplot_metadata(df, plot_id)
            pedigree, hybrid_inbred, nitrogen_treatment, dates = metadata
            # the lists above will be needed to compute r_2_avgs and rmse_avgs after the entire testing sample is iterated through.
            # plot GT vs Predicted:
            row_idx= n // cols  # Calculate the row index
            col_idx = n % cols   # Calculate the column index
            #print(row_idx, col_idx)
            ax[row_idx, col_idx].plot(dates, GT.cpu().detach().numpy(), label='Ground Truth')
            ax[row_idx, col_idx].plot(dates, all_pred.cpu().detach().numpy(), label = 'Prediction')
            ax[row_idx, col_idx].legend()
            if field_id <= 3: # make sure the plot title makes logical sense for the evaluation task
                ax[row_idx, col_idx].set_title('Predictions for ' + pedigree + ': ' + dates[0][0:4])
            if field_id > 3: # make sure the plot title makes logical sense for the evaluation task
                ax[row_idx, col_idx].set_title('Predictions for ' + str(nitrogen_treatment) + ': ' + dates[0][0:4])
            ax[row_idx, col_idx].set_xlabel('Date')
            ax[row_idx, col_idx].set_ylabel('LAI')

            #print('lenght is', GT.cpu().detach().numpy().shape)
            #
            # print('y_true is', y_true)
            # print('y_pred is', y_pred)

            #loss = criterion(all_pred, GT)
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
        fig.savefig('/Users/alim/Documents/prototyping/research_lab/research_code/models/GT_vs_pred_over_time/trans_GT_vs_pred_sample_epoch' + str(epoch) + '_field' + str(int(field_id)) + '_fold' + str(k_fold) + '.jpg')
        fig.clf() # clear figure for future plots
    if plot_1to1:
        # plot the 1:1 line of predictions, GT values:
        plt.figure(figsize=(8,8))
        plt.scatter(y_true, y_pred, label='Pred vs GT')
        plt.plot([0,7], [0,7], label='One to one', color = 'black')
        plt.title('Ground Truth vs Predicted LAI on ' + str(field_dict[int(field_id)]) + ' After Epoch ' + str(epoch))
        plt.xlabel('Ground Truth LAI')
        plt.ylabel('Predicted LAI')
        plt.xlim(0,7)
        plt.ylim(0,7)
        plt.annotate(f'R^2 = {r_2:.2f}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=10)
        plt.annotate(f'RMSE = {rmse:.2f}', xy=(0.05, 0.8), xycoords='axes fraction', fontsize=10)
        plt.annotate(f'rRMSE = {relative_rmse:.2f}', xy=(0.05, 0.7), xycoords='axes fraction', fontsize=10)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(loc='lower right')
        # save after creating plot:
        plt.savefig('/Users/alim/Documents/prototyping/research_lab/research_code/models/one_to_one_plots/trans_one_to_one_epoch' + str(epoch) + '_field' + str(int(field_id)) + '_fold' + str(k_fold) + '.jpg')
        plt.clf() # clear plot/figure

    

for k_fold in range(num_folds):
    # init transformer for fold:
    model = Transformer(input_size = 17, embedding_dim=17, timepoints=timepoints, num_transformer_blocks=8)

    # get number of model params:
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: {}".format(total_params))

    optimizer = optim.Adam(params=model.parameters(), lr= 1e-3, betas = (0.9, 0.99)) # instantiate new optimizer for new randomly initialized model


    # instantiate train/test dataset for the k-fold
    training_data = FeaturesDataset(field = field, train=True, test=False, return_split=k_fold)
    testing_data     = FeaturesDataset(field = field, train=False, test=True, return_split=k_fold)

    # instantiate dataloaders for train/test based on the k-fold training and testing data
    training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, num_workers = 0, drop_last=False, shuffle=True)
    testing_datalodaer  = torch.utils.data.DataLoader(testing_data,  batch_size=1, num_workers = 0, drop_last=False)

    criterion = nn.MSELoss()
    running_loss = []
    r_2_list = [] # used in function test_after_epoch
    rmse_list = [] # used in function test_after_epoch
    relative_rmse_list = [] # used in function test_after_epoch
    total_loss = 0

    # move models to correct location (GPU, CPU):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
        print('USING DEVICE:', device, '(GPU)')
    else:
        device = torch.device("cpu")
        model = model.to(device)
        print('USING DEVICE:', device, '(CPU)')

    if cpu_override:
        print('CUSTOM OVERRIDE TO CPU EVEN THOUGH GPU IS AVAILABLE')
        device = torch.device("cpu")
        model = model.to("cpu")



    for epoch in range(epochs):
        for n, timeseries in enumerate(training_dataloader):
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

            out = model(features)

            loss = criterion(out, GT)
            total_loss = total_loss + loss.item()

            loss.backward()

            optimizer.step()

            if (n + 1) % (idx_to_compute_avg_loss - 1) == 0:
                avg_loss = total_loss / 50
                running_loss.append(avg_loss)
                print('loss in epoch', epoch, ' is ', avg_loss)
                total_loss = 0
        
        print('Starting test...')
        test_after_epoch(epoch=epoch, field_id=field_id, plot_t_preds=True, plot_1to1 = True)
    print('done with fold', k_fold, '\n\n')

    # save model after running training on current fold:
    torch.save(model.state_dict(), 'transformer_models/trained_transformer_'+ args.field + 'fold_' + str(k_fold) + '.pth')


    plt.plot(running_loss)
    plt.xlabel('Epoch') # we compute loss once per epoch, so we can safely assume x-axis is the epoch.
    plt.ylabel('Loss')
    plt.title('Loss over training for ' + testing_data.field)
    plt.savefig('Transformer_loss_plots/transformer_training_running_loss_' + testing_data.field + '_fold' + str(k_fold) + '.jpg')
    plt.clf() # close figure so we can save r_2, rmse values later.
    
            




