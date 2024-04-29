import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


# this file is to store data from each model run so that we can generate statistics about 
# uncertainty for predictions over time. Should answer questions like:
# How good is estimate of LAI for time point 1 vs time point 2 in a certain fold's LSTM

def init_metrics_df(df_name : str):

    df = pd.DataFrame()

    df['model_name'] = None
    df['LAI_preds'] = None
    df['LAI_GTs'] = None
    df['dates'] = None # number of elements in this column should equal number of elements in LAI_preds or LAI_GTs
    df['k_fold'] = None
    df['r2'] = None # note - these end up being computed in the function aggregate_r2_rmse()
    df['rmse'] = None # note - these end up being computed in the function aggregate_r2_rmse()
    df['r_rmse'] = None # note - these end up being computed in the function aggregate_r2_rmse()
    df['epoch'] = None
    df['plot_id'] = None
    df['field_id'] = None
 
    # save data_statistics.py
    df.to_pickle('/Users/alim/Documents/prototyping/research_lab/research_code/analysis/' + df_name + '.pkl')
    df.to_csv('/Users/alim/Documents/prototyping/research_lab/research_code/analysis/' + df_name + '.csv')

    return df

def update_metrics_df(df, model_name , LAI_preds, LAI_GTs, dates, k_fold, r2, rmse, r_rmse, epoch : int, 
                      plot_id, field_id):
    # get current shape of df:
    num_rows = df.shape[0]

    # append function args to next row:
    df.loc[num_rows, 'model_name'] = model_name
    df.at[num_rows, 'LAI_preds'] = LAI_preds
    df.at[num_rows, 'LAI_GTs'] = LAI_GTs
    df.at[num_rows, 'dates'] = dates
    df.loc[num_rows, 'k_fold'] = k_fold
    df.loc[num_rows, 'rmse'] = rmse
    df.loc[num_rows, 'r2'] = r2
    df.loc[num_rows, 'r_rmse'] = r_rmse
    df.loc[num_rows, 'epoch'] = epoch
    df.at[num_rows, 'plot_id'] = plot_id
    df.loc[num_rows, 'field_id'] = field_id
     
    # save
    df.to_pickle('/Users/alim/Documents/prototyping/research_lab/research_code/analysis/' + model_name + '.pkl')
    df.to_csv('/Users/alim/Documents/prototyping/research_lab/research_code/analysis/' + model_name + '.csv')

    return df

def read_metrics_df(fp=None, print_df=False):
    print(fp)
    df = pd.read_pickle(fp)
    if print_df:
        print(df)
    return df

def aggregate_for_r2_rmse(df, fold, epoch):
    # get a list of all the y_true and y_pred for a fold and for an epoch.
    # compute r2, rmse, r_rmse.

    # first, filter df to get the fold and epoch we care about:
    filtered_df = df[(df['epoch'] == epoch) & (df['k_fold'] == fold)] # will have one row for each timeseries
    # put through testing. For example, if the testing dataloader has 10 timeseries, there will be 10 rows in
    # filtered_df
    y_true = []
    y_pred = []

    # below, we put all the preds and GT values into a larger list, without any nested lists, so that
    # we can easily leverage r2 , rmse computation methods from scikit-learn
    for GTs, preds in zip(filtered_df['LAI_GTs'], filtered_df['LAI_preds']):
        for gt, pred in zip(GTs, preds):
            y_true.append(gt)
            y_pred.append(pred)

    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    max_y_true = np.max(np.array(y_true))
    min_y_true = np.min(np.array(y_true)) 
    relative_rmse = rmse / (max_y_true - min_y_true) # relative RMSE def from Purnima's revised RMSE paper (on my iPad)

    return r2, rmse, relative_rmse
    
        

if __name__ == "__main__":    
    df =read_metrics_df(fp = '/Users/alim/Documents/prototyping/research_lab/research_code/analysis/LSTM_w_cv_hips_2021.pkl',
                        print_df = True)
    
    for i in range(10):
        result = aggregate_for_r2_rmse(df, fold=i, epoch=19)
        print(result)    
