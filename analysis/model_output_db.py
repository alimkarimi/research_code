import pandas as pd

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
    df['r2'] = None
    df['rmse'] = None
    df['r_rmse'] = None
    df['epoch'] = None
    df['plot_id'] = None
    df['field_id'] = None

    return df

def update_metrics_df(df, model_name , LAI_preds, LAI_GTs, dates, k_fold, r2, rmse, r_rmse, epoch : int, 
                      plot_id, field_id):
    # get current shape of df:
    print('inside func')
    num_rows = df.shape[0]
    print(num_rows, 'this is num rows')

    # get next row:
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
     

    df.to_pickle(model_name + '.pkl')
    df.to_csv(model_name + '.csv')

    return df

def read_metrics_df(fp=None, print_df=False):
    df = pd.read_pickle(fp)
    if print_df:
        print(df)
    return df

if __name__ == "__main__":
    df = init_metrics_df('model_run_df.pkl')
    df = update_metrics_df(df, model_name='model_name_df.pkl',LAI_preds=[1,1], LAI_GTs = [1, 1.5], dates=['2022', '2023'],
               k_fold=1, r2=1, rmse=1, r_rmse=1, epoch=1, field_id = 1, plot_id=1)
    # print(df)
    # for i in df.columns:
    #     print(i)
    #     print(df[i].values)
    # print(df.shape)

    # update_metrics_df(df, model_name='model_name_df.pkl',LAI_preds=[1,1], LAI_GTs = [1, 1.5], dates=['2022', '2023'],
    #            k_fold=1, r2=1, rmse=1, r_rmse=1, epoch=1)
    
    # print(df)
    df =read_metrics_df(fp = '/Users/alim/Documents/prototyping/research_lab/research_code/analysis/LSTM_w_cv.pkl',
                        print_df = True)

    # for i in df.columns:
    #     print(i)
    #     if i == 'LAI_preds' or i == 'LAI_GTs' or i == 'dates':
    #         print(len(df[i]))
    #         print('length is above!!')
    #         print('IN THIS ONE!!!', i)
    #         for x in df[i]:
    #             print(x)