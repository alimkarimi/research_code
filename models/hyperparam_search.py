from models import statistical_model
from itertools import product
import pandas as pd

def hyperparameter_search(field_id = None):
    # Define the hyperparameter grid
    param_grid = {
        'kernel': ['rbf', 'linear'],
        'C': [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5],
        'epsilon' : [0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0] }

    # Get the list of all parameter values
    param_values = list(param_grid.values())
    print("param_values", param_values)

    # Calculate the Cartesian product of parameter values
    num_combinations = len(list(product(*param_values)))

    # Show all the different configs for hyper-parameter tuning:
    param_combinations = list(product(*param_values))


    print("Number of unique combinations:", num_combinations)

    df = pd.DataFrame()
    df['param_config'] = None
    df['rmse_dict'] = None
    df['r_2_dict'] = None
    df['stratification_dict'] = None
    df['average_r_2'] = None
    df['average_rmse'] = None

    print("Unique Hyperparameter Configurations:")
    for n, hp_config in enumerate(param_combinations):
        print('using:', hp_config, 'for ', field_id)
        svr_model = statistical_model(field_id = field_id, kernel = hp_config[0], C = hp_config[1], epsilon = hp_config[2])
        svr_results = svr_model.svr(debug=False, produce_plot=True, cross_validation=False,
                        cv_stratify=True,  groups=False, grid_search=False, verbose_stratification_and_group_data=False)
        # unpack:
        rmse_dict, r_2_dict, stratification_dict, average_r_2, average_rmse = svr_results

        # save results to df:
        df.at[n, 'param_config'] = str(hp_config)
        df.at[n, 'rmse_dict'] = rmse_dict
        df.at[n, 'r_2_dict'] = r_2_dict
        df.at[n, 'stratification_dict'] = stratification_dict
        df.at[n, 'average_r_2'] = average_r_2
        df.at[n, 'average_rmse'] = average_rmse

    print(df)
    df.to_csv('hyper_param_search_results/hyper_param_search_results_' + svr_model.field_id + '.csv')
    print('created hyper_param_search_results/hyper_param_search_results_' + svr_model.field_id + '.csv')

if __name__ == "__main__":
    # hyperparameter_search(field_id='hips_both_years')
    hyperparameter_search(field_id = 'hips_2021')
    hyperparameter_search(field_id = 'hips_2022')