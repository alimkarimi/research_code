from matplotlib import pyplot as plt
import sys
import pandas as pd

sys.path.append('..')

from vectorization_scripts.read_purnima_features import get_svr_features

df = get_svr_features(debug=False, data_path='hips_2021')

# List of columns to keep
columns_to_keep = ['LAI', 'date', 'Plot']

# Create a boolean mask indicating which columns to keep
columns_to_drop = ~df.columns.isin(columns_to_keep)

# Drop columns that are not in the list
df.drop(df.columns[columns_to_drop], axis=1, inplace=True)

df_20210730 = df[(df['date'] == '20210727')]
plt.plot(df_20210730['Plot'], df_20210730['LAI'], label='20210727')
plt.xticks(df_20210730['Plot'], rotation=45, fontsize=4)
plt.savefig('LAI_by_plot_per_date')

print('number of unique plots:')
print(len(df_20210730['Plot'].unique()))