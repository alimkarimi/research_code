import numpy as np
import sys
sys.path.append("..")
from dataloading_scripts.read_purnima_features import get_svr_features


def get_field_metadata(field_id):
    if field_id == 1:
        field = 'hips_2021'
    if field_id == 2:
        field = 'hips_2022'
    if field_id == 3:
        field = 'hips_both_years'
    if field_id == 4:
        field = '2022_f54'
    df = get_svr_features(debug=False, data_path=field)

    return df

def get_subplot_metadata(df, plot_id):
    plot_id = np.array(plot_id.unique(), dtype=np.float64)
    # query for the pedigree, hybrid/inbred, and dates:
    pedigree = df[df['Plot'] == int(plot_id)]['pedigree'].unique()
    nitrogen_treatment = df[df['Plot'] == int(plot_id)]['nitrogen_treatment'].unique()
    hybrid_or_inbred = df[df['Plot'] == int(plot_id)]['hybrid_or_inbred'].unique()
    dates = df[df['Plot'] == int(plot_id)]['date'].unique()
    if hybrid_or_inbred.shape[0] != 1 or pedigree.shape[0] != 1 or nitrogen_treatment.shape[0] != 1:
        print('WARNING: MULTIPLE PEDIGREES OR N-TREATMENTS MIXED IN A PLOT PREDICITON')

    return *pedigree, *hybrid_or_inbred, *nitrogen_treatment, dates

field_dict = {
    1 : 'HIPS 2021',
    2 : 'HIPS 2022',
    3 : 'HIPS 2021 + 2022',
    4 : 'N-Variation 2022'
}

if __name__ == "__main__":
    df = get_field_metadata(1)