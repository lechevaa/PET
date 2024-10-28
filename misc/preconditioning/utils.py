import os
import sys
import json 
from collections import defaultdict
from typing import List

import numpy as np
from sklearn.linear_model import LinearRegression


def allow_kerasify_import() -> None:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    kerasify_dir = os.path.join(parent_dir, 'opm/opm-common/python/opm/ml/ml_tools')
    sys.path.insert(0, kerasify_dir)



def scalers_to_json(scalers, ml_model_folder, well_name) -> None:
    if len(scalers) == 4:
        scalers_X, scalers_Y, scaler_dt, scaler_resv = scalers
        scaler_wld = None
    elif len(scalers) == 5:
        scalers_X, scalers_Y, scaler_dt, scaler_resv, scaler_wld = scalers

    scaler_fold =  ml_model_folder + os.sep + 'scalers'
    os.makedirs(scaler_fold, exist_ok=True)
    os.makedirs(scaler_fold + os.sep + well_name, exist_ok=True)

    scaler_params_X = defaultdict(list)
    for scaler in scalers_X:
        scaler_params_X['data_min_'].append(float(scaler.data_min_))
        scaler_params_X['data_max_'].append(float(scaler.data_max_))

    scaler_params_Y = defaultdict(list)
    for scaler in scalers_Y:
        scaler_params_Y['data_min_'].append(float(scaler.data_min_))
        scaler_params_Y['data_max_'].append(float(scaler.data_max_))

    scaler_params_dt = {'data_min_': [float(scaler_dt.data_min_)], 'data_max_': [float(scaler_dt.data_max_)]}
    scaler_params_resv = {'data_min_': [float(scaler_resv.data_min_)], 'data_max_': [float(scaler_resv.data_max_)]}

    if scaler_wld:
        scaler_params_wld = {'data_min_': [float(scaler_resv.data_min_)], 'data_max_': [float(scaler_resv.data_max_)]}
        with open(scaler_fold + os.sep + well_name + os.sep + 'wld_scaler.json', 'w') as f:
            json.dump(scaler_params_wld, f)

    with open(scaler_fold + os.sep + well_name + os.sep + 'X_scaler.json', 'w') as f:
        json.dump(scaler_params_X, f)

    with open(scaler_fold + os.sep + well_name + os.sep + 'Y_scaler.json', 'w') as f:
        json.dump(scaler_params_Y, f)

    with open(scaler_fold + os.sep + well_name + os.sep + 'dt_scaler.json', 'w') as f:
        json.dump(scaler_params_dt, f)

    with open(scaler_fold + os.sep + well_name + os.sep + 'RESV_scaler.json', 'w') as f:
        json.dump(scaler_params_resv, f)
    return 

def well_models_ready_to_json(ml_model_folder:str, well_models_ready: List[str]) -> None:
    well_models_ready = [well for well in well_models_ready if well]
    with open(ml_model_folder + os.sep + 'well_models_ready.json', 'w') as f:
        json.dump(well_models_ready, f)
    return 

def perform_linear_regression(true_values, predicted_values):
    # Reshape data for sklearn
    true_values = np.array(true_values).reshape(-1, 1)
    predicted_values = np.array(predicted_values).reshape(-1, 1)
    # Fit linear regression
    reg = LinearRegression().fit(predicted_values, true_values)
    slope = reg.coef_[0][0]
    intercept = reg.intercept_[0]
    r_squared = reg.score(predicted_values, true_values)
    return slope, intercept, r_squared