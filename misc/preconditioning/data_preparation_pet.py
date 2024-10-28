import os
import shutil
import re

import numpy as np
import pandas as pd

def get_filename(folder:str) -> None:
    base_names = [filename.split('.')[0] for filename in os.listdir(folder)]
    if all(name == base_names[0] for name in base_names):
        return base_names[0]
    else:
        return None

def find_ensemble_iter(ml_data_folder, ensemble_iter_folder, filename, member):
    i = 0
    while True:
        ensemble_iter_save_folder = os.path.join(ml_data_folder, f"{ensemble_iter_folder}_{i}")
        file_path = ensemble_iter_save_folder + os.sep + 'En_' + str(member) + "_" + filename + ".DATA"

        if not os.path.isdir(ensemble_iter_save_folder):
            return i 

        if not os.path.isfile(file_path):
            return i 
        i += 1
    return -1 

def extract_solver_props(root:str, ensemble_iter_save_folder:str, member:int ) -> None:
    with open(root + '.INFOSTEP', 'r') as f:
        lines = f.readlines()
    columns_list = lines[0].split()
    data = []
    for line in lines[1:]:
        # Split the line into values using regex for multiple spaces
        values = re.split(r'\s+', line.strip())
        # Append the values to the data list
        data.append(np.array(values, dtype=np.float64))
    df_infostep = pd.DataFrame(data, columns=columns_list)
    df_infostep_ = df_infostep[['Time(day)' , 'TStep(day)', 'NewtIt']]

    
    df_infostep_.to_csv(ensemble_iter_save_folder + os.sep + 'solver_report_' + str(member) + '.csv', sep='\t')
    return 

def main(member):
    ml_data_folder = 'En_ml_data'
    ensemble_iter_folder = 'En_iter'
    os.makedirs(ml_data_folder, exist_ok=True)

    folder = 'En_' + str(member) + os.sep
    filename = get_filename(folder)
    root = folder + os.sep + filename
    ##### Save all .DATA for future rerun #####
    ensemble_iter = find_ensemble_iter(ml_data_folder, ensemble_iter_folder, filename, member)
    ensemble_iter_save_folder = ml_data_folder + os.sep + ensemble_iter_folder + f"_{ensemble_iter}"
    os.makedirs(ensemble_iter_save_folder, exist_ok=True)

    if not os.path.isfile(ensemble_iter_save_folder + os.sep + 'En_' + str(member) + "_" + filename + ".DATA"):
        shutil.copy2(root + ".DATA", ensemble_iter_save_folder + os.sep + 'En_' + str(member) + "_" + filename + ".DATA")

    #### Save all solver reports ####
    os.makedirs(ensemble_iter_save_folder + os.sep + 'standard_newton', exist_ok=True)
    extract_solver_props(root, ensemble_iter_save_folder + os.sep + 'standard_newton', member)


                
                    