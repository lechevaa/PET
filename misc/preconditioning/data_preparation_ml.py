
import os
import numpy as np
import datetime as dt
from collections import deque
import pickle
from collections import defaultdict
from datetime import date, datetime
import subprocess as sp
import csv
import pandas as pd
import re
import shutil
import time 

from input_output import read_config

from resdata.well import WellInfo
from resdata.grid import Grid
from resdata.resfile import ResdataFile
from resdata.summary import Summary

from misc.preconditioning.timing import Timer, measure_function

def get_filename(folder:str) -> None:
    base_names = [filename.split('.')[0] for filename in os.listdir(folder)]
    if all(name == base_names[0] for name in base_names):
        return base_names[0]
    else:
        return None

def get_extensions(folder):
    exts =  [filename.split('.')[1] for filename in os.listdir(folder)]
    sums, simdata, misc_ext = [], [], []
    for ext in exts:
        if ext.startswith('S0'):
            sums.append(ext)
        elif ext.startswith('X0'):
            simdata.append(ext)
        else:
            misc_ext.append(ext)
    return sorted(sums), sorted(simdata), misc_ext

def update_well_dict(well_dict, root, sim, grid, well_names):
    rd_sum = ResdataFile(root + '.' + sim)
    WI = WellInfo(grid, rd_sum)
    for well_name in well_names:
        if well_name in  WI.allWellNames():
            cells_ijk, cells_f, cells_g = [], [], []
            well_state = WI[well_name][0]
            for conn in  well_state.globalConnections():
                ijk = conn.ijk()
                f = grid.get_active_index(ijk=conn.ijk())
                g = grid.global_index(ijk=conn.ijk())
                cells_ijk.append(ijk)
                cells_f.append(f)
                cells_g.append(g)
            well_dict[well_name] = {'ijk': cells_ijk, 'f': cells_f, 'g':cells_g}
    return well_dict


def compute_well_start(well_issue_dates, dts):
    well_start = {}
    cumulative_dts = [0.]
    tot = 0
    for dt in dts:
        tot += dt
        cumulative_dts.append(tot)
   
    for well_name in well_issue_dates.keys():
        well_start[well_name] = cumulative_dts.index(well_issue_dates[well_name]  * 60 * 60 * 24) + 1
    return well_start


def extract_N_well_schedule_and_rates(simdata, root, grid, well_names, well_start, N):
     # If well issue dates is specified, set up the well shchedule at this point in time, otherwise at first opening of the well
     # If a well opening happens at a time t. Then its consequences within the restart files are shown are t+1
    if isinstance(N, int):
        Ns = {well_name: N for well_name in well_names}
    else:
        Ns = N

    def should_start_now(i, well_name, well_start):
        """Check if the current time index is the well's designated start time."""
        return well_name in well_start and i == well_start[well_name]

    def get_well_rate(well_info, well_name):
        """Safely get the well rate if available, else return -inf."""
        try:
            return well_info[well_name][0].volumeRate()
        except (KeyError, IndexError):
            return -np.inf

    # Initialize
    well_schedule = defaultdict(list)
    well_rates = defaultdict(list)
    well_starts = {}  

    for i, sim_name in enumerate(simdata):
        res_data = ResdataFile(root + '.' + sim_name)
        well_info = WellInfo(grid, res_data)
        available_wells = well_info.allWellNames()

        for well_name in well_names:
            is_active = False
            rate = -np.inf
            if well_name in available_wells:
                if well_name not in well_starts:
                    if should_start_now(i, well_name, well_start):
                        is_active = True
                    elif well_name not in well_start:
                        # No start constraint; activate immediately
                        is_active = True
                else:
                    # Check if well is still within its scheduled active duration
                    if i - well_starts[well_name] < Ns[well_name]:
                        is_active = True

                if is_active:
                    rate = get_well_rate(well_info, well_name)

                # Record the start index when the well first becomes active (indicated in the next file)
                if is_active and well_name not in well_starts:
                    well_starts[well_name] = i

            # Store the schedule and rate
            well_schedule[well_name].append(is_active)
            well_rates[well_name].append(rate)
    return well_schedule, well_rates


def compute_Ns(member, en_i, ml_data_folder, well_names, well_issue_dates_in_days, report_timestep_in_days):
    report_folder = ml_data_folder + os.sep + f"En_iter_{en_i}"+ os.sep +  "hybrid_newton" + os.sep + f"solver_report_{member}.csv"
    solver_df = pd.read_csv(report_folder, sep='\t')
    # Extract dt/ Newton report 
    Ns = {}
    newt = solver_df['NewtIt'].values
    if 21 not in newt:
        # If no convergence issue (most of data), then full report converges and we get only one data point in the end
        Ns = {well_name: 2 for well_name in well_names}
        return Ns
    else:
        # At least one time step failed. Need to check if it is in a report at a detected well opening issue
        for well_name in well_names:
            start = well_issue_dates_in_days[well_name]
            end = well_issue_dates_in_days[well_name] + report_timestep_in_days[well_name]
            count = solver_df[(solver_df['Time(day)'].between(start, end)) & (solver_df['NewtIt'] < 21)].shape[0]

            Ns[well_name] = count

    return Ns

def extract_dynamic_props(sim, root, props):
    prop_np = []
    rd = ResdataFile(root + '.' + sim)
    for prop in props:
        # Sometimes 'Pressure' is not recognized
        try:
            p_np = np.array(rd[prop])
            p_np =  np.reshape(p_np, (1, p_np.shape[1], 1))
            prop_np.append(p_np)
        except KeyError:
            # if one key, is missing, return None
            print("KeyError: ", prop, " at ", root + '.' + sim)
            return None
    # There may be some OPM crash that generates files but with no property
    props_np = np.concatenate(prop_np, axis=2)
    return props_np

def extract_timetsteps(root):
    rd_sum = Summary(root)
    dates = rd_sum.dates
    dts = [(dates[i] - dates[i-1]).total_seconds() for i in range(1, len(dates))]
    return dts

def extract_static_props(root:str, props) -> np.array:
    rd = ResdataFile(root + '.INIT')
    prop_np = []
    for prop in props:
        p_np = np.array(rd[prop])
        p_np =  np.reshape(p_np, (1, p_np.shape[1], 1))
        prop_np.append(p_np)
    props_np = np.concatenate(prop_np, axis=2)
    return props_np

def extract_solver_props(root:str, ensemble_iter_save_folder:str, member:int ) -> None:
    for attempt in range(1, 4):
        if os.path.exists(root + '.INFOSTEP'):
            break
        time.sleep(5)
        if attempt == 3:
            print(f"{root} not found, OPM save issue or server lag.")
        return 
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

def bfs_extend_neighborhood(well_dict, grid, max_distance):
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    well_neighbors = defaultdict(dict)
    for well in well_dict.keys():
        well_cells = well_dict[well]['ijk']
        neighborhood = set(well_cells)
        visited = set(well_cells)
        queue = deque([(cell, 0) for cell in well_cells])
        while queue:
            current_cell, distance = queue.popleft()
            if distance >= max_distance:
                continue
            i, j, k = current_cell

            # Check neighbors in each of the 6 directions
            neighbors = [
                (i-1, j, k), (i+1, j, k),
                (i, j-1, k), (i, j+1, k),
                (i, j, k-1), (i, j, k+1)
            ]
            for neighbor in neighbors:
                if neighbor not in visited:
                    ni, nj, nk = neighbor

                    if (0 <= ni < nx) and (0 <= nj < ny) and (0 <= nk < nz) and grid.active(ijk=(ni, nj, nk)):
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
                        neighborhood.add(neighbor)
        well_neighbors[well]['ijk'] = list(neighborhood)
        well_neighbors[well]['f'] = [grid.get_active_index(ijk=ijk) for ijk in well_neighbors[well]['ijk']]
        well_neighbors[well]['g'] = [grid.global_index(ijk=ijk) for ijk in well_neighbors[well]['ijk']]
    return well_neighbors

def save_dict_to_csv(well_dict, savepath) -> None:
    for well_name, cells in well_dict.items():
        with open(savepath + os.sep + f'{well_name}_local_domain.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([well_name, cells['f']])
    return 

def clean_folder(folder_path):
    shutil.rmtree(folder_path)
    return 

def dataset_cleaning(ml_data_folder, well_mode_dict):
    well_names = well_mode_dict['INJ'] + well_mode_dict['PROD']
    for well_name in well_names:
        path = ml_data_folder + os.sep + well_name
        shutil.rmtree(path)
    return 

def dataset_preparation(member, en_i, well_mode_dict, report_timesteps, well_issue_dates, En_member_timer):
    ml_data_folder = 'En_ml_data'
    ensemble_iter_folder = 'En_iter'
    folder = 'En_' + str(member) + os.sep
    filename = get_filename(folder)
    _, simdata, _ = get_extensions(folder)
    root = folder + os.sep + filename
    well_names = well_mode_dict['INJ'] + well_mode_dict['PROD']

    # Extract solver props
    ensemble_iter_save_folder = ml_data_folder + os.sep + ensemble_iter_folder + f"_{en_i}"
    os.makedirs(ensemble_iter_save_folder + os.sep + 'hybrid_newton', exist_ok=True)

    with En_member_timer.timer("Extract solver props"):
        extract_solver_props(root, ensemble_iter_save_folder + os.sep + 'hybrid_newton', member)
    
    Ns = compute_Ns(member, en_i, ml_data_folder, well_names, well_issue_dates, report_timesteps)

    # Extract timesteps
    with En_member_timer.timer("Extract timesteps"):
        dts = extract_timetsteps(root)
    
    with En_member_timer.timer("Extract well cells"):
        # Extract all well cells
        well_dict = {}
        grid = Grid(root + '.EGRID')
        for sim in simdata:
            if not sorted(list(well_dict.keys())) == sorted(well_names):
                well_dict = update_well_dict(well_dict, root, sim, grid, well_names)
            else:
                # Save the dictionary to a file
                with open(ml_data_folder + os.sep + 'well_dict.pkl', 'wb') as f:
                    pickle.dump(well_dict, f)
                break
    
    with En_member_timer.timer("Compute local well domain"):
        # Compute local well domain
        max_distance = 10
        if os.path.isfile(ml_data_folder + os.sep + f'well_dict_local.pkl'):
            well_local_domains = pickle.load(open(ml_data_folder + os.sep + f'well_dict_local.pkl', 'rb'))
        else:
            # first arrived member computes for all
            # Using arbitrary distance
            well_local_domains = bfs_extend_neighborhood(well_dict, grid, max_distance=max_distance)
            
            # before saving, other members may have gone through the if statement too, check again
            if not os.path.isfile(ml_data_folder + os.sep + f'well_dict_local.pkl'):
                # Save the dictionary to a file
                with open(ml_data_folder + os.sep + f'well_dict_local.pkl', 'wb') as f:
                    pickle.dump(well_local_domains, f)
                save_dict_to_csv(well_local_domains, savepath=ml_data_folder)

    # Compute well schedule for input features, we don't care about the last timestep
    with En_member_timer.timer("Extract N well schedule and rates"):
        well_start = compute_well_start(well_issue_dates, dts)
        well_schedules, well_rates =  extract_N_well_schedule_and_rates(simdata[:-1], root, grid, well_names, well_start, N=Ns)

    with En_member_timer.timer("Extract static props"):
        # Static Features 
        stat_props = ['PERMX']
        stat_props_np = extract_static_props(root, stat_props)
    
    assert len(well_schedules[well_names[0]]) == len(simdata) - 1 == len(dts)

   
    with En_member_timer.timer("Extract dynamic solver props"):
        # Iterate over well schedules 
        dyn_props = ['PRESSURE', 'SWAT', 'SGAS', 'RV', 'RS']
        for well_name in well_names:
            # Prepare folder for save 
            well_folder = ml_data_folder + os.sep + well_name
            well_ensemble_folder = well_folder + os.sep + ensemble_iter_folder + '_' + str(en_i)+ os.sep + str(member)
            os.makedirs(well_ensemble_folder, exist_ok=True)

            # Find schedule indexes where well data shall be saved
            well_schedule_indexes = [index - 1 for index, value in enumerate(well_schedules[well_name]) if value]
            # Extract first data
            prev_dyn_props_np = extract_dynamic_props(simdata[well_schedule_indexes[0]], root, dyn_props)
            prev_local_dyn_props_np = prev_dyn_props_np[:, well_local_domains[well_name]['f'], :]
       
            for i in range(1, len(well_schedule_indexes)):
                # Construct dataset from previous and current data
                well_dyn_props = defaultdict(lambda: defaultdict(list))
                well_dyn_props[well_name]['X'] = prev_local_dyn_props_np
                well_dyn_props[well_name]['dt'] = dts[well_schedule_indexes[i-1]]
                well_dyn_props[well_name]['RESV'] = well_rates[well_name][well_schedule_indexes[i]]
    
                # Extract current data
                current_dyn_props_np = extract_dynamic_props(simdata[well_schedule_indexes[i]], root, dyn_props)
                current_local_dyn_props_np = current_dyn_props_np[:, well_local_domains[well_name]['f'], :]

                well_dyn_props[well_name]['Y'] = current_local_dyn_props_np
                # save data
                np.savez(well_ensemble_folder + os.sep + str(well_schedule_indexes[i-1]), stat_props=stat_props_np[:, well_local_domains[well_name]['f'],:], 
                    dyn_props_X=well_dyn_props[well_name]['X'], dyn_props_Y=well_dyn_props[well_name]['Y'], 
                    dt=well_dyn_props[well_name]['dt'], RESV=well_dyn_props[well_name]['RESV'])
                
                # previous data <-- current data 
                prev_local_dyn_props_np = current_local_dyn_props_np
                
    
    clean_folder(folder_path=folder)
                    
    
                    
