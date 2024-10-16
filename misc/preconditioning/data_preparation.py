import os
from misc import ecl, grdecl
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

from input_output import read_config
from misc.preconditioning.figures import plot_tof_hist

from resdata.well import WellInfo
from resdata.grid import Grid
from resdata.resfile import ResdataFile
from resdata.summary import Summary

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

def extract_well_schedule_and_rates(simdata, root, grid, well_names):
    well_schedule = defaultdict(list)
    well_rates = defaultdict(list)

    for sim in simdata:
        rd_sum = ResdataFile(root + '.' + sim)
        WI = WellInfo(grid, rd_sum)
        for well_name in well_names:
            if well_name in  WI.allWellNames():
                # schedule
                well_schedule[well_name].append(True)
                # rates
                well_rates[well_name].append(WI[well_name][0].volumeRate())   
            else:
                well_schedule[well_name].append(False)
                well_rates[well_name].append(-np.inf)
    return well_schedule, well_rates


def extract_N_well_schedule_and_rates(simdata, root, grid, well_names, N):
    well_schedule = defaultdict(list)
    well_rates = defaultdict(list)
    well_starts = defaultdict(list) 
    for i, sim in enumerate(simdata):
        rd_sum = ResdataFile(root + '.' + sim)
        WI = WellInfo(grid, rd_sum)
        for well_name in well_names:
            if well_name in WI.allWellNames():
                if well_name not in well_starts.keys():
                    # schedule
                    well_schedule[well_name].append(True)
                    # rates
                    well_rates[well_name].append(WI[well_name][0].volumeRate())
                elif i - well_starts[well_name] < N:
                    # schedule
                    well_schedule[well_name].append(True)
                    # rates
                    well_rates[well_name].append(WI[well_name][0].volumeRate()) 
                else:
                    well_schedule[well_name].append(False)
                    well_rates[well_name].append(-np.inf)
            else:
                well_schedule[well_name].append(False)
                well_rates[well_name].append(-np.inf)
            if sum(well_schedule[well_name]) == 1:
                well_starts[well_name] = i

    return well_schedule, well_rates

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

def get_initial_date():
    #### Case dependant ####
    _, kf = read_config.read_txt('3D_ES.pipt')
    # _, kf = read_config.read_txt('drogon_1.pipt')
    ####
    return datetime.strptime(kf['startdate'], '%m/%d/%Y')

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

def extract_solver_props(root:str, data_folder:str, member:int) -> None:
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

    df_infostep_.to_csv(data_folder + os.sep + 'solver_report_' + str(member) + '.csv', sep='\t')
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

def compute_TOF(root, sim, data_folder, well_mode_dict):
    def convert_string(s):
        while s and not s[0].isdigit():
            s = s[1:]
        return str(int(s))
    step = convert_string(sim)
    # Run Time Of Flight computation
    os.chdir(data_folder)
    os.makedirs('ToF', exist_ok=True)
    os.chdir('ToF')
    command  = ['OPM-ToF']
    args = [f'case=../../{root}', f'step={step}']
    cmd = command + args
    sp.run(cmd, check=True, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
    os.chdir('../..')

    def parse_file_to_dict(file_path):
        data_dict = {}
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()
                data_dict[int(parts[0])] = float(parts[1])
        return data_dict
    
    tof_filenames = {}
    tof_dict = {}
    for filename in os.listdir(data_folder + '/ToF'):
        if len(filename.split('.')) == 2:
            name, ext = filename.split('.')
            if ext == 'out':
                prop, mode, well_name = name.split('-')
                if well_name in well_mode_dict[mode.upper()] and prop == 'tof':
                    tof_filenames[well_name] = filename
                    tof_dict[well_name] = parse_file_to_dict(data_folder + os.sep + 'ToF' + os.sep + filename)
    return tof_dict

def compute_TOF_local_domain(tof_dict, threshold):
    # Threshold in number of cells
    well_neighbors = defaultdict(dict)
    for well in tof_dict.keys():
        sorted_tof = sorted(tof_dict[well].items(), key=lambda item: item[1])
        filtered_tof = [item[0] for item in sorted_tof[:threshold]]
        well_neighbors[well]['f'] = filtered_tof
    return well_neighbors

def save_dict_to_csv(well_dict, savepath) -> None:
    for well_name, cells in well_dict.items():
        with open(savepath + os.sep + f'{well_name}_local_domain.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([well_name, cells['f']])
    return 


def main(member):
    ml_data_folder = 'En_ml_data'
    os.makedirs(ml_data_folder, exist_ok=True)

    folder = 'En_' + str(member) + os.sep
    filename = get_filename(folder)
    _, simdata, _ = get_extensions(folder)
    root = folder + os.sep + filename

    #### Case dependant ####
    well_mode_dict = {'INJ':['A5', 'A6'],
            'PROD': ['A1', 'A2', 'A3', 'A4']}
   
    # well_mode_dict = {'INJ': ['INJ1', 'INJ2', 'INJ3'], 'PROD':['PRO1', 'PRO2', 'PRO3']}
    ##### 

    well_names = well_mode_dict['INJ'] + well_mode_dict['PROD']

    # Extract solver props
    extract_solver_props(root, ml_data_folder, member)
    # Extract timesteps
    dts = extract_timetsteps(root)
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
    

    # Compute local well domain
    mode = 'NotToF'
    max_distance = 2
    if os.path.isfile(ml_data_folder + os.sep + f'well_dict_local.pkl'):
        well_local_domains = pickle.load(open(ml_data_folder + os.sep + f'well_dict_local.pkl', 'rb'))
    else:
        # need to compute ToF local domain but care parallelisation
        # first arrived member computes for all
        if mode == 'ToF':
            # Using Time of Flight
            tof_dict = compute_TOF(root, simdata[-1], ml_data_folder, well_mode_dict)
            plot_tof_hist(tof_dict, savepath=ml_data_folder + '/ToF')
            well_local_domains = compute_TOF_local_domain(tof_dict, threshold=max_distance)
        else:
            # Using arbitrary distance
            well_local_domains = bfs_extend_neighborhood(well_dict, grid, max_distance=max_distance)
        
        # before saving, other members may have gone through the if statement too, check again
        if not os.path.isfile(ml_data_folder + os.sep + f'well_dict_local.pkl'):
            # Save the dictionary to a file
            with open(ml_data_folder + os.sep + f'well_dict_local.pkl', 'wb') as f:
                pickle.dump(well_local_domains, f)
            save_dict_to_csv(well_local_domains, savepath=ml_data_folder)

    # Compute well schedule for input features, we don't care about the last timestep
    # well_schedules, well_rates = extract_well_schedule_and_rates(simdata[:-1], root, grid, well_names)
    N = 10
    well_schedules, well_rates =  extract_N_well_schedule_and_rates(simdata[:-1], root, grid, well_names, N=N)
    
    # Static Features 
    stat_props = ['PERMX']

    stat_props_np = extract_static_props(root, stat_props)

    # Static and Dynamic features extraction
    # Dynamic features
    dyn_props = ['PRESSURE', 'SWAT', 'SGAS', 'RV', 'RS']
    assert len(well_schedules[well_names[0]]) == len(simdata) - 1 == len(dts)
    # If opm creates empty file due to exceptions, dyn_props_np is None
    dyn_props_np = extract_dynamic_props(simdata[0], root, dyn_props)
    broken = False
    if dyn_props_np is not None:
        for i in range(1, len(simdata)):
            well_dyn_props = defaultdict(lambda: defaultdict(list))
            for well_name in well_names:
                # Check if well is active during this step
                if well_schedules[well_name][i - 1]:
                    # Get local property
                    local_dyn_props_np = dyn_props_np[:, well_local_domains[well_name]['f'], :]
                    well_dyn_props[well_name]['X'] = local_dyn_props_np
                    well_dyn_props[well_name]['dt'] = dts[i-1]
                    well_dyn_props[well_name]['RESV'] = well_rates[well_name][i-1]

            # Full grid feature extraction
            dyn_props_np = extract_dynamic_props(simdata[i], root, dyn_props)
            # There may be some crash at any step of the simulation, need to check and stop if so
            if dyn_props_np is None:
                print(f"Breaking due to corrupted input data at sim {i} from member {member}")
                broken = True
                break
            if broken:
                print("Not supposed to be here")
            for well_name in well_names:
                # Check if well was active during previous step
                if well_schedules[well_name][i-1]:
                    # Get local property
                    local_dyn_props_np = dyn_props_np[:, well_local_domains[well_name]['f'], :]
                    well_dyn_props[well_name]['Y'] = local_dyn_props_np
                    
            for well_name in well_names:
                if well_schedules[well_name][i-1]:
                    well_folder = ml_data_folder + os.sep + well_name
                    well_ensemble_folder = well_folder + os.sep + 'En_' + str(member)
                    os.makedirs(well_ensemble_folder, exist_ok=True)
                    np.savez(well_ensemble_folder + os.sep + str(i), stat_props=stat_props_np[:, well_local_domains[well_name]['f'],:], 
                        dyn_props_X=well_dyn_props[well_name]['X'], dyn_props_Y=well_dyn_props[well_name]['Y'], 
                        dt=well_dyn_props[well_name]['dt'], RESV=well_dyn_props[well_name]['RESV'])
                
                    