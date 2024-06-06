import os
from misc import ecl, grdecl
import numpy as np
import datetime as dt
from collections import deque
import pickle
from collections import defaultdict
from datetime import date, datetime
import subprocess as sp

from input_output import read_config
from misc.preconditioning.figures import plot_tof_hist

from resdata.well import WellInfo
from resdata.grid import Grid
from resdata.resfile import ResdataFile
from resdata.summary import Summary

def get_filename(folder):
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

def get_initial_date():
    _, kf = read_config.read_txt('drogon_1.pipt')
    return datetime.strptime(kf['startdate'], '%m/%d/%Y').date()

def update_well_dict(well_dict, root, sim, grid, well_names):
    rd_sum = ResdataFile(root + '.' + sim)
    WI = WellInfo(grid, rd_sum)
    for well_name in well_names:
        if well_name in  WI.allWellNames():
            cells_ijk, cells_f = [], []
            well_state = WI[well_name][0]
            for conn in  well_state.globalConnections():
                ijk = conn.ijk()
                f = grid.get_active_index(ijk=conn.ijk())
                cells_ijk.append(ijk)
                cells_f.append(f)
            well_dict[well_name] = {'ijk': cells_ijk, 'f': cells_f}
    return well_dict

def extract_well_schedule(simdata, root, grid, well_names):
    well_schedule = defaultdict(list)
    for sim in simdata:
        rd_sum = ResdataFile(root + '.' + sim)
        WI = WellInfo(grid, rd_sum)
        for well_name in well_names:
            if well_name in  WI.allWellNames():
                well_schedule[well_name].append(True)
            else:
                well_schedule[well_name].append(False)
    return well_schedule

def extract_dynamic_props(sim, root, props):
    prop_np = []
    rd = ResdataFile(root + '.' + sim)
    for prop in props:
        p_np = np.array(rd[prop])
        p_np =  np.reshape(p_np, (1, p_np.shape[1], 1))
        prop_np.append(p_np)
    props_np = np.concatenate(prop_np, axis=2)
    return props_np

def extract_static_props(root, props):
    rd = ResdataFile(root + '.INIT')
    prop_np = []
    for prop in props:
        p_np = np.array(rd[prop])
        p_np =  np.reshape(p_np, (1, p_np.shape[1], 1))
        prop_np.append(p_np)
    props_np = np.concatenate(prop_np, axis=2)
    return props_np

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
                index, value = line.split()
                data_dict[int(index)] = float(value)
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
    # Threshold in days, values of tof_dict in seconds
    threshold = threshold*24*60*60
    well_neighbors = defaultdict(dict)
    for well in tof_dict.keys():
        sorted_tof = sorted(tof_dict[well].items(), key=lambda item: item[1])
        filtered_tof = [index for index, value in sorted_tof if value < threshold]
        well_neighbors[well]['f'] = filtered_tof
    return well_neighbors

def main(member):
    ml_data_folder = 'En_ml_data'
    if not os.path.isdir(ml_data_folder):
        os.mkdir(ml_data_folder)

    folder = 'En_' + str(member) + os.sep
    filename = get_filename(folder)
    smrys, simdata, misc_ext = get_extensions(folder)
    root = folder + os.sep + filename
    well_mode_dict = {'INJ':['A5', 'A6'],
            'PROD': ['A1', 'A2', 'A3', 'A4']}
    well_names = well_mode_dict['INJ'] + well_mode_dict['PROD']

    # Extract timesteps
    rd_sum = Summary(root)
    # Summaries contain all dates expect the initial
    rpt = rd_sum.report_dates
    init_rpt = get_initial_date()
    rpt = [init_rpt] + rpt
    dts = [(rpt[i] - rpt[i-1]).days for i in range(1, len(rpt))]

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
    # Using arbitrary distance
    max_distance = 2
    well_local_domains = bfs_extend_neighborhood(well_dict, grid, max_distance=max_distance)
    # Using Time of Flight
    # if member == 0:
    #     tof_dict = compute_TOF(root, simdata[-1], ml_data_folder, well_mode_dict)
    #     plot_tof_hist(tof_dict, savepath=ml_data_folder + '/ToF')
    #     tof_well_local_domains = compute_TOF_local_domain(tof_dict, threshold=10000)
    #     print([(well,len(item['f'])) for well,item in tof_well_local_domains.items()])
    #     print([(well,len(item['f'])) for well,item in well_local_domains.items()])
    # Save the dictionary to a file
    with open(ml_data_folder + os.sep + f'well_dict_local_{max_distance}.pkl', 'wb') as f:
        pickle.dump(well_local_domains, f)
    # Compute well schedule for input features, we don't care about the last timestep
    well_schedules = extract_well_schedule(simdata[:-1], root, grid, well_names)

    # Static and Dynamic features extraction
    # Dynamic features
    dyn_props = ['PRESSURE', 'SGAS']
    assert len(well_schedules[well_names[0]]) == len(simdata) - 1
    well_dyn_props = defaultdict(lambda: defaultdict(list))

    dyn_props_np = extract_dynamic_props(simdata[0], root, dyn_props)
    for i in range(1, len(simdata)):
        for well_name in well_names:
            # Check if well is active during this step
            if well_schedules[well_name][i - 1]:
                # Get local property
                local_dyn_props_np = dyn_props_np[:, well_local_domains[well_name]['f'], :]
                well_dyn_props[well_name]['X'].append(local_dyn_props_np)
                well_dyn_props[well_name]['dts'].append(dts[i-1])

        # Full grid feature extraction
        dyn_props_np = extract_dynamic_props(simdata[i], root, dyn_props)

        for well_name in well_names:
            # Check if well was active during previous step
            if well_schedules[well_name][i-1]:
                # Get local property
                local_dyn_props_np = dyn_props_np[:, well_local_domains[well_name]['f'], :]
                well_dyn_props[well_name]['Y'].append(local_dyn_props_np)

    for well_name in well_names:
        well_dyn_props[well_name]['X'] = np.concatenate(well_dyn_props[well_name]['X'], axis=0)
        well_dyn_props[well_name]['Y'] = np.concatenate(well_dyn_props[well_name]['Y'], axis=0)
    
    # Static Features 
    stat_props = ['PERMX']
    stat_props_np = extract_static_props(root, stat_props)

    for well_name in well_names:
        well_folder = ml_data_folder + os.sep + well_name
        os.makedirs(well_folder, exist_ok=True)
        # Save to npz for ml routine
        np.savez(well_folder + os.sep + 'En_' + str(member), stat_props=stat_props_np[:, well_local_domains[well_name]['f'],:], 
                 dyn_props_X=well_dyn_props[well_name]['X'], dyn_props_Y=well_dyn_props[well_name]['X'], 
                 dts=well_dyn_props[well_name]['dts'])
