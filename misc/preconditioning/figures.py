import matplotlib.pyplot as plt

import pickle 
import os
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Union

import matplotlib.patches as mpatches

# Turn off oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

from misc.preconditioning.utils import perform_linear_regression

def plot_loss(en_hist_dict, figname, ml_model_folder, finetuning):
    """
    Plots the loss for each Keras history object.
    
    Parameters:
    hist_dict (dict): {well_name: Keras fitting history.history}
    """
    if not os.path.isfile(ml_model_folder + os.sep + figname + '.pickle'):
        with open(f'{ml_model_folder}/{figname}.pickle', 'wb') as handle:
            pickle.dump(en_hist_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if not finetuning:
            # Overwrite previous run file
            with open(f'{ml_model_folder}/{figname}.pickle', 'wb') as handle:
                pickle.dump(en_hist_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(f'{ml_model_folder}/{figname}.pickle', 'rb') as handle:
                past_en_hist_dict = pickle.load(handle)
            
            for en in en_hist_dict.keys():
                past_en_hist_dict[sorted(past_en_hist_dict.keys())[-1] + en] = en_hist_dict[en]
            with open(f'{ml_model_folder}/{figname}.pickle', 'wb') as handle:
                pickle.dump(past_en_hist_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            en_hist_dict = past_en_hist_dict

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    plt.figure(figsize=(12, 8))
    plot_dict = defaultdict(list)
    x_plot_dict = defaultdict(list)
    for en in sorted(en_hist_dict.keys()):
        for well_name, history in en_hist_dict[en].items():
            plot_dict[well_name] += history['loss']
            x_plot_dict[well_name].append(len(history['loss']))

    
    colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y', 'w', 'orange', 'purple', 'brown', 'pink', 'lime', 'teal', 'olive', 'navy', 'maroon']
    point_styles = ['X', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'o', 'd']

    for j, (well, x_list) in enumerate(x_plot_dict.items()):
        for i, x in enumerate(x_list):
            if i == 0:
                x_range = list(range(x))
                plt.plot(x_range, np.array(plot_dict[well])[x_range], linestyle='--',  color=colors[j], linewidth=2)
                plt.scatter(x_range, np.array(plot_dict[well])[x_range], color=colors[j], marker=point_styles[j], label=f"$\\mathbf{{{well}}}$", s=8)
                if j == 0:
                    plt.axvline(x=x_range[0] + 0.5, color='black', linestyle='--', label=r'\textbf{Ensemble} \textbf{ iteration}', linewidth=2)
                else:
                    plt.axvline(x=x_range[0] + 0.5, color='black', linestyle='--', linewidth=2)
            else:
                x_range = list(range(x_range[-1] + 1, x_range[-1] + x + 1))
                plt.plot(x_range, np.array(plot_dict[well])[x_range], linestyle='--', color=colors[j], linewidth=2)
                plt.scatter(x_range, np.array(plot_dict[well])[x_range], color=colors[j], marker=point_styles[j], s=8)
                plt.axvline(x=x_range[0] + 0.5, color='black', linestyle='--', linewidth=2)

    plt.title(r'\textbf{Evolution of Loss Plot through Ensemble iterations}', fontsize=16)
    plt.xlabel(r'\textbf{Epoch}', fontsize=14)
    plt.ylabel(r'\textbf{Relative } $\mathbf{L^2}$ \textbf{ Error}', fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig(f'{ml_model_folder}/{figname}.png', dpi=500)
    plt.close()
    return 



def quality_plots(well_name:str, figname:str, ml_model_folder:str, epsilon:float=0.3) -> Union[bool, None]:
    with open(f'{ml_model_folder}/{well_name}_predictions.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
        y_hat = np.array(data_dict['predictions'])
        y_true = np.array(data_dict['solutions'])
    
    # Data processing
    # props = ['PRESSURE', 'SWAT', 'SOIL', 'RS']
    props = [r'p_o', r's_w', r's_o', r'r_s']
    Nc = int(y_true.shape[1]/len(props))

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    fig, axs = plt.subplots(1, len(props), figsize=(6*len(props), 6))
    is_well_model_ready = True
    for i, prop in enumerate(props):
    
        prop_true = y_true[:, i*Nc:(i+1)*Nc]
        prop_pred = y_hat[:, i*Nc:(i+1)*Nc]

        prop_true_l2 = np.linalg.norm(prop_true, axis=1)
        prop_pred_l2 = np.linalg.norm(prop_pred, axis=1)
        slope, _, r2 = perform_linear_regression(prop_true_l2,  prop_pred_l2)

        # Parity Plots                             
        axs[i].scatter(prop_pred_l2, prop_true_l2, label=well_name, s=5)
        plot_range = [min(min(prop_pred_l2), min(prop_true_l2)), max(max(prop_pred_l2), max(prop_true_l2))]
        axs[i].plot(plot_range, plot_range, color='red', linestyle='--')
        axs[i].set_xlabel(f"$\\mathbf{{\\|\\hat{{{prop}}}\\|_2}}$", fontsize=28)
        axs[i].set_ylabel(f"$\\mathbf{{\\|{prop}\\|_2}}$", fontsize=28)
        axs[i].set_title(f"$\\mathbf{{{prop}}}$: $\\mathbf{{R^2 =}}${r2:.2f}, slope: $\\mathbf{{a={slope:.2f}}}$", fontsize=24)

        if not (1. - epsilon <= slope <= 1. + epsilon and 1. - epsilon <= r2):
            is_well_model_ready = False

    fig.tight_layout()
    plt.savefig(f'{ml_model_folder}/{figname}_{well_name}.png', dpi=500)
    plt.close()

    if is_well_model_ready:
        return well_name
    return 


def swim_quality_plots(kerswim, X_train, Y_train, well_name:str, figname:str, ml_model_folder:str) -> None:
    # Data processing
    # props = ['PRESSURE', 'SWAT', 'SOIL', 'RS']
    props = [r'p_o', r's_w', r's_o', r'r_s']
    Nc = int(Y_train.shape[1]/len(props))

    y_hat = kerswim.predict(X_train)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    fig, axs = plt.subplots(1, len(props), figsize=(6*len(props), 6))
    for i, prop in enumerate(props):
    
        prop_true = Y_train[:, i*Nc:(i+1)*Nc]
        prop_pred = y_hat[:, i*Nc:(i+1)*Nc]

        prop_true_l2 = np.linalg.norm(prop_true, axis=1)
        prop_pred_l2 = np.linalg.norm(prop_pred, axis=1)
        slope, _, r2 = perform_linear_regression(prop_true_l2,  prop_pred_l2)

        # Parity Plots                             
        axs[i].scatter(prop_pred_l2, prop_true_l2, label=well_name, s=5)
        plot_range = [min(min(prop_pred_l2), min(prop_true_l2)), max(max(prop_pred_l2), max(prop_true_l2))]
        axs[i].plot(plot_range, plot_range, color='red', linestyle='--')
        axs[i].set_xlabel(f"$\\mathbf{{\\|\\hat{{{prop}}}\\|_2}}$", fontsize=28)
        axs[i].set_ylabel(f"$\\mathbf{{\\|{prop}\\|_2}}$", fontsize=28)
        axs[i].set_title(f"$\\mathbf{{{prop}}}$: $\\mathbf{{R^2 =}}${r2:.2f}, slope: $\\mathbf{{a={slope:.2f}}}$", fontsize=24)


    fig.tight_layout()
    plt.savefig(f'{ml_model_folder}/swim_{figname}_{well_name}.png', dpi=500)
    plt.close()
    return 


def plot_solver_report(ml_data_folder:str, figname:str, en_i, newton:str='standard') -> None:
    NewtEn = []
    for en in range(en_i):
        reports_folder = ml_data_folder + os.sep + f"En_iter_{en}"+ os.sep +  f"{newton}_newton"
        solver_reports = sorted([f for f in os.listdir(reports_folder) if os.path.isfile(os.path.join(reports_folder, f)) and f.startswith('solver_report_') and f.endswith('.csv')])
        cum_NewtIt = []
        for report in solver_reports:
            solver_df = pd.read_csv(reports_folder + os.sep + report, sep='\t')
            cum_NewtIt.append(np.sum(np.array(solver_df['NewtIt'].values, dtype=np.int64)))
        NewtEn.append(cum_NewtIt)

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    fig, ax =  plt.subplots(1, 1, figsize=(8, 8))

    ax.boxplot(NewtEn, patch_artist=True)  

    ax.set_ylabel('Newton iterations')
    ax.set_ylabel('Ensemble iteration')
    fig.tight_layout()

    plt.savefig(f'{ml_data_folder}/{figname}.png')
    plt.close()


def plot_combined_solver_report(ml_data_folder:str, figname:str, en_i):

    NewtonEnCombined = {}
    

    for newton in ["hybrid", "standard"]:
        NewtEn = []
        for en in range(en_i):
            # max_newt, max_idx, report_idx = -1, None, None
            reports_folder = ml_data_folder + os.sep + f"En_iter_{en}"+ os.sep +  f"{newton}_newton"
            solver_reports = sorted([f for f in os.listdir(reports_folder) if os.path.isfile(os.path.join(reports_folder, f)) and f.startswith('solver_report_') and f.endswith('.csv')])
            cum_NewtIt = []
            for i, report in enumerate(solver_reports):
                solver_df = pd.read_csv(reports_folder + os.sep + report, sep='\t')
                cum_NewtIt.append(np.sum(np.array(solver_df['NewtIt'].values, dtype=np.int64)))
                # if np.sum(np.array(solver_df['NewtIt'].values)) > max_newt:
                #     max_newt = np.sum(np.array(solver_df['NewtIt'].values))
                #     max_idx = i
                #     report_idx = reports_folder + os.sep + report
            # print(f"Ensemble iter {en + 1}, Max Newton {max_newt} at index {max_idx} for method {newton}, report {report_idx}")
            NewtEn.append(cum_NewtIt)
        NewtonEnCombined[newton] = NewtEn
    
    data = []
    for i in range(len(NewtonEnCombined["standard"])):
        data.append(NewtonEnCombined["standard"][i])
        data.append(NewtonEnCombined["hybrid"][i])

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    fig, ax =  plt.subplots(1, 1, figsize=(10, 6))

    boxplot = ax.boxplot(data, patch_artist=True)  

    colors = ['lightblue', 'lightgreen']
    for i, patch in enumerate(boxplot['boxes']):
        patch.set_facecolor(colors[i % 2])

    # Create a legend for the color coding
    blue_patch = mpatches.Patch(color='lightblue', label=r'\textbf{Standard}')
    green_patch = mpatches.Patch(color='lightgreen', label=r'\textbf{Hybrid}')
    ax.legend(handles=[blue_patch, green_patch], title=r"\textbf{Newton's method}")

    iteration_labels = [f'Iter {i+1}' for i in range(en_i)]
    ax.set_xticks([i*2 + 1.5 for i in range(en_i)])  
    ax.set_xticklabels(iteration_labels)
    ax.set_title(r'\textbf{Standard VS Hybrid Cumulative Newton Iterations Across Iterations}', fontsize=14)
    ax.set_xlabel(r'\textbf{Ensemble Iteration}', fontsize=12)
    ax.set_ylabel(r'\textbf{Ensemble Cumulative Newton iteration}', fontsize=12)
    fig.tight_layout()
    plt.savefig(f'{ml_data_folder}/{figname}.png')
    plt.close()
    return 

def plot_well_models_history(well_models_history, well_order, save_path):
    # Create a new dictionary for rearranged data
    rearranged_data = {}

    # Iterate over each iteration in the original data
    for iteration, values in well_models_history.items():
        # Create a new list based on the correct parameter order
        rearranged_values = [None] * len(well_order)  # Initialize with None
        for idx, param in enumerate(well_order):
            if param in values:
                rearranged_values[idx] = param if values.count(param) > 0 else None
            else:
                rearranged_values[idx] = None
                
        rearranged_data[iteration] = rearranged_values

    # Define the complete set of parameters in the intended y-axis order
    data = rearranged_data

    # Convert data to DataFrame
    df = pd.DataFrame.from_dict(data, orient='index', columns=well_order)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw lines for active parameters
    for iteration, row in df.iterrows():
        for param_idx, activation in enumerate(row):
            if activation is not None:
                # Draw a horizontal line for active parameters
                ax.plot([iteration - 0.5, iteration + 0.5], [param_idx, param_idx], color='blue', linewidth=2, solid_capstyle='round')

    # Set axis limits
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_ylim(-0.5, len(well_order) - 0.5)

    # Set y-axis labels with all parameters, including inactive ones
    ax.set_yticks(range(len(well_order)))
    ax.set_yticklabels(well_order)

    # Set x-axis labels for iterations
    ax.set_xticks(range(len(df.index)))
    ax.set_xticklabels([f"Iteration {i}" for i in df.index])

    # Add vertical grid lines between iterations
    for i in range(1, len(df.index)):
        ax.axvline(i - 0.5, color='gray', linestyle='--', alpha=0.7)

    # Title and layout adjustments
    ax.set_title("Parameter Activation Summary Across Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Parameters")
    plt.tight_layout()

    plt.savefig(save_path)
    return 


def compute_schedule(start_date, opening_dates: dict):
    return {event: (event_date - start_date).days for event, event_date in opening_dates.items()}


def gather_report_data(ml_data_folder, en_i, well_schedule):
    # Gather all report data 
    all_reports = {"hybrid": [], "standard": []}
    NewtonEnCombined = {}
    NewtonEnCombined_330 = {}
    for newton in ["hybrid", "standard"]:
        NewtEn = []
        NewtEn_330 = []
        for en in range(en_i):
            reports_folder = ml_data_folder + os.sep + f"En_iter_{en}"+ os.sep +  f"{newton}_newton"
            solver_reports = sorted([f for f in os.listdir(reports_folder) if os.path.isfile(os.path.join(reports_folder, f)) and f.startswith('solver_report_') and f.endswith('.csv')])
            en_reports = []
            cum_NewtIt = []
            cum_NewtIt_330 = []
            for report in solver_reports:
                opening_report = {}
                solver_df = pd.read_csv(reports_folder + os.sep + report, sep='\t')
                cum_NewtIt.append(np.sum(np.array(solver_df['NewtIt'].values, dtype=np.int64)))
                # Get sum before 330 days, last well opening is A6 at 320
                cum_NewtIt_330 = solver_df.loc[solver_df['Time(day)'] < 330].sum()
                for well in well_schedule:
                    opening_row = solver_df[solver_df['Time(day)'] == well_schedule[well]]
                    opening_report[well] = opening_row[['Time(day)', 'TStep(day)', 'NewtIt']].to_dict(orient='records')
                en_reports.append(opening_report)
            NewtEn.append(cum_NewtIt)
            NewtEn_330.append(cum_NewtIt_330)
            all_reports[newton].append(en_reports)
        NewtonEnCombined[newton] = NewtEn
        NewtonEnCombined_330[newton] = NewtEn_330
    return all_reports, NewtonEnCombined, NewtonEnCombined_330


def find_suitable_en_well_model(well_models_history, opening_dates, en_i):
    suitable_en_well_model = []
    for iteration, values in well_models_history.items():
        valid_dates = {name: opening_dates[name] for name in values if name is not None and name in opening_dates}
        if not valid_dates:
            continue
        suitable_en_well_model.append((iteration, min(valid_dates, key=lambda name: valid_dates[name])))
    
    # If the same well is suitable at multiple ensemble, take the last ensemble iteration
    max_tuples = {}
    for iteration, well in suitable_en_well_model:
        # Iteration is from 0 to 9, shift it to 1 to 10, en_i however it 10
        iteration += 1
        if iteration < en_i:
            # If the second element is not in the dictionary, or if the current first is higher, update it
            if well not in max_tuples or iteration > max_tuples[well][0]:
                max_tuples[well] = (iteration, well)

    return list(max_tuples.values())


def plot_dt_vs_newt(all_reports, well_models_history, save_path, opening_dates, en_i):
    reports_to_plot = find_suitable_en_well_model(well_models_history, opening_dates, en_i)
    num_plots = len(reports_to_plot)

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    if num_plots == 1:
        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    elif num_plots == 2:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    elif num_plots == 3:
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    elif num_plots == 4:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten() if num_plots > 1 else [axs]

    # Get data for each of these reports
    colors = ["blue", 'orange']
    ms = ["X", 'o']
    s = [10, 5]

    for i, (iteration, well) in enumerate(reports_to_plot):
        for c, newton in enumerate(["hybrid", "standard"]):
            dt_cnv = []
            newtit = []
            # Plot Newton performances at the next iteration as it is when the model is actually used
            std_report = all_reports[newton][iteration]
            for report in std_report:
                dt_cnv.append(report[well][-1]['TStep(day)'])
                newtit.append((len(report[well]) - 1) * 21 + report[well][-1]['NewtIt'])

            axs[i].scatter(dt_cnv, newtit, c=colors[c], s=s[c], label=newton, marker=ms[c])

        axs[i].set_xlabel(r'\textbf{Timestep in days}')
        axs[i].set_ylabel(r'\textbf{Total number of Newton iterations')
        axs[i].set_title(r'$\textbf{Well ' + str(well) + r' at ensemble iteration ' + str(iteration + 1) + '}$')
        axs[i].grid(True)
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(save_path + os.sep + "std_dt_vs_newt.png", dpi=500)
    return 

def plot_best_and_worst_simulation(ml_data_folder, NewtonEnCombined, well_schedule, save_path, figname):
    standard = NewtonEnCombined['standard']
    hybrid = NewtonEnCombined['hybrid']
    
    # Initialize min_delta and max_delta with None
    min_delta = None
    max_delta = None
    min_i, min_j = None, None
    max_i, max_j = None, None
    # Iterate through the outer lists and then through each element
    for i, (std_en, hyb_en) in enumerate(zip(standard, hybrid)):
        for j, (std_sim, hyb_sim) in enumerate(zip(std_en, hyb_en)):
            # Calculate the absolute difference
            delta = std_sim - hyb_sim
            # Update min_delta and max_delta
            if min_delta is None or delta < min_delta:
                min_delta = delta
                min_i, min_j = i, j
            if max_delta is None or delta > max_delta:
                max_delta = delta
                max_i, max_j = i, j

    colors = ["blue", 'orange']
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
    best_count = []
    for i, newton in enumerate(["hybrid", "standard"]):
        reports_folder = ml_data_folder + os.sep + f"En_iter_{max_i}"+ os.sep +  f"{newton}_newton"
        solver_reports = sorted([f for f in os.listdir(reports_folder) if os.path.isfile(os.path.join(reports_folder, f)) and f.startswith('solver_report_') and f.endswith('.csv')])
        best_report_df = pd.read_csv(reports_folder + os.sep + solver_reports[max_j], sep='\t')
        best_NewtIt = best_report_df['NewtIt'].values
        best_dt = best_report_df['Time(day)'].values
        axs[0].scatter(best_dt, np.cumsum(best_NewtIt), c=colors[i], s=10, label=newton)
        best_count.append(np.sum(best_NewtIt))
        
    worst_count = []
    for i, newton in enumerate(["hybrid", "standard"]):
        reports_folder = ml_data_folder + os.sep + f"En_iter_{min_i}"+ os.sep +  f"{newton}_newton"
        solver_reports = sorted([f for f in os.listdir(reports_folder) if os.path.isfile(os.path.join(reports_folder, f)) and f.startswith('solver_report_') and f.endswith('.csv')])
        worst_report_df = pd.read_csv(reports_folder + os.sep + solver_reports[min_j], sep='\t')
        worst_NewtIt = worst_report_df['NewtIt'].values
        worst_dt = worst_report_df['Time(day)'].values
        axs[1].scatter(worst_dt, np.cumsum(worst_NewtIt), c=colors[i], s=10, label=newton)
        worst_count.append(np.sum(worst_NewtIt))

    axs[0].set_xlabel(r'\textbf{Time (days)}')
    axs[0].set_ylabel(r'\textbf{Cumulative Newton iteration}')
    axs[0].set_title(r'\textbf{Best case scenario for Newton Hybrid: final gain of }' + str(best_count[1] - best_count[0]) + r'\textbf{ Newton iterations}')

    axs[1].set_xlabel(r'\textbf{Time (days)}')
    axs[1].set_ylabel(r'\textbf{Cumulative Newton iteration}')
    axs[1].set_title(r'\textbf{Worst case scenario for Newton Hybrid: final loss of }' + str(abs(worst_count[1] - worst_count[0])) + r'\textbf{ Newton iterations}') 

    linestyles = ['-', '--', ':', '-.', (0, (5, 2)), (0, (3, 5, 1, 5))]
    for i, (well, opening) in enumerate(well_schedule.items()):
        axs[0].axvline(opening, linestyle=linestyles[i], label=well, color='black')
        axs[1].axvline(opening, linestyle=linestyles[i], label=well, color='black')

    axs[0].legend()
    axs[1].legend()
    fig.tight_layout()
    plt.savefig(save_path + os.sep + figname)
    return 

def gather_full_reports(en_i, opening_dates, start_date):
    well_schedule = compute_schedule(start_date, opening_dates)
    all_reports = {"hybrid": [], "standard": []}
    for newton in ["hybrid", "standard"]:
        for en in range(en_i):
            reports_folder = 'En_ml_data' + os.sep + f"En_iter_{en}"+ os.sep +  f"{newton}_newton"
            solver_reports = sorted([f for f in os.listdir(reports_folder) if os.path.isfile(os.path.join(reports_folder, f)) and f.startswith('solver_report_') and f.endswith('.csv')])
            en_reports = []
            for report in solver_reports:
                whole_report = {}
                solver_df = pd.read_csv(reports_folder + os.sep + report, sep='\t')
                for well in well_schedule:
                    opening_row = solver_df[solver_df['Time(day)'] == well_schedule[well]]
                    opening_idx = opening_row.index[0]
                    # Get next report time:
                    report_time = opening_row['TStep(day)'].values[0]
                    # Get all data from that report
                    end_report_idx = solver_df[solver_df['Time(day)'] == well_schedule[well] + report_time].index[0]
                    report_data = solver_df.iloc[opening_idx:end_report_idx]
                    whole_report[well] = report_data[['Time(day)', 'TStep(day)', 'NewtIt']].to_dict(orient='records')
                en_reports.append(whole_report)
            all_reports[newton].append(en_reports)
    return all_reports

def plot_full_reports(full_reports, en_i, well_models_history):
    newtCumReport = defaultdict(list)
    for newton in ["hybrid", "standard"]:
        for en in range(en_i):
            en_reports = full_reports[newton][en]
            cumNewt = defaultdict(list)
            for report in en_reports:
                for well, timesteps in report.items():
                    countNewt = 0
                    for timestep in timesteps:
                        countNewt += timestep['NewtIt']
                    cumNewt[well].append(countNewt)
            newtCumReport[newton].append(cumNewt)

    fig, axs = plt.subplots(en_i - 1, 6, figsize=(20, 10))

    for i in range(1, en_i):
        for j, well in enumerate(['A1', 'A2', 'A3', 'A4', 'A5', 'A6']):
            if well in well_models_history[i-1]:
                axs[i-1][j].boxplot([newtCumReport['standard'][i][well], newtCumReport['hybrid'][i][well]],  
                                tick_labels=['standard', 'hybrid'])
            
    plt.savefig('test_plot.png')
    return 



def plot_well_opening_newton(ml_data_folder, en_i, well_models_history, start_date, opening_dates, save_path):
    well_schedule = compute_schedule(start_date=start_date, opening_dates=opening_dates)
    all_reports, NewtonEnCombined, NewtonEnCombined_330 = gather_report_data(ml_data_folder=ml_data_folder, en_i=en_i, well_schedule=well_schedule)
    plot_dt_vs_newt(all_reports=all_reports, well_models_history=well_models_history, 
                    save_path=save_path, opening_dates=opening_dates, en_i=en_i)
    
    plot_best_and_worst_simulation(ml_data_folder=ml_data_folder, NewtonEnCombined=NewtonEnCombined, 
                                   well_schedule=well_schedule, save_path=save_path, figname='best_std_vs_hybrid')
    # Same plot but with cutoff at 330 days, 10 days after the last well opening 
    plot_best_and_worst_simulation(ml_data_folder=ml_data_folder, NewtonEnCombined=NewtonEnCombined_330, 
                                   well_schedule=well_schedule, save_path=save_path, figname='best_std_vs_hybrid_330')
    return