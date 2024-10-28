import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pickle 
import os
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Union

# Turn off oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from misc.preconditioning.depreciated.utils import parse_function_kerasify, perform_linear_regression

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

    plt.figure(figsize=(12, 8))
    plot_dict = defaultdict(list)
    x_plot_dict = defaultdict(list)
    for en in sorted(en_hist_dict.keys()):
        for well_name, history in en_hist_dict[en].items():
            plot_dict[well_name] += history['loss']
            x_plot_dict[well_name].append(len(history['loss']))

    
    colors = ['r', 'g', 'b', 'c', 'm', 'k', 'y', 'w', 'orange', 'purple', 'brown', 'pink', 'lime', 'teal', 'olive', 'navy', 'maroon']
    point_styles = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd']

    for j, (well, x_list) in enumerate(x_plot_dict.items()):
        for i, x in enumerate(x_list):
            if i == 0:
                x_range = list(range(x))
                plt.plot(x_range, np.array(plot_dict[well])[x_range], linestyle='--',  color=colors[j])
                plt.scatter(x_range, np.array(plot_dict[well])[x_range], color=colors[j], marker=point_styles[j], label=f'{well}')
            else:
                x_range = list(range(x_range[-1] + 1, x_range[-1] + x + 1))
                plt.plot(x_range, np.array(plot_dict[well])[x_range], linestyle='--', color=colors[j])
                plt.scatter(x_range, np.array(plot_dict[well])[x_range], color=colors[j], marker=point_styles[j])
            plt.axvline(x=x_range[-1] + 0.5, color='black', linestyle='--')

    plt.title('Loss Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{ml_model_folder}/{figname}.png')
    plt.close()

def plot_tof_hist(tof_dict, savepath:str) -> None:
    for well, tofs in tof_dict.items():
        plt.hist(tofs, bins=100, label=well)
        plt.xlabel('Time of Flight')
        plt.ylabel('Frequency')
        plt.title('Histogram of Time of Flight Values')
        plt.legend()
        plt.savefig(savepath + os.sep + well + '.png')
        plt.close()
    return 


def quality_plots(well_name:str, figname:str, ml_model_folder:str,  model:str, epsilon:float=0.3) -> Union[bool, None]:
    with open(f'{ml_model_folder}/{well_name}_predictions.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)
        y_hat = np.array(data_dict['predictions'])
        y_true = np.array(data_dict['solutions'])
    
    # Data processing
    props = ['PRESSURE', 'SWAT', 'SGAS', 'RV', 'RS']
    Nc = int(y_true.shape[1]/len(props))
    fig, axs = plt.subplots(3, len(props), figsize=(6*len(props), 6*3))
    is_well_model_ready = True
    for i, prop in enumerate(props):
        if model == 'DeepONet':
            prop_true = y_true[:, :, i:i+1]
            prop_pred = y_hat[:, :, i:i+1]
        else:
            prop_true = y_true[:, i*Nc:(i+1)*Nc]
            prop_pred = y_hat[:, i*Nc:(i+1)*Nc]

        prop_true_l2 = np.linalg.norm(prop_true, axis=1)
        prop_pred_l2 = np.linalg.norm(prop_pred, axis=1)
        slope, _, r2 = perform_linear_regression(prop_true_l2,  prop_pred_l2)

        # Parity Plots                             
        axs[0][i].scatter(prop_pred_l2, prop_true_l2, label=well_name, s=5)
        plot_range = [min(min(prop_pred_l2), min(prop_true_l2)), max(max(prop_pred_l2), max(prop_true_l2))]
        axs[0][i].plot(plot_range, plot_range, color='red', linestyle='--')
        axs[0][i].set_xlabel(f'Predicted {prop}')
        axs[0][i].set_ylabel(f'True {prop}')
        axs[0][i].set_title(f"{well_name} {prop}: r2: {r2:.2f} slope: {slope:.2f}")

        # Residual plots
        residuals = prop_true - prop_pred
        norm_residuals = np.linalg.norm(residuals, axis=1)
        axs[1][i].scatter(np.arange(len(norm_residuals)), norm_residuals, label=well_name, s=5)

        # Residual distribution
        flat_residuals = residuals.flatten()
        axs[2][i].hist(flat_residuals, bins=10, density=True, color='green', alpha=0.7)

        if not (1. - epsilon <= slope <= 1. + epsilon and 1. - epsilon <= r2):
            is_well_model_ready = False
    fig.tight_layout()
    plt.savefig(f'{ml_model_folder}/{figname}_{well_name}.png')
    plt.close()

    if is_well_model_ready:
        return well_name
    return 

def plot_solver_report(ml_data_folder:str, figname:str) -> None:
    solver_reports = sorted([f for f in os.listdir(ml_data_folder) if os.path.isfile(os.path.join(ml_data_folder, f)) and f.startswith('solver_report_') and f.endswith('.csv')])
    cum_NewtIt = []
    NewtHist = []
    x_lists = []
    dt_list = []
    NewtRaw = defaultdict(list)
    for i, report in enumerate(solver_reports):
        solver_df = pd.read_csv(ml_data_folder + os.sep + report, sep='\t')
        cum_NewtIt.append(np.cumsum(np.array(solver_df['NewtIt'].values, dtype=np.float64)))
        x_lists.append(np.array(solver_df['Time(day)'].values, dtype=np.float64))
        dt_list += list(np.array(solver_df['TStep(day)'].values))
        NewtHist+= list(solver_df['NewtIt'].values)
        NewtRaw[i] = list(solver_df['NewtIt'].values)
    
    dict_raw = {'NewtIt': NewtRaw}
    if not os.path.isfile(f'{ml_data_folder}/{figname}_raw.pkl'):
        with open(f'{ml_data_folder}/{figname}_raw.pkl', 'wb') as f:
            pickle.dump({0 : dict_raw}, f)
    else:
        with open(f'{ml_data_folder}/{figname}_raw.pkl', 'rb') as f:
            dictNewtRaw = pickle.load(f)
            
        num_iter = len(NewtRaw.keys())
        dictNewtRaw[num_iter] = dict_raw
        
        with open(f'{ml_data_folder}/{figname}_raw.pkl', 'wb') as f:
            pickle.dump(dictNewtRaw, f)

    # 911 is number of days for drogon
    common_x = np.linspace(0, 911, 100)
    interpolated_y_lists = [np.interp(common_x, x, y) for x, y in zip(x_lists, cum_NewtIt)]
    mean_cumulative_y = np.mean(interpolated_y_lists, axis=0)

    if os.path.isfile(f'{ml_data_folder}/{figname}.pkl'):
        with open(f'{ml_data_folder}/{figname}.pkl', 'rb') as f:
            fig = pickle.load(f)
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
            ax3 = fig.axes[2]
    else:
        fig, (ax1, ax2, ax3) =  plt.subplots(1, 3, figsize=(20, 8))

    existing_curves = len(ax1.get_lines()) + 1
    
    cmap = plt.get_cmap('tab10')
    color = cmap(existing_curves % cmap.N)

    ax1.plot(common_x, mean_cumulative_y, label=f'Ensemble {existing_curves}: {mean_cumulative_y[-1]:.2f}', color=color)
    ax1.set_xlabel('Time (Day)')
    ax1.set_ylabel('Mean Cumulative Newton iterations')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(which='both', linestyle='--', linewidth=0.5)

    ax2.hist(NewtHist, bins=21, alpha=0.4, color=color)

    ax3.scatter(dt_list, NewtHist)
    ax3.set_xlabel('TimeStep (Day)')
    ax3.set_ylabel('Newton Iterations')
    ax3.axhline(y=3, color='r', linestyle='--', label='Newton Iterations = 3')

    fig.tight_layout()

    with open(f'{ml_data_folder}/{figname}.pkl', 'wb') as f:
        pickle.dump(fig, f)

    plt.savefig(f'{ml_data_folder}/{figname}.png')
    plt.close()
    