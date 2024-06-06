import matplotlib.pyplot as plt
import pickle 
import os
from collections import defaultdict
import numpy as np

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
    plt.clf()

def plot_tof_hist(tof_dict, savepath):
    for well, tofs in tof_dict.items():
        print(len(list(tofs)))
        plt.hist(tofs, bins=100, label=well)
        plt.xlabel('Time of Flight')
        plt.ylabel('Frequency')
        plt.title('Histogram of Time of Flight Values')
        plt.legend()
        plt.savefig(savepath + os.sep + well + '.png')
        plt.clf()
    return 