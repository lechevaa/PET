import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

def gather_report_data(ml_data_folder, en_i, hybrid=False):
    # Gather report data 
    mode = "standard_newton"
    if hybrid: 
        mode = "hybrid_newton"

    reports_folder = ml_data_folder + os.sep + f"En_iter_{en_i}"+ os.sep +  mode
    solver_reports = [f for f in os.listdir(reports_folder) if os.path.isfile(os.path.join(reports_folder, f)) and f.startswith('solver_report_') and f.endswith('.csv')]
    solver_reports = sorted(solver_reports, key=lambda x: int(x.replace("solver_report_", "").replace(".csv", "")))
    time, newt = [], []
    for report in solver_reports:
        solver_df = pd.read_csv(reports_folder + os.sep + report, sep='\t')
        time.append(solver_df['Time(day)'].values)
        newt.append(solver_df['NewtIt'].values)
    
    return np.concatenate(time), np.concatenate(newt)


def nonlinear_detection(ml_data_folder, en_i, percent, figname, hybrid, y_lim=None, fixed_xticks=None, return_params=False, field='drogon'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    time, newt = gather_report_data(ml_data_folder, en_i, hybrid)

    df = pd.DataFrame({"Time": time, "Value": newt})

    threshold = 21
    above_threshold_counts = df.groupby("Time")["Value"].apply(lambda x: np.sum(x >= threshold))
    # print(above_threshold_counts)
    total_counts = df.groupby("Time")["Value"].count().clip(lower=100)
    normalized_counts = above_threshold_counts / total_counts * 100
    colors = ["blue" if p == 0 else "red" for p in normalized_counts]
    sizes = [50 if p == 0 else 100 for p in normalized_counts]
    plt.rcParams['text.usetex'] = True

    if field == 'norne':
        plt.figure(figsize=(16, 9))
        plt.axvline(x=1321, color='black', linestyle='--', zorder=1)
        plt.axvline(x=414, color='black', linestyle='--', zorder=1)
        plt.xticks(ticks=[1321, 414], labels=['C-1H', 'C-4H'], rotation=45, fontsize=18)
    elif field == 'drogon':
        plt.figure(figsize=(10, 8))
        plt.axvline(x=68, color='black', linestyle='--', zorder=1)
        plt.axvline(x=277, color='black', linestyle='--', zorder=1)
        plt.xticks(ticks=[68, 277], labels=['A2', 'A4'], rotation=45, fontsize=18)
    else:
        plt.figure(figsize=(16, 6))

    plt.scatter(
        normalized_counts.index, 
        normalized_counts,
        c=colors, 
        edgecolors="none",
        s=sizes,
        zorder=3
    )

    if fixed_xticks is not None:
        xticks_to_use = fixed_xticks
    else:
        
        xticks_to_use = normalized_counts.index[normalized_counts > percent]
        xticks_to_use = xticks_to_use[xticks_to_use == xticks_to_use.astype(int)]
    plt.xticks(xticks_to_use, rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    # print(normalized_counts)
    if y_lim is not None:
        plt.ylim(0, y_lim)

    plt.xlabel(r"\textbf{Events through time}", fontsize=20)
    plt.ylabel(r"\textbf{Percentage of failed Newton convergence}", fontsize=20)
    plt.grid()
    plt.savefig(f'{figname}.png', dpi=500)
    plt.close()

    if return_params:
        return normalized_counts.max(), list(xticks_to_use)



def hm_non_linear_detection(ml_data_folder, ne, percent, save_dir, field='drogon'):
    os.makedirs(save_dir, exist_ok=True)
    ymax, xticks = None, None

    for en_i in range(ne):
        figname = f'_nonlinear_detection_en_{en_i}'

        if en_i == 0:
            raw_max, xticks = nonlinear_detection(
                ml_data_folder,
                en_i=en_i,
                percent=percent,
                figname=save_dir + os.sep + 'standard' + figname,
                hybrid=False,
                return_params=True,
                field=field
            )
            buffer = 1  # Or e.g., 0.05 * raw_max for 5% padding
            # Pad here and fix it for all
            if field =='norne':
                buffer = 3
            
            ymax = raw_max + buffer

            # Now plot again using fixed ylim
            nonlinear_detection(
                ml_data_folder,
                en_i=en_i,
                percent=percent,
                figname=save_dir + os.sep + 'standard' + figname,
                hybrid=False,
                y_lim=ymax,
                fixed_xticks=xticks,
                field=field
            )
            nonlinear_detection(
                ml_data_folder,
                en_i=en_i,
                percent=percent,
                figname=save_dir + os.sep + 'hybrid' + figname,
                hybrid=True,
                y_lim=ymax,
                fixed_xticks=xticks,
                field=field
            )
        else:
            nonlinear_detection(
                ml_data_folder,
                en_i=en_i,
                percent=percent,
                figname=save_dir + os.sep + 'standard' + figname,
                hybrid=False,
                y_lim=ymax,
                fixed_xticks=xticks,
                field=field
            )
            nonlinear_detection(
                ml_data_folder,
                en_i=en_i,
                percent=percent,
                figname=save_dir + os.sep + 'hybrid' + figname,
                hybrid=True,
                y_lim=ymax,
                fixed_xticks=xticks,
                field=field
            )
    return 


def plot_logs(log_folder):
    df_a = pd.read_csv(log_folder + os.sep + 'Logs_save' + os.sep + 'timerLog.csv')
    df_b = pd.read_csv(log_folder + os.sep + 'Logs' + os.sep + 'timerLog.csv')

    import matplotlib.ticker as ticker
    import re

    plt.rcParams.update({
        'font.family': 'serif',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    def extract_ensemble(section):
        match = re.search(r'En[ _](\d+)', section)
        return int(match.group(1)) if match else None

    def correct_ensemble_numbers(df):
        new_sections = []
        seen = {}
        ensemble_offset = 0

        for section in df['Section']:
            match = re.match(r'(En[ _])(\d+)(.*)', section)
            if match:
                prefix, num, suffix = match.groups()
                num = int(num)
                true_num = num + ensemble_offset
                key = f"{prefix}{true_num}"

                while key in seen:
                    ensemble_offset += 1
                    true_num = num + ensemble_offset
                    key = f"{prefix}{true_num}"

                seen[key] = True
                new_section = f"{prefix}{true_num}{suffix}"
            else:
                new_section = section

            new_sections.append(new_section)

        df['Section'] = new_sections
        return df

    # --- Load both files ---
    df_a = correct_ensemble_numbers(df_a)

    # Extract ensemble numbers
    df_a['Ensemble'] = df_a['Section'].apply(extract_ensemble)
    df_b['Ensemble'] = df_b['Section'].apply(extract_ensemble)

    # --- Process CPU and ML timings ---
    cpu_a = df_a.dropna(subset=['Ensemble']).groupby('Ensemble')['Time (s)'].sum().rename('CPU_A')
    cpu_b = df_b[df_b['Section'].str.contains('simulation', case=False)].groupby('Ensemble')['Time (s)'].sum().rename('CPU_B')
    ml_b = df_b[df_b['Section'].str.contains('ML training', case=False)].groupby('Ensemble')['Time (s)'].sum().rename('ML_Training_B')

    # Merge and calculate totals
    df = pd.concat([cpu_a, cpu_b, ml_b], axis=1).fillna(0)
    df['Total_B'] = df['CPU_B'] + df['ML_Training_B']
    df['Speedup (%)'] = (1 - df['Total_B'] / df['CPU_A']) * 100

    # --- Total speed-up across all ensembles ---
    total_cpu_a = df['CPU_A'].sum()
    total_b = df['Total_B'].sum()
    overall_speedup = (1 - total_b / total_cpu_a) * 100

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ensembles = df.index.astype(int).sort_values()
    bar_width = 0.35

    # Bars
    ax.bar(ensembles - bar_width/2, df.loc[ensembles, 'CPU_A'], width=bar_width, label='Standard History matching \nsimulation time', color='lightblue', edgecolor='black')
    ax.bar(ensembles + bar_width/2, df.loc[ensembles, 'CPU_B'], width=bar_width, label='Hybrid Newton + History matching\nsimulation time', color='lightgreen', edgecolor='black')
    ax.bar(ensembles + bar_width/2, df.loc[ensembles, 'ML_Training_B'], width=bar_width,
        bottom=df.loc[ensembles, 'CPU_B'], label='Dataset Gathering + Machine Learning training', color='orange', edgecolor='black')

    # Speedup annotations
    for en in ensembles:
        spd = df.loc[en, 'Speedup (%)']
        y = max(df.loc[en, ['CPU_A', 'Total_B']])
        ax.text(en, y + 40, rf"{spd:.1f}\%", ha='center', va='bottom', fontsize=12,
                color='green' if spd > 0 else 'red')

    iteration_labels = [f'Iter {i+1}' for i in ensembles.values] 
    ax.set_xticklabels(iteration_labels)

    ax.set_ylim(0, max(df.loc[en, ['CPU_A', 'Total_B']] + 250))
    # Format
    ax.set_xlabel(r'\textbf{Ensemble Iteration}')
    ax.set_ylabel(r'\textbf{Time (seconds)}')
    # ax.set_title('Per-Ensemble Execution Time Comparison with Speed-up Annotations')
    ax.set_xticks(ensembles)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)

    # Total speed-up annotation
    caption = f"Overall speed-up: {overall_speedup:.1f}" + r"\% " + f"(Total Standard: {total_cpu_a:.0f}s, Total Hybrid: {total_b:.0f}s)"
    plt.figtext(0.99, 0.01, caption, ha='right', va='bottom', fontsize=12, style='italic')


    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.08), ncol=3, frameon=False)

    plt.tight_layout()

    plt.savefig('log_plot', dpi=500)

    return


def compute_total_newton(ml_data_folder, ne):
    total_newt_std = 0
    total_newt_hyb = 0
    for en_i in range(ne):
        _, newt_std = gather_report_data(ml_data_folder, en_i, hybrid=False)
        _, newt_hyb = gather_report_data(ml_data_folder, en_i, hybrid=True)
        total_newt_hyb += newt_hyb.sum()
        total_newt_std += newt_std.sum()

    return  total_newt_std, total_newt_hyb