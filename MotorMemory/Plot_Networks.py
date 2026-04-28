import os
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

exp_names = ['center_out','mov_diff_loc','pro_loc','vis_loc']
phases = ['NF1','FF1','NF2']
measures = ['endpoint', 'lateral']
num_nets = 12


training_sets = {
    "set1": "/Users/pounemirzazadeh/Motornet/MultiNet/Modular_version/new_test",
}


purtb_group = {
    "CW":  {"suffix": "CW", "invert": False, "cmap": plt.cm.Greens},
    "CCW": {"suffix": "CCW", "invert": True, "cmap": plt.cm.Reds}}


def summarize_deviations(base_path, suffix, invert_vals):
    # Loads deviations and returns a nested summary dictionary.
    summary = {exp: {p: {} for p in phases} for exp in exp_names}

    for exp in exp_names:
        for phase in phases:
            for measure in measures:
                collected = defaultdict(list)
                for net_id in range(num_nets):
                    # Load data
                    path = os.path.join(base_path, f"task_{net_id}", f"deviations_{exp}_{suffix}")
                    dev_data = th.load(path)

                    batch_dict = dev_data[phase]
                    for batch_id, value in batch_dict.items():
                        collected[batch_id].append(th.tensor(value[measure]))

                for batch_id, task_values in collected.items():
                    stacked = th.stack(task_values, dim=0)
                    if batch_id not in summary[exp][phase]:
                        summary[exp][phase][batch_id] = {}
                    summary[exp][phase][batch_id][measure] = {
                        "concat": stacked,
                        "mean": stacked.mean(dim=0),
                        "std": stacked.std(dim=0),
                    }
    return summary

all_data = {s_name: {d_name: summarize_deviations(path, param["suffix"], param["invert"])
               for d_name, param in purtb_group.items()}
               for s_name, path in training_sets.items()}


# Plot deviations for each phase and each experiments across networks

total_curves = num_nets * len(training_sets)
colors_CW = plt.cm.Greens(np.linspace(0.4, 0.9, total_curves))
colors_CCW = plt.cm.Reds(np.linspace(0.4, 0.9, total_curves))

mean_summary = {exp: {p: [] for p in phases} for exp in exp_names}
std_summary = {exp: {p: [] for p in phases} for exp in exp_names}
batch_summary = {exp: {p: [] for p in phases} for exp in exp_names}

for exp in exp_names:
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for i, phase in enumerate(phases):
        ax = axs[i]
        data_pool = defaultdict(list)
        for s_idx, set_name in enumerate(training_sets.keys()):
            for d_name, d_param in purtb_group.items():
                curr_sum = all_data[set_name][d_name][exp][phase]
                batch_ids = sorted(curr_sum.keys(), key=lambda x: int(x))
                current_cmap = colors_CW if d_name == "CW" else colors_CCW

                for t_id in range(num_nets):
                    color_index = t_id + (s_idx * num_nets)
                    c = current_cmap[color_index]
                    vals = [curr_sum[b]['lateral']['concat'][t_id].item() for b in batch_ids]
                    ax.plot(batch_ids, vals, color=c, alpha=0.7, linewidth=1)

                for b in batch_ids:
                    data_pool[b].append(curr_sum[b]['lateral']['concat'])

        means, stds = [], []
        sorted_batches = sorted(data_pool.keys(), key=lambda x: int(x))

        for b in sorted_batches:
            all_points_in_batch = th.cat(data_pool[b])

            means.append(all_points_in_batch.mean().item())
            stds.append(all_points_in_batch.std().item())

        mean_summary[exp][phase] = means
        std_summary[exp][phase] = stds
        batch_summary[exp][phase] = sorted_batches

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(sorted_batches, means, color='black', linewidth=4, label="Mean" if i == 0 else "")
        ax.fill_between(sorted_batches, means - stds, means + stds,
                        color='gray', alpha=0.15, label="STD" if i == 0 else "")

        ax.set_title(f"Phase: {phase}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.07, 0.07)
        if i == 0: ax.set_ylabel("Lateral Deviation")

    plt.suptitle(exp)
    plt.tight_layout()
    plt.show()


# Plot summary at the end of FF1 and beginning of NF2 for each experiment across networks.

plot_table = {
    'center_out': {'num': '2', 'color': 'blue'},
    'vis_loc': {'num': '7', 'color': 'orange'},
    'pro_loc': {'num': '8', 'color': 'green'},
    'mov_diff_loc': {'num': '3', 'color': 'red'}
}


fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)  # 3 subplots, same y-axis

for i, phase in enumerate(phases):
    ax = axs[i]

    for j, exp in enumerate(exp_names):
        # get mean endpoint deviation for this experiment & phase
        mean_values = mean_summary[exp][phase]
        batch_values = batch_summary[exp][phase]

        ax.plot(batch_values, mean_values, label=exp, color=plot_table[exp]['color'])

    ax.set_title(f"Phase: {phase}")
    ax.set_xlabel("Batch")
    if i == 0:
        ax.set_ylabel("Lateral Deviation")
    ax.grid(True)
    ax.set_ylim(-0.07, 0.07)
    ax.legend()

plt.tight_layout()
plt.show()





plot_phases = [('FF1', -1), ('NF2', 0)]  #(Phase name, batch index)

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for i, (phase_name, b_idx) in enumerate(plot_phases):
    ax = axs[i]
    for j, exp in enumerate(plot_table.keys()):
        all_points = []
        for set_name in training_sets:
            for dir_name in purtb_group:
                summary = all_data[set_name][dir_name][exp][phase_name]
                b_ids = sorted(summary.keys(), key=lambda x: int(x))
                target_b = b_ids[b_idx]
                all_points.append(summary[target_b]['lateral']['concat'])

        combined = th.cat(all_points)
        m, s = combined.mean().item(), combined.std().item()/ np.sqrt(combined.numel())

        c = plot_table[exp]['color']
        ax.scatter(j, m, color=c, s=50, label=exp if i == 0 else "")
        ax.errorbar(j, m, yerr=s, color=c, capsize=6, lw=2)

    ax.set_xticks(range(len(plot_table)))
    ax.set_xticklabels([v['num'] for v in plot_table.values()])
    ax.set_title(f"{phase_name}")
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(-0.05, 0.05)

axs[0].legend(loc='upper left')
plt.tight_layout()
plt.show()
