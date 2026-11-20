import gc
import motornet as mn
import matplotlib.pyplot as plt
from Plot_utils import *
from matplotlib.lines import Line2D
from Plots import plot_subspace_batch, plot_distance_matrix


plotor = mn.plotor.plot_pos_over_time

exp_names = ['center_out','mov_diff_loc','vis_loc','pro_loc']
phases = ['NF1','FF1','NF2']
plot_measure = 'lateral_all'  # 'lateral_all' or 'force_comp_all' (channel_test_mask)
measures = [plot_measure]
num_nets = 20


training_sets = {
        "set1": "/Users/pounemirzazadeh/Motornet/MultiNet/Force_version/test_no_shift_gu"
        if os.path.isdir("/Users/pounemirzazadeh")
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "channel_test"),
}
saveLoc = training_sets["set1"]


purtb_group = {
    "CW":  {"suffix": "CW", "cmap": plt.cm.Greens},
    "CCW": {"suffix": "CCW", "cmap": plt.cm.Reds}}


plot_table = {
    'center_out': {'num': '2', 'color': 'blue', 'label':'Exp 1'},
    'vis_loc': {'num': '7', 'color': 'orange', 'label':'Exp 3'},
    'pro_loc': {'num': '8', 'color': 'green', 'label':'Exp 4'},
    'mov_diff_loc': {'num': '3', 'color': 'red', 'label':'Exp 2'}

}



# Load and concatenate loss and deviation data

is_force_comp = (plot_measure == 'force_comp_all')
dev_suffix = '_channel' if is_force_comp else ''
dev_ylabel = 'Force Compensation [%]' if is_force_comp else 'Lateral Deviation'

all_data = {s_name: {d_name: summarize_deviations(path, param["suffix"] + dev_suffix, phases = phases, exp_names = exp_names, num_nets = num_nets, measures = measures)
               for d_name, param in purtb_group.items()}
               for s_name, path in training_sets.items()}

all_loss_dict = {s_name: summerize_loss(path, phases = phases, exp_names = exp_names, num_nets = num_nets)
               for s_name, path in training_sets.items()}

all_trajectory = {s_name: {d_name: summarize_trajectory(path, param["suffix"], phases = phases, exp_names = exp_names, num_nets = num_nets)
               for d_name, param in purtb_group.items()}
               for s_name, path in training_sets.items()}


#----------------------------------------------- Loss and Deviations----------------------------------------------
# Plot losses for each phase and each experiments across networks

# Growing up loss plot
fig, ax1 = plt.subplots(figsize=(5, 5))

path = training_sets["set1"]
growing_up_loss = []

for net_id in range(num_nets):
    path_result = os.path.join(path, f"task_{net_id}", f"results_baseline")
    results = th.load(path_result, map_location='cpu', weights_only=False)
    growing_up_loss.append(results['growing_up']['losses']['total'])

mean_values = np.array(growing_up_loss).mean(axis=0).squeeze()
std_values = np.array(growing_up_loss).std(axis=0).squeeze()
epochs = np.arange(len(mean_values))

ax1.plot(epochs, mean_values, color='blue', linewidth=2, label=f'Mean ({num_nets} networks)')
ax1.fill_between(epochs, mean_values - std_values, mean_values + std_values, alpha=0.3, color='steelblue', label='± SD across networks')

ax1.set_title(f'Random reaching', fontsize=16)
ax1.set_xlabel('Batch', fontsize=14)
ax1.set_ylabel('Training Loss', fontsize=14)
#ax1.set_ylim(15, 40)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.tick_params(axis='both', labelsize=14)
ax1.legend(fontsize=12, frameon=False)

fig.tight_layout()
plt.show()


# loss across phases and experiments
fig, axes = plt.subplots(len(exp_names), 3, figsize=(18, 4 * len(exp_names)))
fig.suptitle(f'Training Losses across {num_nets} networks', fontsize=16)

for set in training_sets.keys():
    all_loss = all_loss_dict[set]
    for i, exp in enumerate(exp_names):
        for j, phase in enumerate(phases):
            mean_values = all_loss[exp][phase]['mean']
            std_values = all_loss[exp][phase]['std']

            epochs = np.arange(len(mean_values))

            ax = axes[i, j]
            ax.plot(epochs, mean_values, color='blue', linewidth=2)
            ax.fill_between(epochs, mean_values - std_values, mean_values + std_values, alpha=0.3, color='steelblue')
            ax.set_title(f'{exp} — {phase}')
            if i == len(exp_names) - 1: ax.set_xlabel('Batch')
            if j == 0: ax.set_ylabel('Loss')
            ax.set_ylim(15, 40)

fig.tight_layout()
plt.show()


# average loss for all experiments for each phase in a summary figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(phases):
    ax = axs[i]

    for j, exp in enumerate(exp_names):
        mean_values = all_loss[exp][phase]['mean']
        epochs = np.arange(len(mean_values))

        ax.plot(epochs, mean_values, label=exp, color=plot_table[exp]['color'])

    ax.set_title(f"Phase: {phase}")
    ax.set_xlabel("Batch")
    if i == 0: ax.set_ylabel("Losses")
    ax.grid(True)
    ax.set_ylim(15, 50)
    ax.legend()

plt.tight_layout()
plt.show()


# Plot deviations for each phase and each experiment across networks

total_curves = num_nets * len(training_sets)
colors_CW = plt.cm.Greens(np.linspace(0.4, 0.9, total_curves))
colors_CCW = plt.cm.Reds(np.linspace(0.4, 0.9, total_curves))

mean_summary = {exp: {p: [] for p in phases} for exp in exp_names}
std_summary = {exp: {p: [] for p in phases} for exp in exp_names}
batch_summary = {exp: {p: [] for p in phases} for exp in exp_names}


fig, axes = plt.subplots(len(exp_names), 3, figsize=(18, 4 * len(exp_names)))
fig.suptitle(f"{'Force Compensation' if is_force_comp else 'Lateral Deviation'} across {num_nets} networks", fontsize=16)

for i, exp in enumerate(exp_names):
    for j, phase in enumerate(phases):
        ax = axes[i, j]
        data_pool = defaultdict(list)
        for s_idx, set_name in enumerate(training_sets.keys()):
            set_batch_dirs = defaultdict(list)
            for d_name, d_param in purtb_group.items():
                curr_sum = all_data[set_name][d_name][exp][phase]
                batch_ids = sorted(curr_sum.keys(), key=lambda x: int(x))
                batch_nums = [int(b) for b in batch_ids]
                current_cmap = colors_CW if d_name == "CW" else colors_CCW

                for t_id in range(num_nets):
                    color_index = t_id + (s_idx * num_nets)
                    c = current_cmap[color_index]
                    vals = [curr_sum[b][plot_measure]['concat_only_net'][t_id].item() for b in batch_ids]
                    ax.plot(batch_nums, vals, color=c, alpha=0.7, linewidth=1)

                for b in batch_ids:
                    set_batch_dirs[b].append(curr_sum[b][plot_measure]['concat_only_net'])

            # Average CW and CCW within each network first so across-network SD/SEM reflect network variance.
            for b, dir_tensors in set_batch_dirs.items():
                data_pool[b].append(th.stack(dir_tensors, dim=0).mean(dim=0))

        means, stds = [], []
        sorted_batches = sorted(data_pool.keys(), key=lambda x: int(x))
        sorted_batch_nums = [int(b) for b in sorted_batches]

        for b in sorted_batches:
            all_points_in_batch = th.cat(data_pool[b])

            means.append(all_points_in_batch.mean().item())
            stds.append(all_points_in_batch.std(correction=1 if all_points_in_batch.numel() > 1 else 0).item())

        mean_summary[exp][phase] = means
        std_summary[exp][phase] = stds
        batch_summary[exp][phase] = sorted_batch_nums

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(sorted_batch_nums, means, color='black', linewidth=4, label="Mean" if i == 0 else "")
        ax.fill_between(sorted_batch_nums, means - stds, means + stds,
                        color='gray', alpha=0.15, label="STD" if i == 0 else "")

        ax.set_title(f'{exp} — {phase}')
        if i == len(exp_names) - 1: ax.set_xlabel('Batch')
        if j == 0: ax.set_ylabel(dev_ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(*((-85, 140) if is_force_comp else (-0.07, 0.07)))

plt.tight_layout()
plt.show()


# average lateral deviations for all experiments for each phase in a summary figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(phases):
    ax = axs[i]

    for j, exp in enumerate(exp_names):
        mean_values = mean_summary[exp][phase]
        batch_values = batch_summary[exp][phase]
        std_values = std_summary[exp][phase]

        ax.plot(batch_values, mean_values, label=exp, color=plot_table[exp]['color'])
        ax.fill_between(batch_values, np.array(mean_values) - np.array(std_values), np.array(mean_values) + np.array(std_values),
                        color='gray', alpha=0.15, label="STD" if i == 0 else "")

    ax.set_title(f"Phase: {phase}")
    ax.set_xlabel("Batch")
    if i == 0:
        ax.set_ylabel(dev_ylabel)
    ax.grid(True)
    ax.set_ylim(*((-25, 105) if is_force_comp else (-0.07, 0.07)))
    ax.legend()

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7, 7))

N_NF1_KEEP = 100

break_drawn = False
ax.axhline(0, color="black", linestyle="--", linewidth=1.5, zorder=1)
phase_ends = {}

for j, exp in enumerate(exp_names):
    offset = 0
    for i, phase in enumerate(phases):
        mean_values = np.array(mean_summary[exp][phase])
        batch_values = batch_summary[exp][phase]

        if phase == 'NF1':
            mean_values = mean_values[-N_NF1_KEEP:]
            batch_values = batch_values[-N_NF1_KEEP:]
            print(batch_values)

        x_shifted = np.array(batch_values, dtype=int) - int(batch_values[0]) + offset

        ax.plot(x_shifted, mean_values,
                label=plot_table[exp]['label'] if phase == 'NF1' else "",
                color=plot_table[exp]['color'],zorder=2, linewidth=2)



        if j == 0:
            phase_y = 108 if is_force_comp else 0.06
            if phase == 'NF1':
                x_mid = (x_shifted[-2] + x_shifted[-1]) / 2 if len(x_shifted) >= 2 else x_shifted[0]
                ax.text(
                    x_mid, phase_y, phase,
                    ha="center", va="center", fontsize=14, color = 'black'
                )
            else:
                ax.text(
                    (x_shifted[0] + x_shifted[-1]) / 2, phase_y, phase,
                    ha="center", va="center", fontsize=14, color='black'
                )
        if phase == 'FF1':
            ax.plot(x_shifted[-1], mean_values[-1], marker="s", color=plot_table[exp]['color'],
                    markersize=5, markeredgecolor="black", markeredgewidth=1, zorder=3)
            if j == 0:
                ax.axvspan(x_shifted[0], x_shifted[-1], color="whitesmoke", zorder=0)
                ax.text(
                    (x_shifted[0] + x_shifted[-1]) / 2, 88 if is_force_comp else 0.045, "Exposure",
                    ha="center", va="center", fontsize=12, color='dimgray'
                )
        if phase == 'NF2':
            ax.plot(x_shifted[0], mean_values[0], marker="s", color=plot_table[exp]['color'],
                    markersize=5, markeredgecolor="black", markeredgewidth=1, zorder=3)

        if j == 0:
            phase_ends[phase] = int(x_shifted[-1]) + 1
        offset = x_shifted[-1] + 1

ax.set_xlabel("Batch", fontsize=14)
if offset >= 600:
    nf1_end = phase_ends.get('NF1', 500)
    ff1_end = phase_ends.get('FF1', 600)
    ax.set_xlim(nf1_end - 70, offset)
    ax.set_xticks([nf1_end - 50, nf1_end, ff1_end, offset])
ax.set_ylabel(dev_ylabel, fontsize=14)
if is_force_comp:
    ax.set_ylim(-10, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=14, width=1.5, length=6)
for spine in ("left", "bottom"):
    ax.spines[spine].set_linewidth(1.5)
ax.legend(fontsize=10, frameon=False, loc="upper right")

plt.tight_layout()
plt.show()




# Plot summary at the end of FF1 and beginning of NF2 for each experiment across networks.
plot_phases = [('FF1', -1), ('NF2', 0)]

fig, axs = plt.subplots(1, 2, figsize=(4, 3))

for i, (phase_name, b_idx) in enumerate(plot_phases):
    ax = axs[i]
    for j, exp in enumerate(plot_table.keys()):
        all_points = []
        for set_name in training_sets:
            dir_vals = []
            for dir_name in purtb_group:
                summary = all_data[set_name][dir_name][exp][phase_name]
                b_ids = sorted(summary.keys(), key=lambda x: int(x))
                target_b = b_ids[b_idx]
                dir_vals.append(summary[target_b][plot_measure]['concat_only_net'])
            all_points.append(th.stack(dir_vals, dim=0).mean(dim=0))

        combined = th.cat(all_points)
        sd = combined.std(correction=1 if combined.numel() > 1 else 0).item()
        m, s = combined.mean().item(), sd / np.sqrt(combined.numel())

        c = plot_table[exp]['color']
        ax.scatter(j, m, color=c, s=20, label=exp if i == 0 else "")
        ax.errorbar(j, m, yerr=s, color=c, capsize=4, lw=2)

    # ax.set_xticks(range(len(plot_table)))
    # ax.set_xticklabels([v['num'] for v in plot_table.values()])
    ax.set_title(f"{phase_name}",  fontsize=12)
    # ax.grid(True, axis='y', alpha=0.3)
    ax.set_xlim(-0.5,len(exp_names))
    if phase_name == 'FF1':
        ax.set_title("Final Exposure")
        ax.set_ylim(*((-5, 100) if is_force_comp else (0, 0.04)))
        ax.set_yticks([0, 50, 100] if is_force_comp else [0, 0.04])
        ax.axvspan(-0.5, len(exp_names), color="whitesmoke", zorder=0)
        if is_force_comp:
            ax.axhline(0, color="black", linestyle="--", linewidth=0.5, zorder=1)
    elif phase_name == 'NF2':
        ax.set_title("Post-exposure")
        ax.set_ylim(*((-5, 100) if is_force_comp else (-0.04, 0.005)))
        ax.set_yticks([0, 50, 100] if is_force_comp else [-0.04, 0])
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5, zorder=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.set_xlabel("Experiment", fontsize=10)
    ax.tick_params(axis="both", labelsize=12)

plt.tight_layout()
plt.show()

#------------------------------------------------ Trajectory-------------------------------------------------

plot_color = [v["cmap"] for v in purtb_group.values()]

# This is added for bringing both targets and trajectories to the center
move_array = {'CW': np.array([-0.1, 0.0]),
              'CCW': np.array([ 0.1, 0.0])}

path = "set1"


for exp in exp_names:
    # fig, axes = plt.subplots(len(phases), 2, figsize=(10, 4 * len(phases)))
    fig, axes = plt.subplots(1, 2 * len(phases), figsize=(8 * len(phases), 4))
    fig.suptitle(f'Avg across networks, Exp = {exp}', fontsize=14)

    for f_ind, force_field in enumerate(['CW', 'CCW']):
        shift = move_array[force_field]
        for i, phase in enumerate(phases):
            xy_accum = {'First Batch': [], 'Last Batch': []}
            target_accum = {'First Batch': [], 'Last Batch': []}

            for net_id in range(num_nets):
                entry = all_trajectory[path][force_field][exp][net_id][phase]
                net_shift = -shift if (net_id % 2 == 1) else shift
                for label in ['First Batch', 'Last Batch']:
                    xy_n = entry[label]['xy']
                    tg_n = entry[label]['targets']
                    if exp in ['mov_diff_loc', 'pro_loc']:
                        xy_n = xy_n + th.tensor(net_shift, dtype=xy_n.dtype)
                    if exp in ['mov_diff_loc', 'vis_loc']:
                        tg_n = tg_n + th.tensor(net_shift, dtype=tg_n.dtype)
                    xy_accum[label].append(xy_n)
                    target_accum[label].append(tg_n)

            for j, label in enumerate(['First Batch', 'Last Batch']):
                n_targets = 8
                xy_net_avg = th.stack(xy_accum[label]).mean(dim=0)
                target_net_avg = th.stack(target_accum[label]).mean(dim=0)
                n_steps = xy_net_avg.shape[1]

                xy_avg = xy_net_avg.reshape(-1, n_targets, n_steps, 2).mean(dim=0)
                target_avg = target_net_avg.reshape(-1, n_targets, n_steps, 2).mean(dim=0)

                target_x = target_avg[:, -1, 0]
                target_y = target_avg[:, -1, 1]

                # ax = axes[i, j]
                ax = axes[2*i+j]
                ax.set_xlim([-0.4, 0.4])
                ax.set_ylim([0, 0.6])
                ax.set_title(f'{phase} — {label}')
                plotor(axis=ax, cart_results=xy_avg, cmap=plot_color[f_ind])
                # plotor(axis=ax, cart_results=xy_avg, cmap=plot_color[f_ind], colorbar_flag=0, linewidth=2) #This is a modified plotor function version with two new inputs to control showing the colorbar and linewidth
                ax.scatter(target_x, target_y, s=12, color='blue')

    legend_elements = [
        Line2D([0], [0], color=plt.get_cmap('Greens')(0.6), linewidth=2, label='CW'),
        Line2D([0], [0], color=plt.get_cmap('Reds')(0.6), linewidth=2, label='CCW'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True)
    fig.tight_layout()
    plt.show()



#------------------------------------------------------- Neural Data----------------------------------------------------

angles = [(20, 45), (90, 0), (0, 0), (0, 90)]
n_rows = len(exp_names)
n_cols = len(angles)

current_net = 0
t_before_go_plot = 15

plot_phases = ['FF1','NF2']

# Neural data subspaces

fig_pca, axes_grid = plt.subplots(
    n_rows, n_cols,
    figsize=(6 * n_cols, 8 * n_rows),
    subplot_kw={'projection': '3d'}
)
fig_pca.suptitle(f'Network = {current_net}, timepoint = {-1 * t_before_go_plot}', fontsize=20)

for ind, exp in enumerate(exp_names):
    path_CW = os.path.join(saveLoc, f"task_{current_net}", f"test_data_{exp}_CW")
    data_CW = th.load(path_CW, map_location='cpu', weights_only=False, mmap=True)
    data_CW = {k: data_CW[k] for k in plot_phases}

    path_CCW = os.path.join(saveLoc, f"task_{current_net}", f"test_data_{exp}_CCW")
    data_CCW = th.load(path_CCW, map_location='cpu', weights_only=False, mmap=True)
    data_CCW = {k: data_CCW[k] for k in plot_phases}

    all_hidden_reduced_both, all_labels_both, all_batch_size_both, explained_var, all_hidden = calculate_subspace(
        data=[data_CW, data_CCW],
        t_before_go=t_before_go_plot
    )
    del data_CW, data_CCW, all_hidden
    gc.collect()

    for col, (elev, azim) in enumerate(angles):
        ax = axes_grid[ind, col]
        plot_subspace_batch(
            all_hidden_reduced=all_hidden_reduced_both,
            all_labels=all_labels_both,
            all_batch_size=all_batch_size_both,
            ax=ax,
            data_label=['CW', 'CCW'],
            elev=elev,
            azim=azim,
            cmaps=[plt.cm.Oranges, plt.cm.Greens, plt.cm.Blues, plt.cm.Purples]
        )

    axes_grid[ind, 0].set_ylabel(exp, fontsize=12)

fig_pca.tight_layout()

plt.show()


# plot distance matrices

plot_phases = ['NF1', 'FF1', 'NF2']

RDM_matrix = {exp: [] for exp in exp_names}
ordered_keys = None
data_CW = {}
data_CCW = {}

for set_name, set_path in training_sets.items():
    for net_id in range(num_nets):

        for exp in exp_names:
        #for ax, exp in zip(axes_rdm, exp_names):
            loaded_CW = th.load(os.path.join(set_path, f"task_{net_id}", f"test_data_{exp}_CW"), map_location='cpu', weights_only=False, mmap=True)
            loaded_CCW = th.load(os.path.join(set_path, f"task_{net_id}", f"test_data_{exp}_CCW"), map_location='cpu', weights_only=False, mmap=True)

            data_CW = {k: loaded_CW[k] for k in plot_phases if k in loaded_CW}
            data_CCW = {k: loaded_CCW[k] for k in plot_phases if k in loaded_CCW}

            _, all_labels, all_batch_size, _, all_hidden = calculate_subspace(data=[data_CW, data_CCW],
            t_before_go=t_before_go_plot, data_label=['CW', 'CCW'], phase_label=['NF1', 'FF1', 'NF2'])

            rdm, ordered_keys = compute_distance_matrix_new(
            all_hidden=all_hidden, all_labels=all_labels, all_batch_size=all_batch_size, data_label=['CW', 'CCW'])

            RDM_matrix[exp].append(rdm)
            del loaded_CW, loaded_CCW, data_CW, data_CCW, all_hidden
            gc.collect()

fig_rdm_avg, axes_rdm_avg = plt.subplots(1, len(exp_names), figsize=(4 * len(exp_names), 4.5))

ims_avg = []
Avg_RDM_matrix = {exp: np.mean(RDM_matrix[exp], axis=0) for exp in exp_names}
global_max = max(np.max(m) for m in Avg_RDM_matrix.values())

for j, (ax, exp) in enumerate(zip(axes_rdm_avg, exp_names)):
    im = plot_distance_matrix(Avg_RDM_matrix[exp], ordered_keys, data_label=['CW', 'CCW'],
                              title=f'Exp {j + 1}', ax=ax, colorbar_show=False, max_value=global_max,
                              ylabel_flag=(j == 0))
    ims_avg.append(im)

fig_rdm_avg.tight_layout()
fig_rdm_avg.subplots_adjust(right=0.93)
cbar_ax_avg = fig_rdm_avg.add_axes([0.95, 0.15, 0.015, 0.7])
fig_rdm_avg.colorbar(ims_avg[-1], cax=cbar_ax_avg)

# Show all figures at once at the very end
plt.show()
