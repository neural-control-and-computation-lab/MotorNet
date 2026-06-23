import motornet as mn
import matplotlib.pyplot as plt
from Plot_utils import *
from matplotlib.lines import Line2D
from Plots import plot_subspace_batch, plot_distance_matrix


plotor = mn.plotor.plot_pos_over_time

exp_names = ['center_out','mov_diff_loc','pro_loc','vis_loc']
phases = ['NF1','FF1','NF2']
measures = ['endpoint', 'lateral']
num_nets = 20


training_sets = {
    "set1": "/Users/pounemirzazadeh/Motornet/MultiNet/Modular_version",
}
saveLoc = training_sets["set1"]


purtb_group = {
    "CW":  {"suffix": "CW", "cmap": plt.cm.Greens},
    "CCW": {"suffix": "CCW", "cmap": plt.cm.Reds}}


plot_table = {
    'center_out': {'num': '2', 'color': 'blue'},
    'vis_loc': {'num': '7', 'color': 'orange'},
    'pro_loc': {'num': '8', 'color': 'green'},
    'mov_diff_loc': {'num': '3', 'color': 'red'}
}



# Load and concatenate loss and deviation data

all_data = {s_name: {d_name: summarize_deviations(path, param["suffix"], phases = phases, exp_names = exp_names)
               for d_name, param in purtb_group.items()}
               for s_name, path in training_sets.items()}

all_loss_dict = {s_name: summerize_loss(path, phases = phases, exp_names = exp_names)
               for s_name, path in training_sets.items()}

all_trajectory = {s_name: {d_name: summarize_trajectory(path, param["suffix"],phases = phases, exp_names = exp_names)
               for d_name, param in purtb_group.items()}
               for s_name, path in training_sets.items()}


#----------------------------------------------- Loss and Deviations----------------------------------------------
# Plot losses for each phase and each experiments across networks

# Growing up loss plot
fig, ax1 = plt.subplots(figsize=(6, 4))

path = training_sets["set1"]
growing_up_loss = []

for net_id in range(num_nets):
    path_result = os.path.join(path, f"task_{net_id}", f"results_baseline")
    results = th.load(path_result)
    growing_up_loss.append(results['growing_up']['losses']['total'])

mean_values = np.array(growing_up_loss).mean(axis=0).squeeze()
std_values = np.array(growing_up_loss).std(axis=0).squeeze()
epochs = np.arange(len(mean_values))

ax1.plot(epochs, mean_values, color='blue', linewidth=2)
ax1.fill_between(epochs, mean_values - std_values, mean_values + std_values, alpha=0.3, color='steelblue')

ax1.set_title(f'Training Losses across {num_nets} networks - growing_up', fontsize=12)
ax1.set_xlabel('Batch')
ax1.set_ylabel('Loss')
#ax1.set_ylim(15, 40)

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
            if i == len(exp_names) :ax.set_xlabel('Batch')
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


# Plot deviations for each phase and each experiments across networks

total_curves = num_nets * len(training_sets)
colors_CW = plt.cm.Greens(np.linspace(0.4, 0.9, total_curves))
colors_CCW = plt.cm.Reds(np.linspace(0.4, 0.9, total_curves))

mean_summary = {exp: {p: [] for p in phases} for exp in exp_names}
std_summary = {exp: {p: [] for p in phases} for exp in exp_names}
batch_summary = {exp: {p: [] for p in phases} for exp in exp_names}


fig, axes = plt.subplots(len(exp_names), 3, figsize=(18, 4 * len(exp_names)))
fig.suptitle(f'Lateral Deviation across {num_nets} networks', fontsize=16)

for i, exp in enumerate(exp_names):
    for j, phase in enumerate(phases):
        ax = axes[i, j]
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

        ax.set_title(f'{exp} — {phase}')
        if i == len(exp_names): ax.set_xlabel('Batch')
        if j == 0: ax.set_ylabel('Lateral Deviation')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.07, 0.07)

plt.tight_layout()
plt.show()


# average lateral deviations for all experiments for each phase in a summary figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, phase in enumerate(phases):
    ax = axs[i]

    for j, exp in enumerate(exp_names):
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


# Plot summary at the end of FF1 and beginning of NF2 for each experiment across networks.
plot_phases = [('FF1', -1), ('NF2', 0)]

fig, axs = plt.subplots(1, 2, figsize=(6, 5), sharey=True)

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

axs[0].legend(loc='upper right')
plt.tight_layout()
plt.show()

#------------------------------------------------ Trajectory-------------------------------------------------

plot_color = [v["cmap"] for v in purtb_group.values()]

path = "set1"


for exp in exp_names:
    fig, axes = plt.subplots(len(phases), 2, figsize=(10, 4 * len(phases)))
    fig.suptitle(f'Avg across networks, Exp = {exp}', fontsize=14)

    for f_ind, force_field in enumerate(['CW', 'CCW']):
        for i, phase in enumerate(phases):
            xy_accum = {'First Batch': [], 'Last Batch': []}
            target_accum = {'First Batch': [], 'Last Batch': []}

            for net_id in range(num_nets):
                entry = all_trajectory[path][force_field][exp][net_id][phase]
                for label in ['First Batch', 'Last Batch']:
                    xy_accum[label].append(entry[label]['xy'])
                    target_accum[label].append(entry[label]['targets'])

            for j, label in enumerate(['First Batch', 'Last Batch']):
                n_targets = 8
                xy_net_avg = th.stack(xy_accum[label]).mean(dim=0)
                target_net_avg = th.stack(target_accum[label]).mean(dim=0)

                xy_avg = xy_net_avg.reshape(-1, n_targets, 200, 2).mean(dim=0)
                target_avg = target_net_avg.reshape(-1, n_targets, 200, 2).mean(dim=0)

                target_x = target_avg[:, -1, 0]
                target_y = target_avg[:, -1, 1]

                ax = axes[i, j]
                ax.set_xlim([-0.4, 0.4])
                ax.set_ylim([0, 0.6])
                ax.set_title(f'{phase} — {label}')
                plotor(axis=ax, cart_results=xy_avg, cmap=plot_color[f_ind])
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
    path_CW = os.path.join(training_sets[set_name], f"task_{current_net}", f"test_data_{exp}_CW")
    data_CW = th.load(path_CW)
    data_CW = {k: data_CW[k] for k in plot_phases}

    path_CCW = os.path.join(training_sets[set_name], f"task_{current_net}", f"test_data_{exp}_CCW")
    data_CCW = th.load(path_CCW)
    data_CCW = {k: data_CCW[k] for k in plot_phases}

    all_hidden_reduced_both, all_labels_both, all_batch_size_both = calculate_subspace(
        data=[data_CW, data_CCW],
        t_before_go=t_before_go_plot
    )

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
            cmaps=[plt.cm.Oranges, plt.cm.Reds, plt.cm.Blues, plt.cm.Purples]
        )

    axes_grid[ind, 0].set_ylabel(exp, fontsize=12)

fig_pca.tight_layout()

plt.show()


# plot distance matrices

RDM_matrix = {exp: [] for exp in exp_names}
ordered_keys = None

for set_name, set_path in training_sets.items():
    for net_id in range(num_nets):

        #fig_rdm, axes_rdm = plt.subplots(1, len(exp_names), figsize=(6 * len(exp_names), 8))
        #fig_rdm.suptitle(f'Network = {net_id}, t = {t_before_go_plot}', fontsize=20)

        for exp in exp_names:
        #for ax, exp in zip(axes_rdm, exp_names):
            loaded_CW = th.load(os.path.join(set_path, f"task_{net_id}", f"test_data_{exp}_CW"))
            loaded_CCW = th.load(os.path.join(set_path, f"task_{net_id}", f"test_data_{exp}_CCW"))

            data_CW = {k: loaded_CW[k] for k in plot_phases if k in loaded_CW}
            data_CCW = {k: loaded_CCW[k] for k in plot_phases if k in loaded_CCW}

            all_hidden_reduced, all_labels, all_batch_size = calculate_subspace(
                data=[data_CW, data_CCW],
                t_before_go=t_before_go_plot, data_label = ['CW', 'CCW'], phase_label = plot_phases
            )

            rdm, ordered_keys = compute_distance_matrix(
                all_hidden_reduced=all_hidden_reduced,
                all_labels=all_labels,
                all_batch_size=all_batch_size,
                data_label=['CW', 'CCW']
            )

            RDM_matrix[exp].append(rdm)
            #plot_distance_matrix(rdm, ordered_keys,
               #      title=f'Exp = {exp}', ax=ax)

        #fig_rdm.tight_layout()

    # Averaged RDM across networks
    fig_rdm_avg, axes_rdm_avg = plt.subplots(1, len(exp_names), figsize=(6 * len(exp_names), 8))
    fig_rdm_avg.suptitle(f'Averaged across {num_nets} networks, t = {t_before_go_plot}', fontsize=20)

    for ax, exp in zip(axes_rdm_avg, exp_names):
        avg_rdm = np.mean(RDM_matrix[exp], axis=0)  # (num_nets, N, N) -> (N, N)
        plot_distance_matrix(avg_rdm, ordered_keys, data_label=['CW', 'CCW'],
                 title=f'Exp = {exp}', ax=ax)

    fig_rdm_avg.tight_layout()

plt.show()

