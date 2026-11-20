import os
import torch as th
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


exp_names = ['center_out','mov_diff_loc','pro_loc','vis_loc']
phases = ['NF1','FF1','NF2']
# measures = ['endpoint', 'lateral']
measures = ['endpoint_all', 'lateral_all']
num_nets = 20


# Summary functions

def summarize_deviations(base_path, suffix, phases = phases, exp_names = exp_names, num_nets = num_nets, measures = measures):
    # Loads deviations and returns a nested summary dictionary.
    summary_deviation = {exp: {p: {} for p in phases} for exp in exp_names}

    for exp in exp_names:
        for phase in phases:
            for measure in measures:
                collected = defaultdict(list)
                for net_id in range(num_nets):
                    # Load data
                    path = os.path.join(base_path, f"task_{net_id}", f"deviations_{exp}_{suffix}")
                    dev_data = th.load(path, map_location='cpu', weights_only=False)
                    batch_dict = dev_data[phase]
                    for batch_id, value in batch_dict.items():
                        collected[batch_id].append(th.cat([t.detach().cpu() for t in value[measure]], dim=0))
                for batch_id, task_values in collected.items():
                    stacked = th.stack(task_values, dim=0)
                    corr = 1 if stacked.shape[0] > 1 else 0
                    if batch_id not in summary_deviation[exp][phase]:
                        summary_deviation[exp][phase][batch_id] = {}
                    summary_deviation[exp][phase][batch_id][measure] = {
                        "concat": stacked,
                        "concat_only_net": stacked.mean(dim = 1),
                        "mean_net": stacked.mean(dim=0),
                        "std_net": stacked.std(dim=0, correction=corr),
                        "mean_all": stacked.mean(),
                        "std_all": stacked.std(correction=1 if stacked.numel() > 1 else 0),
                    }
    return summary_deviation



def summarize_trajectory(base_path, suffix, phases = phases, exp_names = exp_names, num_nets = num_nets):
    # Loads test data and returns a nested summary dictionary of trajectory and targets.
    summary_testdata = {exp: {} for exp in exp_names}

    for exp in exp_names:
        for net_id in range(num_nets):
            # Load data
            path = os.path.join(base_path, f"task_{net_id}", f"test_data_{exp}_{suffix}")
            test_data = th.load(path, map_location='cpu', weights_only=False, mmap=True)
            summary_testdata[exp][net_id] = {}
            for phase in phases:
                batches = list(test_data[phase])
                fb, lb = batches[0], batches[-1]

                summary_testdata[exp][net_id][phase] = {}
                for batch, label in [(fb, 'First Batch'), (lb, 'Last Batch')]:
                    episode_data = test_data[phase][batch]
                    summary_testdata[exp][net_id][phase][label] = {
                        'xy': episode_data['xy'].detach()[:, :, :2].clone(),
                        'targets': episode_data['targets'].detach().clone()
                    }
            del test_data
    return summary_testdata



def summerize_loss(base_path, phases = phases, exp_names = exp_names, num_nets = num_nets):

    summary_loss = {exp: {p: {} for p in phases} for exp in exp_names}

    for exp in exp_names:
        all_losses = {p: [] for p in phases}
        for net_id in range(num_nets):
            path_result = os.path.join(base_path, f"task_{net_id}", f"results_{exp}")
            results = th.load(path_result, map_location='cpu', weights_only=False)
            for phase in phases:
                all_losses[phase].append(results[phase]['losses']['total'])

        for phase in phases:
            summary_loss[exp][phase] = {
                "stack": np.array(all_losses[phase]),
                "mean": np.array(all_losses[phase]).mean(axis=0),
                "std": np.array(all_losses[phase]).std(axis=0)}

    return summary_loss




# This function calculate the hidden neural activity subspaces at the time relative to go cue
def calculate_subspace(data, t_before_go=10, data_label = None, phase_label = None):
    all_hidden = []
    all_batch_size = []
    all_labels = []


    datalist = data if isinstance(data, list) else [data]

    data_names = data_label if data_label is not None else range(len(datalist))

    for dataset_idx, dataset in enumerate(datalist):
        phase_names = phase_label if phase_label is not None else list(dataset.keys())
        for phase_idx, phase in enumerate(phase_names):
            phase_data = dataset[phase]
            batch_list = list(phase_data.keys())
            for train_batch_number in batch_list:
                data_temp = phase_data[train_batch_number]
                go_signal = data_temp['inp'][:, :, 2]
                is_zero = (go_signal < 0.5)
                indices = th.where(is_zero.any(dim=1), th.argmax(is_zero.int(), dim=1), go_signal.shape[1])
                timepoints = th.clamp(indices - t_before_go, min=0, max=data_temp['hidden'].shape[1] - 1).to(th.long)
                batch_size = indices.shape[0]
                all_batch_size.append(batch_size)
                batch_indices = th.arange(batch_size, dtype=th.long)
                # Clone sliced hidden states so memory-mapped test_data files can be released.
                all_hidden.append(data_temp['hidden'][batch_indices, timepoints, :].detach().clone())
                all_labels.append((data_names[dataset_idx], phase_names[phase_idx]))

    n_components = 3
    pca = PCA(n_components=n_components)
    # Cast to float64 before PCA to avoid float32 matmul overflow warnings.
    all_hidden_reduced = pca.fit_transform(np.squeeze(th.cat(all_hidden, dim=0).detach().cpu().numpy()).astype(np.float64))
    # print('Explained Variance = ', pca.explained_variance_ratio_)

    return all_hidden_reduced, all_labels, all_batch_size, pca.explained_variance_ratio_, all_hidden



# This function calculate the distance matrices from the dimensionally reducted hidden data
def compute_distance_matrix_new(all_hidden, all_labels, all_batch_size, phase_order=None, data_label=None):

    if phase_order is None:
        phase_order = ['NF1', 'FF1', 'NF2']

    pre_data, pre_phase = None, None
    i_data_phase = 0

    all_hidden_concat = np.stack([t.detach().cpu().numpy() for t in all_hidden], axis=0)
    all_hidden_reshape = np.transpose(all_hidden_concat, axes = (1, 0, 2))

    raw_keys = []
    for dataset_idx, phase_name in all_labels:
        if pre_data == dataset_idx and pre_phase == phase_name:
            i_data_phase += 1
        else:
            i_data_phase = 0

        raw_keys.append((dataset_idx, phase_name, i_data_phase))
        pre_data, pre_phase = dataset_idx, phase_name

    # Preserve caller dataset order (e.g. ['CW', 'CCW']) while ordering phases canonical-first.
    key_to_idx = {k: idx for idx, k in enumerate(raw_keys)}
    dataset_list = list(dict.fromkeys(ds for ds, _, _ in raw_keys))
    ordered_keys = []
    for dataset_idx in dataset_list:
        for phase in phase_order:
            batch_indices = sorted(k[2] for k in raw_keys if k[0] == dataset_idx and k[1] == phase)
            for batch_idx in batch_indices:
                ordered_keys.append((dataset_idx, phase, batch_idx))
    order = [key_to_idx[k] for k in ordered_keys]

    rdm_batches = []
    for batch_idx in range(all_hidden_reshape.shape[0]):
        hidden_matrix = all_hidden_reshape[batch_idx, order, :]
        rdm_batches.append(cdist(hidden_matrix, hidden_matrix, metric='euclidean'))

    rdm = np.mean(rdm_batches, axis=0)

    return rdm, ordered_keys


def compute_pred_matrix(results, phase_order=None, metric='r2_test'):

    if phase_order is None:
        phase_order = ['NF1', 'FF1', 'NF2']

    # unique (phase, batch) pairs, ordered by phase_order then batch number
    keys = sorted(
        {(k[0], k[1]) for k in results.keys()},
        key=lambda pb: (phase_order.index(pb[0]), pb[1])
    )

    n = len(keys)
    matrix = np.full((n, n), np.nan)

    for i, (train_phase, train_batch) in enumerate(keys):
        for j, (test_phase, test_batch) in enumerate(keys):
            key = (train_phase, train_batch, test_phase, test_batch)
            if key in results:
                matrix[i, j] = results[key][metric]

    return matrix, keys
