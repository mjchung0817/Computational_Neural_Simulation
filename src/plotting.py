import os

import numpy as np
import matplotlib.pyplot as plt


def plot_synchrony_relationship(shared_prob_values, measured_sync_medians,
                                measured_sync_all_pairs, current_date_time, plot_dir):
    """
    Plots the relationship between the input 'sharedProb' parameter
    and the actual measured synchrony, with IQR error bars.
    """
    print("  Plotting synchrony relationship...")

    y_errors_lower = []
    y_errors_upper = []

    for median, all_pairs_list in zip(measured_sync_medians, measured_sync_all_pairs):
        q1, q3 = np.percentile(all_pairs_list, [25, 75])
        y_errors_lower.append(median - q1)
        y_errors_upper.append(q3 - median)

    y_errors = [y_errors_lower, y_errors_upper]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        shared_prob_values,
        measured_sync_medians,
        yerr=y_errors,
        fmt='bo-',
        markersize=8,
        linewidth=2,
        capsize=5,
        label='Median (IQR)'
    )

    ax.set_title("Model Synchrony Control Verification", fontsize=16)
    ax.set_xlabel("Input Synchrony Parameter (sharedProb)", fontsize=12)
    ax.set_ylabel("Measured L6CT Synchrony (Median Pairwise Prop.)", fontsize=12)

    x_data = np.array(shared_prob_values)
    x_range = x_data.max() - x_data.min()
    x_padding = max(x_range * 0.1, 0.1)
    ax.set_xlim(x_data.min() - x_padding, x_data.max() + x_padding)

    y_data_min = np.min(measured_sync_medians)
    y_data_max = np.max(measured_sync_medians)
    y_range = y_data_max - y_data_min
    y_padding = max(y_range * 0.1, 0.01)
    ax.set_ylim(y_data_min - y_padding, y_data_max + y_padding)

    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    filename = os.path.join(plot_dir, f"synchrony_relationship_plot_{current_date_time}.png")
    plt.savefig(filename)
    plt.close(fig)


def plot_l6ct_raster(l6ct_spikes_trial, config, shared_prob, session_index,
                     current_date_time, plot_dir):
    """
    Plots a raster of the L6CT input neurons for a single trial.
    """
    print(f"  Plotting L6CT Raster for p={shared_prob}...")
    tStep = config['tStep']
    num_neurons, num_steps = l6ct_spikes_trial.shape

    fig, ax = plt.subplots(figsize=(10, 5))

    for neuron_idx in range(num_neurons):
        spikes = l6ct_spikes_trial[neuron_idx, :]
        spike_times = np.where(spikes)[0] * tStep
        if spike_times.size > 0:
            ax.plot(spike_times, np.ones_like(spike_times) * neuron_idx, 'k.', markersize=4)

    ax.axvline(config['ctStartTime'], color='r', linestyle='--', label='L6CT On')
    ax.set_xlim(0, config['endTime'])
    ax.set_ylim(-1, num_neurons)
    ax.set_ylabel("L6CT Neuron #")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"L6CT Input Spike Raster (Single Trial)\nInput Sync (p={shared_prob:.1f})")

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    filename = os.path.join(plot_dir, f"l6ct_raster_p{session_index}_{current_date_time}.png")
    plt.savefig(filename)
    plt.close(fig)


def plot_vpm_trace(sim_data, config, shared_prob, session_index,
                   current_date_time, plot_dir):
    """
    Plots the VPm voltage and conductances for a single trial.
    """
    print(f"  Plotting VPm Single Trial Trace for p={shared_prob}...")
    tStep = config['tStep']
    num_steps = int(config['endTime'] / tStep)
    times = np.arange(num_steps) * tStep

    trial_index = 0

    vpm_vm = sim_data['voltages'][trial_index, 1, :]

    g_ct_vpm = sim_data['conductances']['g_ct'][trial_index, 1, :]
    g_inh_vpm = sim_data['conductances']['g_ps_inh'][trial_index, 0, :] * config['PSweight'][1, 0]

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}
    )

    fig.suptitle(f"VPm Single Trial Dynamics\nInput Sync (p={shared_prob:.1f})", fontsize=16)

    # --- Plot 1: VPm Voltage ---
    ax1.plot(times, vpm_vm, 'k', label='VPm Vm')
    spike_times = np.where(sim_data['output_spikes'][trial_index, 1, :])[0] * tStep
    if spike_times.size > 0:
        ax1.plot(spike_times, np.ones_like(spike_times) * config['Vth'], 'rv',
                 markersize=5, label='Spikes')

    ax1.axhline(config['Vth'], color='gray', linestyle=':', label='Threshold')
    ax1.axvline(config['ctStartTime'], color='r', linestyle='--', label='L6CT On')
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_ylim(-85, -50)
    ax1.legend(loc='upper right')
    ax1.set_title("VPm Membrane Potential (Trial 0)")

    # --- Plot 2: Conductances ---
    ax2.plot(times, g_ct_vpm, 'g', label='Exc. (from L6CT)')
    ax2.plot(times, g_inh_vpm, 'b', label='Inh. (from TRN)')
    ax2.axvline(config['ctStartTime'], color='r', linestyle='--')
    ax2.set_ylabel("Conductance (uS)")
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim(0, config['endTime'])
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper right')
    ax2.set_title("VPm Conductances (Trial 0)")

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    filename = os.path.join(plot_dir, f"vpm_trace_p{session_index}_{current_date_time}.png")
    plt.savefig(filename)
    plt.close(fig)
