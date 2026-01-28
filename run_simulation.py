"""
Main entry point for the thalamocortical LIF simulation.
Replicates the experiment from LIFModel4PaperV2.m

Usage:
    python run_simulation.py
"""
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt

from src.config import get_parameters
from src.spike_generation import generate_l6ct_inputs
from src.lif_model import run_lif_simulation
from src.analysis import compute_ct_synchrony, compute_sta, make_psth
from src.plotting import plot_synchrony_relationship, plot_l6ct_raster, plot_vpm_trace


def main():
    print("Initializing simulation...")

    current_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    plot_dir = os.path.join(os.getcwd(), "plots_generated_from_Dimwamwa_python")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # --- 1. Define the Experiment ---
    base_config = get_parameters()
    shared_prob_sweep = [0.0, 0.3, 0.6, 0.9]

    ct_fr_to_test = 17.6  # Hz
    base_config['ctFR'] = ct_fr_to_test

    nTrials = base_config['nTrials']

    # --- 2. Create Figure for Plotting ---
    num_sessions = len(shared_prob_sweep)
    fig, axes = plt.subplots(
        nrows=4,
        ncols=num_sessions,
        figsize=(5 * num_sessions, 12),
        squeeze=False
    )
    fig.suptitle(f"LIF Model (L6CT FR = {ct_fr_to_test} Hz)", fontsize=16)

    measured_sync_results = []
    measured_sync_all_pairs = []

    # --- 3. Run the Parameter Sweep ---
    for i, shared_prob in enumerate(shared_prob_sweep):

        print(f"\n--- Running Session {i+1}/{num_sessions} (sharedProb = {shared_prob}) ---")
        config = base_config.copy()
        config['sharedProb'] = shared_prob

        # --- A. Generate Inputs ---
        l6ct_spikes_all_trials = generate_l6ct_inputs(config)

        # --- B. Run Simulation ---
        sim_data = run_lif_simulation(config, l6ct_spikes_all_trials)

        # --- C. Plot Single-Trial Diagnostics ---
        plot_l6ct_raster(l6ct_spikes_all_trials[0, :, :], config, shared_prob,
                         i, current_date_time, plot_dir)

        plot_vpm_trace(sim_data, config, shared_prob, i, current_date_time, plot_dir)

        # --- D. Analyze Input Synchrony ---
        print("  Analyzing input synchrony...")
        pwsync_prop_list, final_measured_sync = compute_ct_synchrony(
            l6ct_spikes_all_trials, config
        )
        session_title = f"Input Sync (p={shared_prob:.1f}), Measured={final_measured_sync:.3f}"

        measured_sync_results.append(final_measured_sync)
        measured_sync_all_pairs.append(pwsync_prop_list)

        vpm_spikes_arr = sim_data['output_spikes'][:, 1, :]
        trn_spikes_arr = sim_data['output_spikes'][:, 0, :]

        # --- Plot 1: VPm Raster ---
        print("  Plotting summary figure...")
        ax = axes[0, i]
        for trial_idx, spikes in enumerate(vpm_spikes_arr):
            spike_times = np.where(spikes)[0] * config['tStep']
            ax.plot(spike_times, np.ones_like(spike_times) * trial_idx, 'k.', markersize=1)
        ax.set_title(f"VPm Raster\n{session_title}")
        ax.set_ylabel("Trial #")
        ax.axvline(config['ctStartTime'], color='r', linestyle='--', label='L6CT On')

        # --- Plot 2: VPm PSTH ---
        ax = axes[1, i]
        psth_time, psth_rate = make_psth(vpm_spikes_arr, config)
        ax.plot(psth_time, psth_rate, 'k')
        ax.set_title("VPm PSTH")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.axvline(config['ctStartTime'], color='r', linestyle='--')
        ax.set_ylim(bottom=0)

        # --- Plot 3: TRN PSTH ---
        ax = axes[2, i]
        psth_time, psth_rate = make_psth(trn_spikes_arr, config)
        ax.plot(psth_time, psth_rate, 'b')
        ax.set_title("TRN PSTH")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.axvline(config['ctStartTime'], color='r', linestyle='--')
        ax.set_ylim(bottom=0)

        # --- Plot 4: Spike-Triggered Conductance ---
        ax = axes[3, i]

        g_ct_vpm = sim_data['conductances']['g_ct'][:, 1, :]
        g_inh_vpm = sim_data['conductances']['g_ps_inh'][:, 0, :] * config['PSweight'][1, 0]

        sta_time, sta_exc = compute_sta(vpm_spikes_arr, g_ct_vpm, config)
        ax.plot(sta_time, sta_exc, 'g', label='Exc. (from L6CT)')

        sta_time, sta_inh = compute_sta(vpm_spikes_arr, g_inh_vpm, config)
        ax.plot(sta_time, sta_inh, 'b', label='Inh. (from TRN)')

        ax.set_title("Spike-Triggered Conductance (VPm)")
        ax.set_xlabel("Time from VPm Spike (ms)")
        ax.set_ylabel("Conductance (uS)")
        ax.axvline(0, color='k', linestyle=':', label='VPm Spike')
        ax.legend()

    # --- 5. Finalize Plot ---
    sta_pre = config['sta_window_pre_ms']
    sta_post = config['sta_window_post_ms']
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(0, config['endTime'])

    for col_idx in range(num_sessions):
        axes[3, col_idx].set_xlim(-sta_pre, sta_post)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    main_fig_filename = os.path.join(plot_dir, f"lif_model_replication_{current_date_time}.png")
    plt.savefig(main_fig_filename)
    plt.close(fig)

    plot_synchrony_relationship(shared_prob_sweep, measured_sync_results,
                                measured_sync_all_pairs, current_date_time, plot_dir)

    print(f"\n--- Simulation complete. Plots saved to '{plot_dir}' with timestamp "
          f"'{current_date_time}' ---")


if __name__ == "__main__":
    main()
