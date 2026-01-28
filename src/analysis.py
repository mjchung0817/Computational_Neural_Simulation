import numpy as np
from scipy.signal import correlate


def compute_ct_synchrony(l6ct_spike_trains_all_trials, config):
    """
    Measures the synchrony of the L6CT input spike trains.
    Replicates ComputeCTSynchrony.m logic.

    Takes the full (nTrials, n_neurons, num_steps) boolean array.

    Returns:
    - pwsync_prop_list: list of all pairwise synchrony proportions
    - median synchrony: scalar
    """
    nTrials, n_neurons, num_steps = l6ct_spike_trains_all_trials.shape
    tStep = config['tStep']

    sync_window_steps = int((config['sync_window_ms'] / 1000.0) / tStep)
    corr_window_steps = int(50e-3 / tStep)

    pwsync_prop_list = []

    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):

            spk1_all_trials = l6ct_spike_trains_all_trials[:, i, :].astype(float)
            spk2_all_trials = l6ct_spike_trains_all_trials[:, j, :].astype(float)

            spk1_sum = np.sum(spk1_all_trials, axis=0)
            spk2_sum = np.sum(spk2_all_trials, axis=0)

            ccg = correlate(spk1_sum, spk2_sum, mode='full')

            full_lags = np.arange(-num_steps + 1, num_steps)
            center_index = np.where(full_lags == 0)[0][0]

            ccg_window = ccg[center_index - corr_window_steps:
                             center_index + corr_window_steps + 1]

            total_coincidences = np.sum(ccg_window)

            if total_coincidences > 0:
                sync_center_idx = len(ccg_window) // 2

                sync_coincidences = np.sum(
                    ccg_window[sync_center_idx - sync_window_steps:
                               sync_center_idx + sync_window_steps + 1]
                )
                prop = sync_coincidences / total_coincidences
                pwsync_prop_list.append(prop)

    if not pwsync_prop_list:
        return [], 0.0

    return pwsync_prop_list, np.median(pwsync_prop_list)


def compute_sta(spikes_all_trials, signal_all_trials, config):
    """
    Computes the Spike-Triggered Average (STA).
    Replicates ComputeSpikeTrigConductance.m

    - spikes_all_trials: (nTrials, nSteps) boolean array (e.g., VPm spikes)
    - signal_all_trials: (nTrials, nSteps) array (e.g., L6CT conductance)

    Returns:
    - sta_time (1D array in ms)
    - sta (1D array)
    """
    tStep = config['tStep']
    sta_pre_steps = int((config['sta_window_pre_ms'] / 1000.0) / tStep)
    sta_post_steps = int((config['sta_window_post_ms'] / 1000.0) / tStep)

    spikes_flat = spikes_all_trials.flatten()
    signal_flat = signal_all_trials.flatten()

    spike_indices = np.where(spikes_flat)[0]

    sta_clips = []
    for idx in spike_indices:
        if (idx > sta_pre_steps) and (idx < (len(spikes_flat) - sta_post_steps - 1)):
            clip = signal_flat[idx - sta_pre_steps: idx + sta_post_steps + 1]
            sta_clips.append(clip)

    if not sta_clips:
        t = np.arange(-sta_pre_steps, sta_post_steps + 1) * tStep * 1000
        return t, np.zeros_like(t)

    sta = np.mean(sta_clips, axis=0)
    sta_time = np.arange(-sta_pre_steps, sta_post_steps + 1) * tStep * 1000.0

    return sta_time, sta


def make_psth(spikes_array, config):
    """
    Generates a Peri-Stimulus Time Histogram (PSTH).
    Replicates MakePSTH.m

    - spikes_array: (nTrials, num_steps) boolean array

    Returns:
    - psth_time (1D array)
    - psth_rate (1D array in Hz)
    """
    nTrials, num_steps = spikes_array.shape
    tStep = config['tStep']
    binsz = config['psth_binsz']

    binsz_steps = int(binsz / tStep)
    num_bins = int(num_steps // binsz_steps)

    total_spikes_per_tstep = np.sum(spikes_array, axis=0)

    total_spikes_per_tstep = total_spikes_per_tstep[:num_bins * binsz_steps]
    spikes_per_bin = np.sum(
        total_spikes_per_tstep.reshape(num_bins, binsz_steps),
        axis=1
    )

    psth_rate = spikes_per_bin / nTrials / binsz

    psth_time = (np.arange(num_bins) + 0.5) * binsz

    return psth_time, psth_rate
