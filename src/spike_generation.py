import numpy as np


def poisson_process(rate, duration, tStep):
    """
    Generates a boolean spike train from a homogeneous Poisson process.
    - rate (Hz): Firing rate
    - duration (s): Total time
    - tStep (s): Time step
    Returns: 1D boolean array (True at time of a spike)
    """
    num_steps = int(duration / tStep)
    prob = rate * tStep
    return np.random.rand(num_steps) < prob


def generate_l6ct_inputs(config):
    """
    Generates the 35 correlated L6CT spike trains using the "Dimwawa Algorithm".
    Replicates ctSpikes.m / ctSpikes_ShortWin.m

    Returns:
    - (nTrials, numCTNeurons, num_steps) boolean array of spike trains
    """
    rate = config['ctFR']
    num_neurons = config['numCTNeurons']
    duration = config['endTime']
    tStep = config['tStep']
    sharedProb = config['sharedProb']
    nTrials = config['nTrials']

    num_steps = int(duration / tStep)

    start_step = int(config['ctStartTime'] / tStep)
    time_mask = np.zeros(num_steps, dtype=bool)
    time_mask[start_step:] = True

    all_trials_spike_trains = np.zeros((nTrials, num_neurons, num_steps), dtype=bool)

    for trial in range(nTrials):
        shared_train = poisson_process(rate, duration, tStep)

        all_spike_trains_this_trial = np.zeros((num_neurons, num_steps), dtype=bool)

        for i in range(num_neurons):
            ind_train = poisson_process(rate, duration, tStep)

            shared_spikes_used = (shared_train) & (np.random.rand(num_steps) < sharedProb)
            ind_spikes_used = (ind_train) & (np.random.rand(num_steps) < (1.0 - sharedProb))

            combined_train = shared_spikes_used | ind_spikes_used
            all_spike_trains_this_trial[i] = combined_train & time_mask

        all_trials_spike_trains[trial] = all_spike_trains_this_trial

    return all_trials_spike_trains
