import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import correlate
import datetime
import os

##our current code utilizes a simple LIF model without adaptation being considered. 
## This could be modified by integrating the ExcInhModelFormalParameters.m in the matlab script from Dimwawa.
def get_parameters():
    """
    Holds all biophysical and simulation parameters.
    Equivalent to ExcInhModelFormalParameters.m
    """
    
    # --- Neuron Parameters (Indices: 0=TRN, 1=VPm) ---
    config = {
        # Biophysics from LIFModel.m
        'Vth': -54.0,       # Spike threshold (mV)
        'vReset': -80.0,    # Reset potential (mV)
        'eLeak': np.array([-80.0, -70.0]), # Resting potential [TRN, VPm] (mV)
        'gL': np.array([7e-3, 2.9e-2]),     # Leak conductance [TRN, VPm] (uS)
        'memCap': np.array([1.4e-4, 2.9e-4]), # Membrane capacitance [TRN, VPm] (uF)
        'rpLength': 0.001,  # Refractory period (s)
        # 'rpLength': 0.002,  # Refractory period (s)
        
        # Synaptic parameters
        'tauS': 0.010,      # Synaptic decay time constant (s)
        'E_exc': 0.0,       # Reversal potential for excitation (mV)
        'E_inh': -80.0,     # Reversal potential for inhibition (mV)
        'Pmax': 1.0,        # Synaptic conductance jump on spike (uS)
        'extInMax': 0.01,   # Noise input conductance jump (uS)
        # 'extInMax': 1.0,   # Noise input conductance jump (uS)
        'latency': 0.001,   # Synaptic latency (s)
        
        # --- Circuit Wiring (Weights) ---
        # From LIFModel4PaperV2.m, line 42
        # [[TRN->TRN, VPm->TRN],
        #  [TRN->VPm, VPm->VPm]]
        # 'PSweight': np.array([
        #     [0.0, 0.0],    # Inputs to TRN (Neuron 0)
        #     [0.12, 0.0]    # Inputs to VPm (Neuron 1)
        # ]), ##these are synaptic weights, the VPM --> TRN and self activating is left out for model simplicity. 
        'PSweight': np.array([
            [0.0, 0.0],    # Inputs to TRN (Neuron 0)
            [0.06, 0.0]    # Inputs to VPm (Neuron 1)
        ]), ##these are synaptic weights, the VPM --> TRN and self activating is left out for model simplicity. 
        
        # --- Input Parameters ---
        'numCTNeurons': 35,     # Number of L6CT input neurons
        'gCTmax': np.array([0.0028, 0.0028]) * 3.0, # L6CT -> [TRN, VPm] (uS) ##gCTMax intentionally very small as it is multiplied by n_spikes. 
        'spontTRNFr': 2.0,   # Background noise rate to TRN (Hz), matching biological reality where not just static without l6ct
        'spontVPmFr': 7.0,   # Background noise rate to VPm (Hz)
        
        # --- Simulation Parameters ---
        'startTime': 0.0,     # s
        'endTime': 2.0,       # s
        'ctStartTime': 1.0, # Time L6CT input turns on (s)
        'tStep': 1e-4,        # 0.1 ms simulation time step (s)
        'nTrials': 40,        # Number of trials per session
        
        # --- Analysis Parameters ---
        'psth_binsz': 0.025,  # Bin size for PSTH (s)
        'sync_window_ms': 7.5, # Window for synchrony metric (+/- ms)
        # Updated to match ComputeSpikeTrigConductance.m (st=-25ms, en=50ms)
        'sta_window_pre_ms': 25.0, 
        'sta_window_post_ms': 50.0,
    }
    return config

def poisson_process(rate, duration, tStep):
    """
    Generates a boolean spike train from a homogeneous Poisson process.
    - rate (Hz): Firing rate
    - duration (s): Total time
    - tStep (s): Time step
    Returns: 1D boolean array (True at time of a spike)
    """
    num_steps = int(duration / tStep)
    # Probability of a spike in any given time bin
    prob = rate * tStep
    # Generate random numbers and compare to probability
    return np.random.rand(num_steps) < prob

def generate_l6ct_inputs(config):
    """
    Generates the 35 correlated L6CT spike trains using the "Dimwawa Algorithm".
    Replicates ctSpikes.m / ctSpikes_ShortWin.m
    
    Returns: 
    - (nTrials, numCTNeurons, num_steps) boolean array of spike trains
    """
    # Get parameters from config
    rate = config['ctFR']
    num_neurons = config['numCTNeurons']
    duration = config['endTime']
    tStep = config['tStep']
    sharedProb = config['sharedProb']
    nTrials = config['nTrials']
    
    num_steps = int(duration / tStep)
    
    start_step = int(config['ctStartTime'] / tStep)
    time_mask = np.zeros(num_steps, dtype=bool)
    time_mask[start_step:] = True # Only allow spikes AFTER start_step
    
    # Store all trials in 3d array
    all_trials_spike_trains: np.array = np.zeros((nTrials, num_neurons, num_steps), dtype=bool)

    for trial in range(nTrials): ##iterate over trials ##3d array(trials, num neurons, spike train data)
        # 1. Create one "shared" spike train
        shared_train = poisson_process(rate, duration, tStep)
        
        # 2. Create spike trains for each neuron
        all_spike_trains_this_trial = np.zeros((num_neurons, num_steps), dtype=bool)
        
        for i in range(num_neurons): ##iterate over neurons
            # 3. Create a unique "independent" train
            ind_train = poisson_process(rate, duration, tStep)
            
            # 4. Randomly select spikes from shared train
            shared_spikes_used = (shared_train) & (np.random.rand(num_steps) < sharedProb)
            
            # 5. Randomly select spikes from independent train
            ind_spikes_used = (ind_train) & (np.random.rand(num_steps) < (1.0 - sharedProb))
            
            # 6. Combine them. A spike occurs if it's in *either* set.
            combined_train = shared_spikes_used | ind_spikes_used
            
            all_spike_trains_this_trial[i] = combined_train & time_mask
            
        all_trials_spike_trains[trial] = all_spike_trains_this_trial
        
    return all_trials_spike_trains




###add a function to raster plot these 



def run_lif_simulation(config, l6ct_spike_trains_all_trials): ##l6ct_spike_trains_all_trials == all_trials_spike_trains from generate_l6ct_inputs function.
    """
    Runs all trials of the 2-neuron (TRN, VPm) LIF simulation.
    This is the Python equivalent of the core loop in LIFModel.m.
    
    Returns:
    A dictionary containing all simulation data (voltages, spikes, conductances)
    """
    
    # --- 1. Get parameters from config ---
    tStep = config['tStep']
    num_steps = int(config['endTime'] / tStep)
    nTrials = config['nTrials']
    
    # Biophysical parameters (2-element vectors: [TRN, VPm])
    eLeak = config['eLeak']
    gL = config['gL']
    memCap = config['memCap']
    Vth = config['Vth']
    vReset = config['vReset']
    rpLength = config['rpLength']
    
    
    latency = config['latency']
    # Convert latency to time steps
    latency_steps = int(latency / tStep)
    
    # Synaptic parameters
    tauS = config['tauS']
    E_exc = config['E_exc']
    E_inh = config['E_inh']
    Pmax = config['Pmax']
    extInMax = config['extInMax']
    
    # Weights
    PSweight = config['PSweight'] # Internal circuit weights
    gCTmax = config['gCTmax']     # L6CT input weights
    
    # --- 2. Initialize simulation variables ---
    
    # Output arrays (will require four of 3D arrays for 4 different output parameters(voltage, g_ct conductance generated from L6CT , g_ps conductance generated from TRN) )
    all_voltages = np.zeros((nTrials, 2, num_steps)) ##shape = (trials, [TRN,VPM] , time)
    all_output_spikes = np.zeros((nTrials, 2, num_steps), dtype=bool) 
    all_g_ct_trace = np.zeros((nTrials, 2, num_steps))
    all_g_ps_trace = np.zeros((nTrials, 2, num_steps))

    # --- 3. Run the loop for each trial ---
    for trial in tqdm(range(nTrials), desc="  Running Trials"): ##tqdm used to generate progress bar
        
        # Get inputs for this trial
        l6ct_spike_trains = l6ct_spike_trains_all_trials[trial] # (35, num_steps) ##spike train data across 35 neurons across timesteps num_steps
        l6ct_spike_counts = np.sum(l6ct_spike_trains, axis=0)  # 1D array (num_steps) #across neurons
#______________________A Bunch of initialization for conductance, baseline noise, refract time for l6ct and trn___________         
        voltages = np.zeros((2, num_steps))
        voltages[:, 0] = eLeak  # Start at resting potential for both TRN and VPM. 
        output_spikes = np.zeros((2, num_steps), dtype=bool)
        
        # Conductance state variables (g)
        g_ps = np.zeros(2)      # Internal synapse (TRN->VPm)
        g_ct = np.zeros(2)      # L6CT input
        g_noise_trn = np.zeros(2) # TRN background noise
        g_noise_vpm = np.zeros(2) # VPm background noise
        
        # Refractory period timer
        refract_timer = np.zeros(2) # Time (s) remaining in refractory period
        
        # Storage for analysis
        g_ct_trace = np.zeros((2, num_steps))
        g_ps_trace = np.zeros((2, num_steps))

        # Generate "Noise" spike trains for this trial
        noise_trn_spikes = poisson_process(config['spontTRNFr'], config['endTime'], tStep)
        noise_vpm_spikes = poisson_process(config['spontVPmFr'], config['endTime'], tStep)

        # --- 4. Run the main simulation loop (Euler's Method) ---
        # This loop replicates LIFModel.m, lines 121-229
        
        for t in range(1, num_steps):
            #At time 0, voltages[:, 0] = eLeak  # Start at resting potential for both TRN and VPM. 
            # A. Copy previous state 
            voltages[:, t] = voltages[:, t-1]
            
            # B. Handle refractory period
            not_in_refract = (refract_timer <= 0)
            in_refract = ~not_in_refract
            
            voltages[in_refract, t] = vReset
            refract_timer[in_refract] -= tStep
            
            # C. Exponential decay of conductances (for non-refractory neurons)
            # g_ps[not_in_refract] -= (g_ps[not_in_refract] * tStep / tauS)
            # g_ct[not_in_refract] -= (g_ct[not_in_refract] * tStep / tauS)
            # g_noise_trn[not_in_refract] -= (g_noise_trn[not_in_refract] * tStep / tauS)
            # g_noise_vpm[not_in_refract] -= (g_noise_vpm[not_in_refract] * tStep / tauS)
            
            ##Modification from John
            g_ps -= (g_ps * tStep / tauS)
            g_ct -= (g_ct * tStep / tauS)
            g_noise_trn -= (g_noise_trn * tStep / tauS)
            g_noise_vpm -= (g_noise_vpm * tStep / tauS)


            # D. Conductance "kicks" from incoming spikes
            
            # # L6CT input (from line 171)
            # n_spikes = l6ct_spike_counts[t]
            # if n_spikes > 0:
            #     g_ct += gCTmax * n_spikes 
            
            # # TRN noise input (from line 178)
            # if noise_trn_spikes[t]:
            #     g_noise_trn[0] += extInMax  # Index 0 is TRN
                
            # # VPm noise input (from line 175)
            # if noise_vpm_spikes[t]:
            #     g_noise_vpm[1] += extInMax  # Index 1 is VPm
            
            if t > latency_steps:
                # D. Conductance "kicks" from incoming spikes
                
                # L6CT input 
                n_spikes = l6ct_spike_counts[t - latency_steps] # Check past
                if n_spikes > 0:
                    g_ct += gCTmax * n_spikes 
                
                # TRN noise input (from line 178)
                if noise_trn_spikes[t - latency_steps]: # Check past
                    g_noise_trn[0] += extInMax  # Index 0 is TRN
                    
                # VPm noise input (from line 175)
                if noise_vpm_spikes[t - latency_steps]: # Check past
                    g_noise_vpm[1] += extInMax  # Index 1 is VPm
                
                # Internal synapse kick (from TRN spike)
                if output_spikes[0, t - latency_steps]: # Check past
                    g_ps[0] += Pmax
                    
            # E. Calculate currents (I = g * (V - E)) (for non-refractory neurons)
            V_old = voltages[not_in_refract, t-1] ##V_old is now filtered for neurons that are not in their refractory period. 
            
            leakCurrent = gL[not_in_refract] * (eLeak[not_in_refract] - V_old)
            ctCurrent = g_ct[not_in_refract] * (E_exc - V_old)
            noiseTrnCurrent = g_noise_trn[not_in_refract] * (E_exc - V_old)
            noiseVpmCurrent = g_noise_vpm[not_in_refract] * (E_exc - V_old)
            
            # Internal circuit current (I_ps) (from line 127)
            preSynCurrent = np.zeros(np.sum(not_in_refract))
            if not_in_refract[1]: 
                vpm_idx = np.sum(not_in_refract[:1]) 
                preSynCurrent[vpm_idx] = g_ps[0] * (E_inh - V_old[vpm_idx]) * PSweight[1,0]
                
            # F. Calculate dV/dt (The Differential Equation) (from line 131)
            I_total = (leakCurrent + ctCurrent + noiseTrnCurrent + 
                       noiseVpmCurrent + preSynCurrent)
            
            dV = I_total / memCap[not_in_refract]
            
            # G. Integrate (Euler's Method) (from line 147)
            voltages[not_in_refract, t] = V_old + dV * tStep
            
            # H. Spike and Reset (from lines 192-229)
            firing = (voltages[:, t] > Vth)
            
            if np.any(firing):
                output_spikes[firing, t] = True
                voltages[firing, t] = vReset      
                refract_timer[firing] = rpLength  
                
            #     if firing[0]: # If TRN (neuron 0) fired
            #         g_ps[0] += Pmax

            # Store conductances for analysis
            g_ct_trace[:, t] = g_ct
            g_ps_trace[0, t] = g_ps[0] # Store raw TRN conductance
        
        # --- End of single trial loop ---
        all_voltages[trial] = voltages
        all_output_spikes[trial] = output_spikes
        all_g_ct_trace[trial] = g_ct_trace
        all_g_ps_trace[trial] = g_ps_trace

    # --- 5. Return all results ---
    return {
        "voltages": all_voltages, # (nTrials, 2, nSteps)
        "output_spikes": all_output_spikes, # (nTrials, 2, nSteps)
        "conductances": {
            "g_ct": all_g_ct_trace,   # (nTrials, 2, nSteps)
            "g_ps_inh": all_g_ps_trace # (nTrials, 2, nSteps)
        }
    }

def plot_synchrony_relationship(shared_prob_values, measured_sync_medians, measured_sync_all_pairs, current_date_time, plot_dir):
    """
    Plots the relationship between the input 'sharedProb' parameter 
    and the actual measured synchrony, with error bars showing the
    interquartile range (IQR) of all neuron pairs.
    """
    print("  Plotting synchrony relationship...")
    
    # Calculate error bars (Interquartile Range - IQR)
    y_errors_lower = []
    y_errors_upper = []
    
    # Iterate over the medians and the corresponding full lists
    for median, all_pairs_list in zip(measured_sync_medians, measured_sync_all_pairs):
        # Get 25th and 75th percentiles
        q1, q3 = np.percentile(all_pairs_list, [25, 75])
        
        # Calculate error relative to the median
        lower_error = median - q1
        upper_error = q3 - median
        
        y_errors_lower.append(lower_error)
        y_errors_upper.append(upper_error)
    
    # Combine into the (2, N) shape required by plt.errorbar
    y_errors = [y_errors_lower, y_errors_upper]
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use plt.errorbar instead of plt.plot
    ax.errorbar(
        shared_prob_values, 
        measured_sync_medians, 
        yerr=y_errors, 
        fmt='bo-',          # Format: blue, circles, solid line
        markersize=8, 
        linewidth=2,
        capsize=5,          # Add error bar caps
        label='Median (IQR)'
    )
    
    ax.set_title("Model Synchrony Control Verification", fontsize=16)
    ax.set_xlabel("Input Synchrony Parameter (sharedProb)", fontsize=12)
    ax.set_ylabel("Measured L6CT Synchrony (Median Pairwise Prop.)", fontsize=12)
    
  # --- NEW: Auto-zooming axes (based ONLY on median points) ---
    
    # --- X-Axis Zoom ---
    x_data = np.array(shared_prob_values)
    x_range = x_data.max() - x_data.min()
    # Add 10% padding, or a minimum of 0.1 padding if range is 0
    x_padding = max(x_range * 0.1, 0.1) 
    ax.set_xlim(x_data.min() - x_padding, x_data.max() + x_padding)
    
    # --- Y-Axis Zoom (Based ONLY on medians) ---
    y_data_min = np.min(measured_sync_medians)
    y_data_max = np.max(measured_sync_medians)
    y_range = y_data_max - y_data_min
    # Add 10% padding, or a minimum of 0.1 padding if range is 0
    y_padding = max(y_range * 0.1, 0.01) 
    ax.set_ylim(y_data_min - y_padding, y_data_max + y_padding)
    
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Ensure plot directory exists
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    filename = f"{plot_dir}/synchrony_relationship_plot_{current_date_time}.png"
    plt.savefig(filename)
    plt.close(fig)

def compute_ct_synchrony(l6ct_spike_trains_all_trials: np.array, config): 
    """
    Measures the synchrony of the L6CT input spike trains.
    Replicates ComputeCTSynchrony.m logic.
    
    Takes the full (nTrials, n_neurons, num_steps) boolean array.
    
    Returns:
    - A scalar: the median "Pair-Wise Synchrony (Proportion)"
    """
    nTrials, n_neurons, num_steps = l6ct_spike_trains_all_trials.shape
    tStep = config['tStep']
    
    # Get the +/- window size in *time steps*
    sync_window_steps = int((config['sync_window_ms'] / 1000.0) / tStep)
    # Full correlation window
    corr_window_steps = int(50e-3 / tStep) # +/- 50ms, same as MATLAB
    
    pwsync_prop_list = [] # List to hold all 595 pair scores
    
    start_step = int(config['ctStartTime'] / config['tStep'])
    
    for i in range(n_neurons):
        for j in range(i + 1, n_neurons):
            
            # Get the spike trains for this pair *across all trials*
            spk1_all_trials = l6ct_spike_trains_all_trials[:, i, :].astype(float)
            spk2_all_trials = l6ct_spike_trains_all_trials[:, j, :].astype(float)
            
            # Sum spikes across trials *before* correlating
            # This replicates the MATLAB logic of summing `pairccg` across trials
            #still retains the time dimension
            spk1_sum = np.sum(spk1_all_trials, axis=0)
            spk2_sum = np.sum(spk2_all_trials, axis=0)
            
            # Calculate the cross-correlation
            # mode='full' gives all lags, 'same' crops to input length
            # We want a specific lag window, so 'full' is better
            ccg = correlate(spk1_sum, spk2_sum, mode='full')
            
            # Lags for 'full' mode go from -(N-1) to +(N-1)
            full_lags = np.arange(-num_steps + 1, num_steps)
            center_index = np.where(full_lags == 0)[0][0]
            
            # Get the +/- 50ms window (same as MATLAB's CCGDur)
            ccg_window = ccg[center_index - corr_window_steps : 
                             center_index + corr_window_steps + 1]
            
            total_coincidences = np.sum(ccg_window)
            
            if total_coincidences > 0:
                # Sum the coincidences *within* the +/- 7.5ms window
                # Find the center of our *new* ccg_window
                sync_center_idx = len(ccg_window) // 2
                
                sync_coincidences = np.sum(
                    ccg_window[sync_center_idx - sync_window_steps : 
                               sync_center_idx + sync_window_steps + 1]
                )
                prop = sync_coincidences / total_coincidences
                pwsync_prop_list.append(prop)
    
    if not pwsync_prop_list:
        return [],0.0
        
    # Return the median of all 595 pairs
    return pwsync_prop_list, np.median(pwsync_prop_list)

def compute_sta(spikes_all_trials, signal_all_trials, config):
    """
    Computes the Spike-Triggered Average (STA).
    Replicates ComputeSpikeTrigConductance.m
    
    - spikes_all_trials: (nTrials, nSteps) boolean array (e.g., VPm spikes)
    - signal_all_trials: (nTrials, nSteps) array (e.g., L6CT conductance)
    
    Returns:
    - sta_time (1D array)
    - sta (1D array)
    """
    tStep = config['tStep']
    # Use asymmetric window from MATLAB
    sta_pre_steps = int((config['sta_window_pre_ms'] / 1000.0) / tStep)
    sta_post_steps = int((config['sta_window_post_ms'] / 1000.0) / tStep)
    
    num_steps = spikes_all_trials.shape[1]
    
    ##Only use spikes AFTER ctStartTime ---
    start_step = int(config['ctStartTime'] / tStep)
    
    # Flatten all trials into one long stream
    spikes_flat = spikes_all_trials.flatten()
    signal_flat = signal_all_trials.flatten()
    
    # Find all the spike indices
    spike_indices = np.where(spikes_flat)[0]
    
    sta_clips = []
    for idx in spike_indices:
        # Ensure the clip is not at the edge
        if (idx > sta_pre_steps) and (idx < (len(spikes_flat) - sta_post_steps - 1)):
            clip = signal_flat[idx - sta_pre_steps : idx + sta_post_steps + 1]
            sta_clips.append(clip)
            
    if not sta_clips:
        # Return empty arrays if no spikes
        t = np.arange(-sta_pre_steps, sta_post_steps + 1) * tStep * 1000
        return t, np.zeros_like(t)

    # Average all clips together
    sta = np.mean(sta_clips, axis=0)
    
    # Create the time vector (in ms)
    sta_time = np.arange(-sta_pre_steps, sta_post_steps + 1) * tStep * 1000.0
    
    return sta_time, sta


import matplotlib.pyplot as plt
import numpy as np

def plot_l6ct_raster(l6ct_spikes_trial, config, shared_prob, session_index, current_date_time):
    """
    Plots a raster of the L6CT input neurons for a single trial.
    - l6ct_spikes_trial: (nNeurons, nSteps) boolean array for one trial
    - config: The simulation config dictionary
    - shared_prob: The 'p' value for this session
    - session_index: The index (0, 1, 2...) of this session
    - current_date_time: The timestamp for unique filenames
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
    
    # Ensure plot directory exists
    plot_dir = "plots_generated_from_Dimwamwa_python"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    filename = f"{plot_dir}/l6ct_raster_p{session_index}_{current_date_time}.png"
    plt.savefig(filename)
    plt.close(fig)

import matplotlib.pyplot as plt
import numpy as np

def plot_vpm_trace(sim_data, config, shared_prob, session_index, current_date_time):
    """
    Plots the VPm voltage and conductances for a single trial.
    - sim_data: The full output dictionary from the simulation
    - config: The simulation config dictionary
    - shared_prob: The 'p' value for this session
    - session_index: The index (0, 1, 2...) of this session
    - current_date_time: The timestamp for unique filenames
    """
    print(f"  Plotting VPm Single Trial Trace for p={shared_prob}...")
    tStep = config['tStep']
    num_steps = int(config['endTime'] / tStep)
    times = np.arange(num_steps) * tStep
    
    # Use the first trial
    trial_index = 0
    
    # Get Vm
    vpm_vm = sim_data['voltages'][trial_index, 1, :]
    
    # Get Conductances
    g_ct_vpm = sim_data['conductances']['g_ct'][trial_index, 1, :]
    g_inh_vpm = sim_data['conductances']['g_ps_inh'][trial_index, 0, :] * config['PSweight'][1,0]
    
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
    # Plot spikes
    spike_times = np.where(sim_data['output_spikes'][trial_index, 1, :])[0] * tStep
    if spike_times.size > 0:
        ax1.plot(spike_times, np.ones_like(spike_times) * config['Vth'], 'rv', markersize=5, label='Spikes')
    
    ax1.axhline(config['Vth'], color='gray', linestyle=':', label='Threshold')
    ax1.axvline(config['ctStartTime'], color='r', linestyle='--', label='L6CT On')
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_ylim(-85, -50) # Zoom in on subthreshold dynamics
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

    # Ensure plot directory exists
    plot_dir = "plots_generated_from_Dimwamwa_python"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    filename = f"{plot_dir}/vpm_trace_p{session_index}_{current_date_time}.png"
    plt.savefig(filename)
    plt.close(fig)

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
    
    # 1. Sum spikes across all trials
    total_spikes_per_tstep = np.sum(spikes_array, axis=0)
    
    # 2. Reshape into bins and sum
    # Trim to be an even multiple of bin size
    total_spikes_per_tstep = total_spikes_per_tstep[:num_bins * binsz_steps]
    spikes_per_bin = np.sum(
        total_spikes_per_tstep.reshape(num_bins, binsz_steps), 
        axis=1
    )
    
    # 3. Average and normalize to get rate (Hz)
    # (Total Spikes) / (nTrials) / (bin_duration_in_seconds)
    psth_rate = spikes_per_bin / nTrials / binsz
    
    # 4. Create time vector (center of each bin)
    psth_time = (np.arange(num_bins) + 0.5) * binsz
    
    return psth_time, psth_rate
    


def main():
    """
    Main "Wrapper" function to run the experiment.
    Replicates LIFModel4PaperV2.m
    """
    print("Initializing simulation...")
    
    ## NEW ##
    # Get a single timestamp for all plots in this run
    current_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    ## NEW ##
    # Ensure plot directory exists
    plot_dir = "plots_generated_from_Dimwamwa_python"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    # --- 1. Define the Experiment ---
    base_config = get_parameters()
    # Define the parameter sweep for L6CT synchrony
    shared_prob_sweep = [0.0, 0.3, 0.6, 0.9] # Low, Medium, High synchrony 
    
    ## Define the L6CT firing rate to test
    ct_fr_to_test = 17.6 # Hz (from LIFModel4PaperV2.m, line 49)  
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
    measured_sync_all_pairs =[]
    
    # --- 3. Run the Parameter Sweep ---
    # This is the "for sweepnum = 1:10" loop (Line 55)
    for i, shared_prob in enumerate(shared_prob_sweep):
        
        print(f"\n--- Running Session {i+1}/{num_sessions} (sharedProb = {shared_prob}) ---")
        config = base_config.copy()
        config['sharedProb'] = shared_prob
        
        # --- A. Generate Inputs (Replicates ctSpikes) ---
        # Returns (nTrials, nNeurons, nSteps)
        l6ct_spikes_all_trials = generate_l6ct_inputs(config)
        
        # --- B. Run Simulation (Replicates LIFModel.m loop) ---
        # Returns dict with (nTrials, 2, nSteps) arrays
        sim_data = run_lif_simulation(config, l6ct_spikes_all_trials)
        
        ## NEW ##
        # --- C. Plot Single-Trial Diagnostics ---
        # Plot L6CT Raster (using trial 0)
        plot_l6ct_raster(l6ct_spikes_all_trials[0, :, :], config, shared_prob, i, current_date_time)
        
        # Plot VPm Voltage & Conductance Trace (using trial 0)
        plot_vpm_trace(sim_data, config, shared_prob, i, current_date_time)
        
        
        # --- D. Analyze Input Synchrony (Replicates ComputeCTSynchrony) ---
        print("  Analyzing input synchrony...")
        pwsync_prop_list, final_measured_sync = compute_ct_synchrony(l6ct_spikes_all_trials, config)
        session_title = f"Input Sync (p={shared_prob:.1f}), Measured={final_measured_sync:.3f}"
        
        # Ssve measured synchrony strength
        measured_sync_results.append(final_measured_sync)
        measured_sync_all_pairs.append(pwsync_prop_list)
        
        # Get (nTrials, nSteps) arrays for VPm and TRN
        vpm_spikes_arr = sim_data['output_spikes'][:, 1, :] # VPm is index 1
        trn_spikes_arr = sim_data['output_spikes'][:, 0, :] # TRN is index 0
        
        # --- Plot 1: VPm Raster ---
        print("  Plotting summary figure...")
        ax = axes[0, i]
        for trial_idx, spikes in enumerate(vpm_spikes_arr):
            spike_times = np.where(spikes)[0] * config['tStep']
            ax.plot(spike_times, np.ones_like(spike_times) * trial_idx, 'k.', markersize=1)
        ax.set_title(f"VPm Raster\n{session_title}")
        ax.set_ylabel("Trial #")
        ax.axvline(config['ctStartTime'], color='r', linestyle='--', label='L6CT On')

        # --- Plot 2: VPm PSTH (Replicates MakePSTH) ---
        ax = axes[1, i]
        psth_time, psth_rate = make_psth(vpm_spikes_arr, config)
        ax.plot(psth_time, psth_rate, 'k')
        ax.set_title("VPm PSTH")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.axvline(config['ctStartTime'], color='r', linestyle='--')
        ax.set_ylim(bottom=0)
        
        # --- Plot 3: TRN PSTH (Replicates MakePSTH) ---
        # (This plotting logic is unchanged...)
        ax = axes[2, i]
        psth_time, psth_rate = make_psth(trn_spikes_arr, config)
        ax.plot(psth_time, psth_rate, 'b')
        ax.set_title("TRN PSTH")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.axvline(config['ctStartTime'], color='r', linestyle='--')
        ax.set_ylim(bottom=0)

        # --- Plot 4: Spike-Triggered Conductance (Replicates Fig 4d) ---
        # (This plotting logic is unchanged...)
        ax = axes[3, i]
        
        # Get (nTrials, nSteps) arrays for VPm conductances
        # Excitatory L6CT conductance onto VPm (neuron 1)
        g_ct_vpm = sim_data['conductances']['g_ct'][:, 1, :]
        
        # Inhibitory TRN->VPm conductance
        g_inh_vpm = sim_data['conductances']['g_ps_inh'][:, 0, :] * config['PSweight'][1,0]

        # STA for Excitatory (L6CT) Conductance
        sta_time, sta_exc = compute_sta(vpm_spikes_arr, g_ct_vpm, config)
        ax.plot(sta_time, sta_exc, 'g', label='Exc. (from L6CT)')
        
        # STA for Inhibitory (TRN) Conductance
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
    axes[3,0].set_xlim(-sta_pre, sta_post)
    axes[3,1].set_xlim(-sta_pre, sta_post)
    axes[3,2].set_xlim(-sta_pre, sta_post)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    ## NEW ##
    # Save the main figure with the timestamp
    main_fig_filename = f"{plot_dir}/lif_model_replication_{current_date_time}.png"
    plt.savefig(main_fig_filename)
    plt.close(fig) # Close the main figure
    
    # Validate that our synchrony generation worked 
    plot_synchrony_relationship(shared_prob_sweep, measured_sync_results, measured_sync_all_pairs, current_date_time,plot_dir)
    
    print(f"\n--- Simulation complete. Plots saved to '{plot_dir}' with timestamp '{current_date_time}' ---")





# def main():
#     """
#     Main "Wrapper" function to run the experiment.
#     Replicates LIFModel4PaperV2.m
#     """
#     print("Initializing simulation...")
    
#     # --- 1. Define the Experiment ---
#     base_config = get_parameters()
#     # Define the parameter sweep for L6CT synchrony
#     shared_prob_sweep = [0.0, 0.4, 0.9] # Low, Medium, High synchrony 
    
#     ## Define the L6CT firing rate to test
#     ct_fr_to_test = 5.0 # Hz (from LIFModel4PaperV2.m, line 49)  
#     base_config['ctFR'] = ct_fr_to_test
    
#     nTrials = base_config['nTrials']
    
#     # --- 2. Create Figure for Plotting ---
#     num_sessions = len(shared_prob_sweep)
#     fig, axes = plt.subplots(
#         nrows=4, 
#         ncols=num_sessions, 
#         figsize=(5 * num_sessions, 12), 
#         squeeze=False
#     )
#     fig.suptitle(f"LIF Model (L6CT FR = {ct_fr_to_test} Hz)", fontsize=16)

#     # --- 3. Run the Parameter Sweep ---
#     # This is the "for sweepnum = 1:10" loop (Line 55)
#     for i, shared_prob in enumerate(shared_prob_sweep):
        
#         print(f"\n--- Running Session {i+1}/{num_sessions} (sharedProb = {shared_prob}) ---")
#         config = base_config.copy()
#         config['sharedProb'] = shared_prob
        
#         # --- A. Generate Inputs (Replicates ctSpikes) ---
#         # Returns (nTrials, nNeurons, nSteps)
#         l6ct_spikes_all_trials = generate_l6ct_inputs(config)
        
#         # --- B. Run Simulation (Replicates LIFModel.m loop) ---
#         # Returns dict with (nTrials, 2, nSteps) arrays
#         sim_data = run_lif_simulation(config, l6ct_spikes_all_trials)
        
#         # --- C. Analyze Input Synchrony (Replicates ComputeCTSynchrony) ---
#         # This is now done once per session, matching MATLAB logic
#         print("  Analyzing input synchrony...")
#         final_measured_sync = compute_ct_synchrony(l6ct_spikes_all_trials, config)
#         session_title = f"Input Sync (p={shared_prob:.1f}), Measured={final_measured_sync:.3f}"
            
#         # --- 4. Analyze and Plot Session Results ---
        
#         # Get (nTrials, nSteps) arrays for VPm and TRN
#         vpm_spikes_arr = sim_data['output_spikes'][:, 1, :] # VPm is index 1
#         trn_spikes_arr = sim_data['output_spikes'][:, 0, :] # TRN is index 0
        
#         # --- Plot 1: VPm Raster ---
#         print("  Plotting...")
#         ax = axes[0, i]
#         for trial_idx, spikes in enumerate(vpm_spikes_arr):
#             spike_times = np.where(spikes)[0] * config['tStep']
#             ax.plot(spike_times, np.ones_like(spike_times) * trial_idx, 'k.', markersize=1)
#         ax.set_title(f"VPm Raster\n{session_title}")
#         ax.set_ylabel("Trial #")
#         ax.axvline(config['ctStartTime'], color='r', linestyle='--', label='L6CT On')

#         # --- Plot 2: VPm PSTH (Replicates MakePSTH) ---
#         ax = axes[1, i]
#         psth_time, psth_rate = make_psth(vpm_spikes_arr, config)
#         ax.plot(psth_time, psth_rate, 'k')
#         ax.set_title("VPm PSTH")
#         ax.set_ylabel("Firing Rate (Hz)")
#         ax.axvline(config['ctStartTime'], color='r', linestyle='--')
#         ax.set_ylim(bottom=0)
        
#         # --- Plot 3: TRN PSTH (Replicates MakePSTH) ---
#         ax = axes[2, i]
#         psth_time, psth_rate = make_psth(trn_spikes_arr, config)
#         ax.plot(psth_time, psth_rate, 'b')
#         ax.set_title("TRN PSTH")
#         ax.set_ylabel("Firing Rate (Hz)")
#         ax.axvline(config['ctStartTime'], color='r', linestyle='--')
#         ax.set_ylim(bottom=0)

#         # --- Plot 4: Spike-Triggered Conductance (Replicates Fig 4d) ---
#         ax = axes[3, i]
        
#         # Get (nTrials, nSteps) arrays for VPm conductances
#         # Excitatory L6CT conductance onto VPm (neuron 1)
#         g_ct_vpm = sim_data['conductances']['g_ct'][:, 1, :]
        
#         # Inhibitory TRN->VPm conductance
#         # This is g_ps[0] * PSweight[1,0]
#         # g_ps[0] is stored in g_ps_inh[:, 0, :]
#         # PSweight[1,0] is config['PSweight'][1,0]
#         g_inh_vpm = sim_data['conductances']['g_ps_inh'][:, 0, :] * config['PSweight'][1,0]

#         # STA for Excitatory (L6CT) Conductance
#         # Trigger is VPm spikes (vpm_spikes_arr)
#         # Signal is L6CT conductance (g_ct_vpm)
#         sta_time, sta_exc = compute_sta(vpm_spikes_arr, g_ct_vpm, config)
#         ax.plot(sta_time, sta_exc, 'g', label='Exc. (from L6CT)')
        
#         # STA for Inhibitory (TRN) Conductance
#         # Trigger is VPm spikes (vpm_spikes_arr)
#         # Signal is TRN conductance (g_inh_vpm)
#         sta_time, sta_inh = compute_sta(vpm_spikes_arr, g_inh_vpm, config)
#         ax.plot(sta_time, sta_inh, 'b', label='Inh. (from TRN)')
        
#         ax.set_title("Spike-Triggered Conductance (VPm)")
#         ax.set_xlabel("Time from VPm Spike (ms)")
#         ax.set_ylabel("Conductance (uS)")
#         ax.axvline(0, color='k', linestyle=':', label='VPm Spike')
#         ax.legend()

#     # --- 5. Finalize Plot ---
#     sta_pre = config['sta_window_pre_ms']
#     sta_post = config['sta_window_post_ms']
#     for ax_row in axes:
#         for ax in ax_row:
#             ax.set_xlim(0, config['endTime'])
#     axes[3,0].set_xlim(-sta_pre, sta_post)
#     axes[3,1].set_xlim(-sta_pre, sta_post)
#     axes[3,2].set_xlim(-sta_pre, sta_post)
    
#     current_date_time = datetime.datetime.now()  
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.savefig(f"plots_generated_from_Dimwamwa_python/lif_model_replication_{current_date_time}.ps_Weight = 0.06.png")
#     print(f"\n--- Simulation complete. Plot saved to 'lif_model_replication_{current_date_time}.png' ---")


if __name__ == "__main__":
    # This block acts as the main "wrapper" script
    # (like LIFModel4PaperV2.m or synchronymodel.m)
    main()

