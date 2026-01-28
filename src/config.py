import numpy as np


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

        # Synaptic parameters
        'tauS': 0.010,      # Synaptic decay time constant (s)
        'E_exc': 0.0,       # Reversal potential for excitation (mV)
        'E_inh': -80.0,     # Reversal potential for inhibition (mV)
        'Pmax': 1.0,        # Synaptic conductance jump on spike (uS)
        'extInMax': 0.01,   # Noise input conductance jump (uS)
        'latency': 0.001,   # Synaptic latency (s)

        # --- Circuit Wiring (Weights) ---
        # [[TRN->TRN, VPm->TRN],
        #  [TRN->VPm, VPm->VPm]]
        'PSweight': np.array([
            [0.0, 0.0],    # Inputs to TRN (Neuron 0)
            [0.06, 0.0]    # Inputs to VPm (Neuron 1)
        ]),

        # --- Input Parameters ---
        'numCTNeurons': 35,     # Number of L6CT input neurons
        'gCTmax': np.array([0.0028, 0.0028]) * 3.0, # L6CT -> [TRN, VPm] (uS)
        'spontTRNFr': 2.0,   # Background noise rate to TRN (Hz)
        'spontVPmFr': 7.0,   # Background noise rate to VPm (Hz)

        # --- Simulation Parameters ---
        'startTime': 0.0,     # s
        'endTime': 2.0,       # s
        'ctStartTime': 1.0,   # Time L6CT input turns on (s)
        'tStep': 1e-4,        # 0.1 ms simulation time step (s)
        'nTrials': 40,        # Number of trials per session

        # --- Analysis Parameters ---
        'psth_binsz': 0.025,  # Bin size for PSTH (s)
        'sync_window_ms': 7.5, # Window for synchrony metric (+/- ms)
        'sta_window_pre_ms': 25.0,
        'sta_window_post_ms': 50.0,
    }
    return config
