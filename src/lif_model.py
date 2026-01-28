import numpy as np
from tqdm import tqdm

from .spike_generation import poisson_process


def run_lif_simulation(config, l6ct_spike_trains_all_trials):
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

    eLeak = config['eLeak']
    gL = config['gL']
    memCap = config['memCap']
    Vth = config['Vth']
    vReset = config['vReset']
    rpLength = config['rpLength']

    latency = config['latency']
    latency_steps = int(latency / tStep)

    tauS = config['tauS']
    E_exc = config['E_exc']
    E_inh = config['E_inh']
    Pmax = config['Pmax']
    extInMax = config['extInMax']

    PSweight = config['PSweight']
    gCTmax = config['gCTmax']

    # --- 2. Initialize output arrays ---
    all_voltages = np.zeros((nTrials, 2, num_steps))
    all_output_spikes = np.zeros((nTrials, 2, num_steps), dtype=bool)
    all_g_ct_trace = np.zeros((nTrials, 2, num_steps))
    all_g_ps_trace = np.zeros((nTrials, 2, num_steps))

    # --- 3. Run the loop for each trial ---
    for trial in tqdm(range(nTrials), desc="  Running Trials"):

        l6ct_spike_trains = l6ct_spike_trains_all_trials[trial]
        l6ct_spike_counts = np.sum(l6ct_spike_trains, axis=0)

        voltages = np.zeros((2, num_steps))
        voltages[:, 0] = eLeak
        output_spikes = np.zeros((2, num_steps), dtype=bool)

        g_ps = np.zeros(2)
        g_ct = np.zeros(2)
        g_noise_trn = np.zeros(2)
        g_noise_vpm = np.zeros(2)

        refract_timer = np.zeros(2)

        g_ct_trace = np.zeros((2, num_steps))
        g_ps_trace = np.zeros((2, num_steps))

        noise_trn_spikes = poisson_process(config['spontTRNFr'], config['endTime'], tStep)
        noise_vpm_spikes = poisson_process(config['spontVPmFr'], config['endTime'], tStep)

        # --- 4. Main simulation loop (Euler's Method) ---
        for t in range(1, num_steps):
            voltages[:, t] = voltages[:, t-1]

            # Handle refractory period
            not_in_refract = (refract_timer <= 0)
            in_refract = ~not_in_refract

            voltages[in_refract, t] = vReset
            refract_timer[in_refract] -= tStep

            # Exponential decay of conductances
            g_ps -= (g_ps * tStep / tauS)
            g_ct -= (g_ct * tStep / tauS)
            g_noise_trn -= (g_noise_trn * tStep / tauS)
            g_noise_vpm -= (g_noise_vpm * tStep / tauS)

            if t > latency_steps:
                # Conductance "kicks" from incoming spikes
                n_spikes = l6ct_spike_counts[t - latency_steps]
                if n_spikes > 0:
                    g_ct += gCTmax * n_spikes

                if noise_trn_spikes[t - latency_steps]:
                    g_noise_trn[0] += extInMax

                if noise_vpm_spikes[t - latency_steps]:
                    g_noise_vpm[1] += extInMax

                # Internal synapse kick (from TRN spike)
                if output_spikes[0, t - latency_steps]:
                    g_ps[0] += Pmax

            # Calculate currents (I = g * (V - E))
            V_old = voltages[not_in_refract, t-1]

            leakCurrent = gL[not_in_refract] * (eLeak[not_in_refract] - V_old)
            ctCurrent = g_ct[not_in_refract] * (E_exc - V_old)
            noiseTrnCurrent = g_noise_trn[not_in_refract] * (E_exc - V_old)
            noiseVpmCurrent = g_noise_vpm[not_in_refract] * (E_exc - V_old)

            # Internal circuit current (TRN->VPm inhibition)
            preSynCurrent = np.zeros(np.sum(not_in_refract))
            if not_in_refract[1]:
                vpm_idx = np.sum(not_in_refract[:1])
                preSynCurrent[vpm_idx] = g_ps[0] * (E_inh - V_old[vpm_idx]) * PSweight[1, 0]

            # Calculate dV/dt
            I_total = (leakCurrent + ctCurrent + noiseTrnCurrent +
                       noiseVpmCurrent + preSynCurrent)

            dV = I_total / memCap[not_in_refract]

            # Integrate (Euler's Method)
            voltages[not_in_refract, t] = V_old + dV * tStep

            # Spike and Reset
            firing = (voltages[:, t] > Vth)

            if np.any(firing):
                output_spikes[firing, t] = True
                voltages[firing, t] = vReset
                refract_timer[firing] = rpLength

            # Store conductances for analysis
            g_ct_trace[:, t] = g_ct
            g_ps_trace[0, t] = g_ps[0]

        # --- End of single trial loop ---
        all_voltages[trial] = voltages
        all_output_spikes[trial] = output_spikes
        all_g_ct_trace[trial] = g_ct_trace
        all_g_ps_trace[trial] = g_ps_trace

    # --- 5. Return all results ---
    return {
        "voltages": all_voltages,
        "output_spikes": all_output_spikes,
        "conductances": {
            "g_ct": all_g_ct_trace,
            "g_ps_inh": all_g_ps_trace,
        }
    }
