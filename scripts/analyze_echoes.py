import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def analyze():
    print("--- KERNEL_GRAVITY: Observation Module ---")
    
    try:
        data = pd.read_csv('../echoes.csv')
    except:
        data = pd.read_csv('echoes.csv')
        
    signal = data['signal'].values
    step = data['step'].values
    
    # 1. Visualization of the raw signal
    plt.figure(figsize=(12, 6))
    plt.plot(step, signal, label='GW Flux (Density Boundary)')
    plt.title('Gravitational Echo Flux - Spacetime Grid')
    plt.xlabel('Step')
    plt.ylabel('Mass Density Flux')
    plt.grid(True)
    plt.savefig('scripts/echo_signal.png')
    print("Raw signal plot saved to scripts/echo_signal.png")

    # 2. Fourier Transform to find periodicities
    N = len(signal)
    T = 1.0 # assume unit time steps for now
    yf = fft(signal - np.mean(signal)) # Remove DC component
    xf = fftfreq(N, T)[:N//2]
    
    psd = 2.0/N * np.abs(yf[0:N//2])
    
    plt.figure(figsize=(12, 6))
    plt.plot(xf, psd)
    plt.title('Power Spectral Density (Echo frequencies)')
    plt.xlabel('Frequency (1/steps)')
    plt.ylabel('Power')
    plt.grid(True)
    plt.savefig('scripts/fourier_analysis.png')
    print("Fourier analysis plot saved to scripts/fourier_analysis.png")

    # 3. Findings
    peaks = np.where(psd > np.max(psd) * 0.5)[0]
    if len(peaks) > 0:
        main_freq = xf[peaks[0]]
        print(f"Main Echo Frequency Detected: {main_freq:.4f}")
        print("This frequency corresponds to the bounce-rate off the Quantum Solid core.")
    else:
        print("No significant periodicity detected yet. Simulation may need more steps.")

if __name__ == "__main__":
    analyze()
