import numpy as np
import matplotlib.pyplot as plt

def visualize_shadow():
    print("--- KERNEL_GRAVITY: Shadow Analysis ---")
    
    # We will generate a mock shadow based on the predicted density profile
    # until we can export a full 3D slice from the engine.
    
    size = 128
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    dist = np.sqrt(x**2 + y**2)
    
    # Clock speed logic: clock = 1 / (1 + density_overflow)
    # Inside the core, clock speed drops to 0. 
    # The Event Horizon Shadow is where clock speed < 0.01 (Light takes 'forever' to cross)
    
    r_shadow = 0.2
    shadow = 1.0 / (1.0 + np.exp(-(dist - r_shadow) * 50))
    
    plt.figure(figsize=(8, 8))
    plt.imshow(shadow, cmap='hot', extent=[-1, 1, -1, 1])
    plt.colorbar(label='Clock Speed (Relative Time)')
    plt.title('Simulated Event Horizon Shadow (Quantum Solid Core)')
    plt.xlabel('X (Planck Units)')
    plt.ylabel('Y (Planck Units)')
    plt.savefig('scripts/shadow_mock.png')
    print("Shadow visualization saved to scripts/shadow_mock.png")

if __name__ == "__main__":
    visualize_shadow()
