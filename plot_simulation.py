import numpy as np
import matplotlib.pyplot as plt
from generate_data import get_patient_physiology, pbpk_system
from scipy.integrate import solve_ivp

def plot_single_simulation():
    # Setup a typical patient
    weight = 75.0 # kg
    height = 175.0 # cm
    dose_rate = 50.0 # mg/hr
    
    phys = get_patient_physiology(weight, height)
    
    t_span = (0, 24)
    t_eval = np.linspace(0, 24, 200)
    C0 = [0.0, 0.0, 0.0, 0.0]
    
    print("Running simulation for typical patient...")
    sol = solve_ivp(
        fun=pbpk_system,
        t_span=t_span,
        y0=C0,
        t_eval=t_eval,
        args=(phys, dose_rate),
        method='RK45'
    )
    
    t = sol.t
    C_blood, C_healthy, C_tumor, C_elim = sol.y
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t, C_blood, label='C_blood(t) (Systemic Blood)', lw=2)
    plt.plot(t, C_healthy, label='C_healthy(t) (Healthy Tissue/Heart)', lw=2)
    plt.plot(t, C_tumor, label='C_tumor(t) (Tumor Site)', lw=2)
    plt.plot(t, C_elim, label='C_elim(t) (Elimination Organs)', lw=2)
    
    plt.axhline(y=2.5, color='r', linestyle='--', label='Toxicity Threshold (2.5 mg/L)')
    plt.axvspan(0, 4, color='gray', alpha=0.2, label='IV Infusion Period (0-4h)')
    
    plt.title('24-Hour Drug Concentration Curves (PBPK Model)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Concentration (mg/L)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    
    plt.savefig('time_series_plot.png', dpi=300)
    print("Time-series plot saved to 'time_series_plot.png'.")

if __name__ == "__main__":
    plot_single_simulation()
