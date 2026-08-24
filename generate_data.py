import concurrent.futures
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ---------------------------------------------------------
# 1. Physiological Allometric Scaling
# ---------------------------------------------------------
def get_patient_physiology(weight_kg, height_cm):
    """Scales cardiac output, organ volumes, and flow rates."""
    bsa = np.sqrt((height_cm * weight_kg) / 3600.0)
    q_cardiac = 3.0 * bsa * 60.0  # L/hr (3.0 L/min/m^2 baseline)
    
    # Compartment Volumes (L)
    v_blood = 0.07 * weight_kg
    v_healthy = 0.21 * weight_kg
    v_tumor = 0.15
    v_elim = 0.03 * weight_kg
    
    # Blood Flow Distribution (70% Healthy, 25% Elimination, 5% Tumor)
    q_healthy = 0.70 * q_cardiac
    q_elim = 0.25 * q_cardiac
    q_tumor = 0.05 * q_cardiac
    
    return {
        'v_blood': v_blood, 'v_healthy': v_healthy, 'v_tumor': v_tumor, 'v_elim': v_elim,
        'q_healthy': q_healthy, 'q_elim': q_elim, 'q_tumor': q_tumor, 'q_cardiac': q_cardiac
    }

# ---------------------------------------------------------
# 2. 4-Compartment PBPK Differential System
# ---------------------------------------------------------
def pbpk_system(t, C, phys, dose_rate):
    """Governing ODEs for CSTR network kinetics."""
    C_blood, C_healthy, C_tumor, C_elim = C
    
    Kp = 1.5      # Partition coefficient
    PA = 2.0      # Permeability-surface area product (L/hr)
    k_cl = 0.8    # Elimination kinetic constant (1/hr)
    
    # 4-hour constant IV infusion profile
    R_inf = dose_rate if t <= 4.0 else 0.0
    
    # Mass Balance Derivatives (dC/dt)
    dC_blood = (1.0 / phys['v_blood']) * (
        R_inf + 
        (phys['q_healthy'] * C_healthy / Kp) + 
        (phys['q_tumor'] * C_tumor / Kp) + 
        (phys['q_elim'] * C_elim / Kp) - 
        (phys['q_cardiac'] * C_blood)
    )
    
    dC_healthy = (1.0 / phys['v_healthy']) * (
        phys['q_healthy'] * (C_blood - C_healthy / Kp) - 
        PA * (C_healthy - C_blood)
    )
    
    dC_tumor = (1.0 / phys['v_tumor']) * (
        phys['q_tumor'] * (C_blood - C_tumor / Kp) + 
        PA * (C_blood - C_tumor)
    )
    
    dC_elim = (1.0 / phys['v_elim']) * (
        phys['q_elim'] * (C_blood - C_elim / Kp) - 
        k_cl * C_elim * phys['v_elim']
    )
    
    return [dC_blood, dC_healthy, dC_tumor, dC_elim]

# ---------------------------------------------------------
# 3. Worker Function for Parallel Runs
# ---------------------------------------------------------
def run_single_simulation(sample_id, weight, height, dose_rate):
    phys = get_patient_physiology(weight, height)
    
    t_span = (0, 24)
    t_eval = np.linspace(0, 24, 100)
    C0 = [0.0, 0.0, 0.0, 0.0]
    
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
    
    # Compute output targets
    max_healthy_conc = np.max(C_healthy)
    max_tumor_conc = np.max(C_tumor)
    auc_tumor = np.trapezoid(C_tumor, t)
    auc_healthy = np.trapezoid(C_healthy, t)
    
    # Cardiotoxicity threshold limit set at 2.5 mg/L
    is_toxic = 1 if max_healthy_conc > 2.5 else 0
    
    return {
        'patient_id': f"PAT_{sample_id:04d}",
        'weight_kg': round(weight, 2),
        'height_cm': round(height, 2),
        'dose_rate_mg_hr': round(dose_rate, 2),
        'max_healthy_conc': round(max_healthy_conc, 4),
        'max_tumor_conc': round(max_tumor_conc, 4),
        'auc_tumor': round(auc_tumor, 4),
        'auc_healthy': round(auc_healthy, 4),
        'is_toxic': is_toxic
    }

# ---------------------------------------------------------
# 4. Dataset Generation Launcher
# ---------------------------------------------------------
def generate_100000_dataset(num_samples=100000):
    print(f"Generating {num_samples} patient simulations in parallel...")
    
    np.random.seed(42)  # Ensures reproducibility
    
    weights = np.random.uniform(45.0, 110.0, num_samples)
    heights = np.random.uniform(145.0, 195.0, num_samples)
    dose_rates = np.random.uniform(10.0, 100.0, num_samples)
    
    # Fast multi-core processing
    tasks = [(i + 1, weights[i], heights[i], dose_rates[i]) for i in range(num_samples)]
    
    dataset = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_single_simulation, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            dataset.append(future.result())
            
    # Sort by patient ID
    df = pd.DataFrame(dataset)
    df.sort_values(by='patient_id', inplace=True)
    
    # Save to CSV file
    file_name = "dataset_100000.csv"
    df.to_csv(file_name, index=False)
    
    print(f"Successfully generated and saved {len(df)} samples to '{file_name}'!")
    print("\nDataset Preview:")
    print(df.head())
    print("\nToxicity Distribution:")
    print(df['is_toxic'].value_counts())

if __name__ == "__main__":
    generate_100000_dataset(num_samples=100000)
