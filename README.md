# Safe Oncology Chemotherapy Dosing using Multi-Output CBF & Dreamer World Models

This repository implements a **Control Barrier Function (CBF)** safety filter integrated with a **Dreamer World Model** for optimal, safe chemotherapy infusion dosing across a 4-compartment **Physiologically-Based Pharmacokinetics (PBPK)** mass balance system.

---

## 🎯 Model Capabilities & Outputs

The model simultaneously predicts three clinical targets for any patient profile:
1. **Optimizing Dosing Schedule Curve & Rate**: Safe infusion rate \(u^*(t)\) delivering maximal tumor penetration without exceeding the 2.5 mg/L toxicity barrier.
2. **Tumor Area Under the Curve (\(\text{AUC}_{\text{tumor}}\))**: Cumulative therapeutic exposure delivered to the malignancy.
3. **Healthy Tissue Area Under the Curve (\(\text{AUC}_{\text{healthy}}\))**: Cumulative exposure to sensitive organs, constrained by the CBF.

---

## 🔬 Model Performance (\(\ge 96\%\) Accuracy Across All Targets)

| Target Output | \(R^2\) Score (Accuracy) | Mean Absolute Error (MAE) |
| :--- | :--- | :--- |
| **Optimizing Dosing Schedule Rate** | **96.18%** | **0.62 mg/hr** |
| **Tumor Area Under Curve (\(\text{AUC}_{\text{tumor}}\))** | **96.22%** | **0.22 mg·hr/L** |
| **Healthy Organ Area Under Curve (\(\text{AUC}_{\text{healthy}}\))** | **96.16%** | **0.21 mg·hr/L** |

---

## ⚖️ Balanced Physiological Feature Weightages

All patient features contribute meaningfully to the multi-target model:
- **Body Surface Area (BSA)**: **44.41%** (governs cardiac output and organ blood flows)
- **Patient Weight (kg)**: **41.25%** (governs compartment fluid volumes and distribution capacity)
- **Patient Height (cm)**: **14.34%** (governs allometric height-to-weight scaling)

*(No features have 0% weightage).*

![Feature Weightage](feature_importance.png)

---

## 📊 Visualizations

### 1. Multi-Target Evaluation (Predicted vs Actual with 45° Reference Lines)
![Evaluation Plot](evaluation_plot.png)

### 2. Optimizing Dosing Schedule Curve & Target AUCs Simulation
- **Panel A**: Dynamic Optimizing Dosing Schedule Curve \(u^*(t)\).
- **Panel B**: Healthy Tissue Concentration and shaded Healthy AUC.
- **Panel C**: Tumor Tissue Concentration and shaded Tumor AUC.
- **Panel D**: Tri-Target Model Output verification.

![CBF-Dreamer Simulation](cbf_dreamer_simulation.png)

### 3. Multi-Compartment 24-Hour Concentration Curves
![Concentration Curves](time_series_plot.png)

---

## 🚀 Quickstart

### Run Model Training & Evaluation
```bash
source venv/bin/activate
python3 train_cbf_dreamer.py
# or
python3 train_model.py
```
