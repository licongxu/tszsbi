#!/usr/bin/env python3
"""
Test script for tSZ power spectrum and foreground theory modules.
"""

import numpy as np
from tszpower_cobaya_theory import tSZ_PS_Theory, tSZ_FG_Theory

# Instantiate the theory modules.
ps_theory = tSZ_PS_Theory()
fg_theory = tSZ_FG_Theory()

# Initialize the modules.
ps_theory.initialize()
fg_theory.initialize()

# Define a set of reference parameter values (these should be near the 'ref' values in your YAML):
params = {
    "omega_b": 0.0224,
    "omega_cdm": 0.1202,
    "H0": 67.27,
    "tau_reio": 0.055,
    "ln10_10A_s": 2.9939341,  # note: this is passed as 'ln10_10A_s'
    "n_s": 0.965,
    "B": 1.4,
    # For the foreground theory:
    "A_cib": 0.66,
    "A_ir": 2.04,
    "A_rs": 0.004,
}

# Create an empty state dictionary to be filled by the theories.
state = {}

# Calculate the SZ power spectrum.
print("Running tSZ_PS_Theory.calculate()...")
try:
    ps_theory.calculate(state, **params)
except Exception as e:
    print("Error in tSZ_PS_Theory.calculate():", e)

# Calculate the SZ foreground.
print("Running tSZ_FG_Theory.calculate()...")
try:
    fg_theory.calculate(state, **params)
except Exception as e:
    print("Error in tSZ_FG_Theory.calculate():", e)

# Print the outputs.
print("\n--- Theory Module Outputs ---")
if state.get("Cl_sz") is not None:
    print("Cl_sz (dict):")
    for key, value in state["Cl_sz"].items():
        print(f"  {key}: shape = {np.shape(value)}")
else:
    print("Cl_sz: Not computed.")

if state.get("ell") is not None:
    print("ell: shape =", np.shape(state["ell"]))
else:
    print("ell: Not computed.")

if state.get("Cl_sz_foreground") is not None:
    print("Cl_sz_foreground: shape =", np.shape(state["Cl_sz_foreground"]))
else:
    print("Cl_sz_foreground: Not computed.")
