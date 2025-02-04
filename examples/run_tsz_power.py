#!/usr/bin/env python3

"""
Example command-line script to compute the tSZ power spectrum
and save or plot the result.
"""

import tszpower
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

import tszpower.warmup

# 1) Define parameters
allpars = {
    'omega_b': 0.02242,
    'omega_cdm': 0.1193,
    'H0': 67.66,
    'tau_reio': 0.0561,
    'ln10^{10}A_s': 3.047,
    'n_s': 0.9665,
    'M_min': 1e10,
    'M_max': 3.5e15,
    'ell_min': 2,
    'ell_max': 8000,
    'dlogell': 0.1,
    'z_min': 0.005,
    'z_max': 3.0,
    'P0GNFW': 8.130,
    'c500': 1.156,
    'gammaGNFW': 0.3292,
    'alphaGNFW': 1.0620,
    'betaGNFW': 5.4807,
    'B': 1.0,
    'jax': 1
}

# 2) Setup
tszpower.classy_sz.set(allpars)
# tszpower.classy_sz.compute_class_szfast()
tszpower.warmup()

# Warm-up call (triggers JIT compilation)
# _ = tszpower.compute_integral(params_values_dict=allpars)

# 3) Compute power spectrum
start_time = time.time()
C_ell = tszpower.compute_integral(params_values_dict=allpars)
end_time = time.time()
print(f"compute_integral took {end_time - start_time:.2f} seconds")

ell = tszpower.get_ell_range()

# 4) Plot
plt.figure()
plt.loglog(ell, ell*(ell+1)*C_ell/(2*jnp.pi)*1e12, label='tSZ Power')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$10^{12} D_{\ell}^{yy}$')
plt.grid(visible=True, which="both", alpha=0.2, linestyle='--')
plt.legend()
plt.title("Example tSZ Power Spectrum")
plt.show()
