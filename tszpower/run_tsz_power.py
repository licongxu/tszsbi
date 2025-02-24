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
    'omega_cdm':  0.11933,
    'H0': 67.66, 
    'tau_reio': 0.0561,
    'ln10^{10}A_s': 3.047,
    'n_s': 0.9665, 
    "cosmo_model": 0, # use mnu-lcdm emulators

    "ell_min" : 2,
    "ell_max" : 8000,

    'dlogell': 0.2,
    'z_min' : 0.005,
    'z_max' : 3.0,
    'M_min' : 1.0e10, 
    'M_max' : 3.5e15,
    # 'pressure_profile':'GNFW', 
    "P0GNFW": 8.130,
    "c500": 1.156,
    "gammaGNFW": 0.3292,
    "alphaGNFW": 1.0620,
    "betaGNFW":5.4807,
    "B": 1.0,
    "jax": 1,
}

# 2) Setup
tszpower.classy_sz.set(allpars)
tszpower.classy_sz.compute_class_szfast()
# tszpower.warmup()

# Warm-up call (triggers JIT compilation)
# _ = tszpower.compute_integral(params_values_dict=allpars)

# 3) Compute power spectrum
start_time = time.time()
C_ell = tszpower.compute_integral(params_values_dict=allpars)
end_time = time.time()
print(f"compute_integral took {end_time - start_time:.2f} seconds")

ell = tszpower.get_ell_range()
D_ell = ell*(ell+1)*C_ell/(2*jnp.pi)*1e12
# D_ell_fid = jnp.array([6.19417603e-04, 8.77347992e-04, 1.24932097e-03, 1.78642458e-03,
#        2.56158796e-03, 3.67701090e-03, 5.27295430e-03, 7.53578208e-03,
#        1.07031406e-02, 1.50629373e-02, 2.09483340e-02, 2.87279455e-02,
#        3.88003979e-02, 5.15948466e-02, 6.75554125e-02, 8.71538487e-02,
#        1.10917877e-01, 1.39509501e-01, 1.73790227e-01, 2.14858057e-01,
#        2.64104804e-01, 3.23203749e-01, 3.94046270e-01, 4.78651284e-01,
#        5.79186171e-01, 6.97522276e-01, 8.35220428e-01, 9.93201932e-01,
#        1.17103737e+00, 1.36690425e+00, 1.57672044e+00, 1.79414378e+00,
#        2.01008331e+00, 2.21321359e+00, 2.39076857e+00, 2.52918148e+00,
#        2.61593538e+00, 2.64133660e+00, 2.59951975e+00, 2.49067650e+00,
#        2.32069591e+00, 2.10110329e+00])
# print(D_ell/D_ell_fid)
# 4) Plot
plt.figure()
plt.loglog(ell, ell*(ell+1)*C_ell/(2*jnp.pi)*1e12, label='tSZ Power')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$10^{12} D_{\ell}^{yy}$')
plt.grid(visible=True, which="both", alpha=0.2, linestyle='--')
plt.legend()
plt.title("Example tSZ Power Spectrum")
plt.savefig("tsz_power_spectrum.png")
plt.show()
