#!/usr/bin/env python3

"""
Example command-line script to compute the tSZ power spectrum
and save or plot the result.
"""

import tszpower
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

# import tszpower.warmup

# 1) Define parameters
allpars = {
    'omega_b': 0.049*0.68**2,
    'omega_cdm':  0.261*0.68**2,
    'H0': 68.,
    'tau_reio': 0.0544,
    'ln10^{10}A_s': 3.035,
    # 'sigma8': 0.81,
    'n_s': 0.965,
    "cosmo_model": 0, # 1: use mnu-lcdm emulators; 0: use lcdm with fixed neutrino mass

    #-------------------
    "B": 1.4,
    'M_min': 1e14*0.68,
    'M_max': 1e16*0.68,
    # 'ell_min': 2,
    # 'ell_max': 8000,
    # 'dlogell': 0.1,
    'z_min': 0.01,
    'z_max': 3.0,
    'P0GNFW': 8.130,
    'c500': 1.156,
    'gammaGNFW': 0.3292,
    'alphaGNFW': 1.0620,
    'betaGNFW': 5.4807,
    'jax': 1
}

# 2) Setup

tszpower.classy_sz.set(allpars)
# tszpower.classy_sz.compute_class_szfast()
tszpower.warmup()


# print(tszpower.classy_sz.get_sigma8_and_der(params_values_dict=allpars))
# Warm-up call (triggers JIT compilation)
# _ = tszpower.compute_integral(params_values_dict=allpars)

# 3) Compute power spectrum
start_time = time.time()
ell = tszpower.get_ell_range()
C_ell = tszpower.compute_integral(params_values_dict=allpars)
end_time = time.time()
print(f"compute_integral took {end_time - start_time:.4f} seconds")



start_time = time.time()
# ell, T_ell_ellprime = tszpower.compute_trispectrum(params_values_dict=allpars)
ell_arr, M, M_G= tszpower.compute_tsz_covariance(params_values_dict=allpars, noise_ell=None, f_sky=1.0)
end_time = time.time()
plt.imshow(M)
plt.colorbar()
plt.savefig("tsz_covariance_matrix.png")
# Now you can, for example, slice out the diagonal to get error bars:
diag_vals = jnp.diag(M)   # shape (n_ell,)
# print(diag_vals_G)
sigma = jnp.sqrt(diag_vals)
sigma = ell*(ell+1)*sigma/(2*jnp.pi)
D_ell = ell*(ell+1)*C_ell/(2*jnp.pi)
# print(D_ell)
# 4) Plot
label_size = 17
title_size = 18
legend_size = 13
handle_length = 1.5

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.tick_params(axis='x', which='both', length=5, direction='in', pad=10)
ax.tick_params(axis='y', which='both', length=5, direction='in', pad=5)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.setp(ax.get_xticklabels(), fontsize=label_size)
plt.setp(ax.get_yticklabels(), fontsize=label_size, rotation='horizontal')

ax.grid(visible=True, which='both', alpha=0.2, linestyle='--')

ax.set_xlabel(r'$\ell$', size=title_size)
ax.set_ylabel(r'$D_\ell^{yy}$', size=title_size)
ax.set_xscale('log')
ax.set_yscale('log')

ax.errorbar(ell, D_ell*1e12, yerr=sigma*1e12, fmt='.-',
            label=r'$D_\ell^{yy}$ (with errors)', color='b')

ax.legend(fontsize=legend_size, handlelength=handle_length)
ax.set_title('tSZ Power Spectrum and Diagonal Errors', fontsize=title_size)
# ax.set_xlim(100, 1e4)
# ax.set_ylim(6e-2, 9e-1)
plt.savefig("tsz_power_spectrum_with_error.png")

plt.show()

