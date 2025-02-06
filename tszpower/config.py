from classy_sz import Class as Class_sz

# Define default cosmology parameters
params = {
    'omega_b': 0.02242,
    'omega_cdm': 0.1193,
    'H0': 67.66,
    'tau_reio': 0.0561,
    'ln10^{10}A_s': 3.047,
    'n_s': 0.9665,
    'M_min': 1e11,
    'M_max': 5e15,
    # 'ell_min': 2,
    # 'ell_max': 8000,
    # 'dlogell': 0.1,
    'z_min': 1e-6,
    'z_max': 6.0,
    'P0GNFW': 8.130,
    'c500': 1.156,
    'gammaGNFW': 0.3292,
    'alphaGNFW': 1.0620,
    'betaGNFW': 5.4807,
    'B': 1.0,
    'cosmo_model': 1, # use mnu-lcdm emulators
    'jax': 1
}

# Initialize classy_sz
classy_sz = Class_sz()
classy_sz.set(params)  # Set parameters
classy_sz.compute_class_szfast()  # Required before calling other functions
