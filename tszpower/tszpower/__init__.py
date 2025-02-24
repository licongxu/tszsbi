from .config import classy_sz
from .warmup import warmup  

warmup()

# Import and re-export functions/classes from submodules:
from .profiles import (
    gnfw_pressure_profile,
    window_function,
    hankel_integrand,
    mpc_per_h_to_cm,
    y_ell_prefactor,
    y_ell_complete,
    y_ell_interpolate
)

from .massfuncs import (
    MF_T08,
    get_hmf_grid,
    get_hmf_at_z_and_m
)

from .tsz import (
    dVdzdOmega,
    simpson,
    get_ell_range,
    get_integral_grid,
    compute_integral,
    get_integral_grid_trisp,
    compute_trispectrum,
    compute_tsz_covariance
)

from .theory import (
    compute_sz_power,
    compute_foreground_lkl
)

from .tsz_sbi import (
    ensure_array,
    get_batch_size,
    broadcast_to_batch,
    compute_Cl_yy_noiseless,
    compute_Nl_yy,
    compute_Cl_yy,
    compute_foreground,
    compute_Cl_yy_total
)

# tszpower/__init__.py

from .likelihood import TSZPowerLikelihood


def likelihood(allparams,
               data_directory="data",
               data_file="data_ps-ell-y2-erry2_total-planck-collab-15.txt",
               cov_file=None,
               fg_template_path="data/data_fg-ell-cib_rs_ir_cn-total-planck-collab-15.txt"):
    """
    Top-level function to compute the tSZ likelihood.
    
    Parameters
    ----------
    cosmology_params : dict
        Cosmological parameters (keys: "omega_b", "omega_cdm", "H0", "tau_reio", "ln10_10A_s", "n_s", "B").
    nuisance_params : dict
        Nuisance (foreground) parameters (keys: "A_cib", "A_ir", "A_rs").
    data_directory, data_file, cov_file : str, optional
        Data locations for the observed tSZ power spectrum.
    fg_template_path : str, optional
        Path to the foreground template file.
    
    Returns
    -------
    loglike : float
        The log-likelihood value.
    """
    lik = TSZPowerLikelihood(data_directory=data_directory,
                             data_file=data_file,
                             cov_file=cov_file)
    return lik.compute(allparams, fg_template_path=fg_template_path)
