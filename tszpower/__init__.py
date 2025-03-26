from .config import classy_sz
from .initialise import *
from .profiles import *
from .massfuncs import *
from .tsz import *
from .theory import *
from .tsz_sims import *
from .utils import *
from .likelihood import TSZPowerLikelihood
from .power_spectra import *
from .tsz_sbi_inference import *
# from .tsz_sbi import *


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