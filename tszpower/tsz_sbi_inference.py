import numpy as np
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from sbi.analysis import pairplot
from sbi.inference import NPE
from sbi.inference import NLE
from .tsz_sims import simulator


def draw_samples_from_uniform_prior(low = None, high = None, num_sims = 2000):

    prior = BoxUniform(low=low, high=high)
    theta = prior.sample((num_sims,))

    return theta

def draw_simulated_data_from_samples(theta, simulator_func):
    
    x = simulator_func(theta)

    return x

def check_prior_consistency(prior, simulator_func):

    processed_prior, num_parameters, prior_returns_numpy = process_prior(prior)
    processed_simulator = process_simulator(simulator_func, processed_prior, prior_returns_numpy)
    check_sbi_inputs(processed_simulator, processed_prior)

    return "Check prior is successful."

# Usage:
# prior = draw_uniform_prior(low=torch.tensor([...]), high=torch.tensor([...]))
# print(check_prior_consistency(prior, tsz_simulator))

def infer_npe(theta, x, prior = None, show_train_summary=True, num_sims = 2000,
            training_batch_size = 100,
            stop_after_epochs = 200, 
            learning_rate = 5e-5,
            num_posterior_samples = 100000, 
            true_data_path = "/home/lx256/tsz_project/tszpower/data/data_ps-ell-y2-erry2_total-planck-collab-15.txt"):

    inference = NPE(prior=prior)
    inference.append_simulations(theta, x).train(show_train_summary=show_train_summary,
                                                 training_batch_size=training_batch_size,
                                                 stop_after_epochs=stop_after_epochs, 
                                                 learning_rate=learning_rate)
    posterior = inference.build_posterior()


    # TODO: This only works for my own paper. Need to make it more general.
    data_file = np.loadtxt(true_data_path)
    x_o = data_file[:, 1]
    samples = posterior.sample((num_posterior_samples,), x=x_o)

    return samples.numpy()


def infer_nle_single_round(theta, x, prior = None, 
                           training_batch_size = 100,
                           show_train_summary=True, 
                           num_sims = 2000, 
                           num_rounds = 1, 
                           stop_after_epochs = 200, 
                           learning_rate = 5e-5,
              num_posterior_samples = 100000, true_data_path = "/home/lx256/tsz_project/tszpower/data/data_ps-ell-y2-erry2_total-planck-collab-15.txt"):

    inference = NLE(prior=prior)
    proposal = prior
    for round_idx in range(1, num_rounds + 1):
        print(f"Processing round {round_idx}")
        _ = inference.append_simulations(theta, x).train(show_train_summary=show_train_summary, stop_after_epochs=stop_after_epochs, learning_rate=learning_rate) 
        posterior = inference.build_posterior(mcmc_method="slice_np_vectorized",
                                              mcmc_parameters={"num_chains": 20,
                                                               "thin": 5})
        # TODO: This only works for my own paper. Need to make it more general.
        data_file = np.loadtxt(true_data_path)
        x_o = data_file[:, 1]

        proposal = posterior.set_default_x(x_o)
    
    samples = proposal.sample((num_posterior_samples,))

    return samples.numpy()

# In NLE including n_rounds, we need to update the proposal after each round.
def infer_nle_likelihood(low = None, high = None, prior = None, show_train_summary=True, 
            simulator_func = None,
            num_sims_round = 200, 
            training_batch_size = 20,
            num_rounds = 5, 
            stop_after_epochs = 200, 
            learning_rate = 5e-5,
            num_posterior_samples = 100000, 
            true_data_path = "/home/lx256/tsz_project/tszpower/data/data_ps-ell-y2-erry2_total-planck-collab-15.txt"):
    # TODO: This only works for my own paper. Need to make it more general.
    data_file = np.loadtxt(true_data_path)
    x_o = data_file[:, 1]

    inference = NLE(prior=prior)
    proposal = prior
    for round_idx in range(1, num_rounds + 1):
        print(f"Processing round {round_idx}")
        theta = proposal.sample((num_sims_round,))
        x = simulator_func(theta)

        _ = inference.append_simulations(theta, x).train(show_train_summary=show_train_summary, 
                                                        training_batch_size=training_batch_size,
                                                        stop_after_epochs=stop_after_epochs, 
                                                        learning_rate=learning_rate) 
        
        posterior = inference.build_posterior(mcmc_method="slice_np_vectorized",
                                                mcmc_parameters={"num_chains": 20,
                                                                 "thin": 5})
        proposal = posterior.set_default_x(x_o)

    samples = proposal.sample((num_posterior_samples,))

    return samples.numpy()


