import torch
import math
from botorch.utils.sampling import draw_sobol_samples
import numpy as np
from scipy.stats import norm


def generate_initial_data(problem, init_size):
    '''
    :param problem: TestProblem
    :param init_size: number of initial evaluated data.
    :return: initial 'init_size' x 'dim' train_X, 'init_size' x 'dim' train_Y, train_Noise
    '''
    X = draw_sobol_samples(bounds=problem.bounds[:, :(problem.dim-problem.contexts_dim)], n=init_size, q=1).squeeze(1)
    Y, contexts = problem(X)
    return X, Y, contexts

def update_edf(contexts, discretization):
    '''
    :param contexts: 'N x dim' observed contexts.
    :param discretization: 'N_discrete x dim' discretized contexts.
    :return: The approximated edf 'N_discrete x 1'.
    '''
    omega = torch.zeros(discretization.shape[0]).to(device=discretization.device)
    for context in contexts:
        closest_j = 0
        min_dist = 10000.0
        for j, disc in enumerate(discretization):
            # Choose the closest one.
            dist = torch.norm(context-disc)
            if dist < min_dist:
                min_dist = dist
                closest_j = j
        omega[closest_j] += 1
    omega/=contexts.shape[0]
    return omega

def get_kernel_matrix(model, before_dim, contexts):
    '''
    :param model: GP model.
    :param before_dim: the dimension before the contexts. ( Or, dimension of decision variables)
    :param contexts: 'num_discretized_contexts x contexts_dim' discretized contexts.
    :return: 'num_discretized_contexts x num_discretized_contexts' matrix.
    '''
    num_contexts = contexts.shape[0]
    X = torch.zeros((num_contexts, before_dim)).to(device=contexts.device)
    X_contexts = torch.cat((X, contexts), dim=1).to(device=contexts.device)
    with torch.no_grad():
        covar_module = model.covar_module
        covar = covar_module(X_contexts)
        kernel_matrix = covar.to_dense()
    return kernel_matrix

def sample_kde(contexts, N, bounds):
    '''
    :param bounds: bounds of contexual variable.
    :param contexts: 'n x d' - dim contexts.
    :param N: Number of KDE samples.
    :return: 'N' Kernel Density Estimation samples of 'd'-dimensional noise. 'N' x 'd' tensors.
    '''
    contexts = contexts.detach().numpy()
    context_dim = contexts.shape[1]
    h = math.pow(4/(2+context_dim), 1/(4+context_dim)) * math.pow(contexts.shape[0], -1/(4+context_dim))
    std = np.std(contexts, axis=0)
    h = h*std
    h = h.reshape(1,-1)
    # Rule of thumb bandwidth
    # 'd' dimensional bandwidth
    weights = np.ones(contexts.shape[0])/contexts.shape[0]
    mixture_indices = np.random.choice(len(weights), size=N, p=weights)
    mean_samp = contexts[mixture_indices]
    kde_samples = norm.rvs(size=(N,context_dim)) * h + mean_samp
    kde_samples = torch.tensor(kde_samples)
    kde_samples = torch.clamp(kde_samples, bounds[0], bounds[1])
    return kde_samples
