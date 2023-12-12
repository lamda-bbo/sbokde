import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
import numpy as np
from torch.quasirandom import SobolEngine
import cvxpy as cp


class KDE_UCB(AnalyticAcquisitionFunction):
    def __init__(
            self,
            model,
            beta,
            kde_samples,
            posterior_transform=None,
            maximize=True,
            **kwargs,
    ) -> None:
        """
        :param model: GP model.
        :param beta: beta_t
        :param kde_samples: 'N_samples x d' samples from Kernel Density Estimation.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("kde_samples", kde_samples)
        self.maximize = maximize
        self.N_samples = self.kde_samples.shape[0]

    def forward(self, X):
        """
        :param X: 'batch_size x 1 x dim' decision variables.
        :return: the mean value under the kde samples for each x in X.
        """
        X = X.squeeze(dim=1)
        dim = X.shape[1]
        batch_size = X.shape[0]
        # This is 'batch x d'
        X_tmp = torch.tile(X, (1, self.N_samples))
        # This is 'batch x d*N_samples'
        X_tmp = torch.reshape(X_tmp, (batch_size, self.N_samples, dim))
        # This is 'batch x N_samples x dim'
        contexts_dim = self.kde_samples.shape[1]
        reshaped_kde_samples = self.kde_samples.reshape(1, self.N_samples, contexts_dim)
        # This is '1 x N_samples x contexts_dim'
        Contexts = torch.tile(reshaped_kde_samples, (batch_size, 1, 1))
        # This is 'batch x N_samples x contexts_dim'
        X_Contexts = torch.cat((X_tmp, Contexts), dim=-1)
        # This is 'batch x N_samples x (dim+contexts_dim)
        X_Contexts = X_Contexts.reshape(-1, dim + contexts_dim)
        # This is 'batch * N_samples x (dim+contexts_dim)
        posterior = self.model.posterior(X_Contexts)
        mean = posterior.mean
        var = posterior.variance
        ucb = mean + self.beta * var.sqrt()
        # This is '(batch * N_samples) x 1'
        ucb = ucb.reshape(batch_size, self.N_samples)
        expected_ucb = torch.mean(ucb, dim=1)
        return expected_ucb


class Stable_UCB(AnalyticAcquisitionFunction):
    def __init__(
            self,
            model,
            beta,
            contexts_observed,
            posterior_transform=None,
            maximize=True,
            **kwargs,
    ) -> None:
        """
        :param model: GP model.
        :param model_for_minimization: GP model for calculating inner minimization problem.(呃这个好像在这里不用)
        :param beta: beta_t
        :param contexts_observed: 'i x d' observed contexts.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))
        contexts_mean = contexts_observed.mean(dim=0, keepdim=True)
        contexts_std = contexts_observed.std(dim=0, keepdim=True)
        self.register_buffer("contexts_bound", torch.cat((contexts_mean-contexts_std,
                                                          contexts_mean+contexts_std), dim=0))
        #the bound is set as [mean ± 1 std].
        self.maximize = maximize

    def forward(self, X):
        """
        :param X: 'batch_size x 1 x dim' decision variables.
        :return: the min value under 1024 sobol samples for each x in X.
        """
        X = X.squeeze(dim=1)
        # This is 'batch x d'
        sobol_eng = torch.quasirandom.SobolEngine(dimension=self.contexts_bound.shape[1], scramble=True)
        N_samples = 1024
        contexts = (sobol_eng.draw(N_samples).to(device=X.device))*(self.contexts_bound[0]-self.contexts_bound[1]) + self.contexts_bound[0]
        dim = X.shape[1]
        batch_size = X.shape[0]
        # This is 'batch x d'
        X_tmp = torch.tile(X, (1, N_samples))
        # This is 'batch x d*N_samples'
        X_tmp = torch.reshape(X_tmp, (batch_size, N_samples, dim))
        contexts_dim = self.contexts_bound.shape[1]
        reshaped_contexts = contexts.reshape(1, N_samples, contexts_dim)
        Contexts = torch.tile(reshaped_contexts, (batch_size, 1, 1))
        # This is 'batch x N_samples x contexts_dim'
        X_Contexts = torch.cat((X_tmp, Contexts), dim=-1)
        # This is 'batch x N_samples x (dim+contexts_dim)
        X_Contexts = X_Contexts.reshape(-1, dim+contexts_dim)
        posterior = self.model.posterior(X_Contexts)
        mean = posterior.mean
        var = posterior.variance
        ucb = mean + self.beta * var.sqrt()
        # This is '(batch * N_samples) x 1'
        ucb = ucb.reshape(batch_size, N_samples)
        min_ucb = torch.min(ucb, dim=1)[0]
        return min_ucb


class MMD_UCB(AnalyticAcquisitionFunction):
    def __init__(
            self,
            model,
            beta,
            distribution,
            discretized_contexts,
            kernel_matrix,
            dist,
            posterior_transform=None,
            maximize=True,
            **kwargs,
    ) -> None:
        """
        :param model: GP model.
        :param beta: beta_t
        :param distribution: 'N_samples' dim of The empirical distribution.
        :param discretized_contexts: The contexts after discretizaiton.
        :param kernel_matrix: The kernel matrix for MMD distance.
        :param dist: A scalar. The MMD distance/radius of the DRO ball.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.dist = dist
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("discretized_contexts", discretized_contexts)
        self.maximize = maximize
        self.distribution = distribution.detach().numpy().reshape(-1, )
        kernel_matrix = kernel_matrix.detach().numpy()
        self.kernel_matrix = (kernel_matrix + kernel_matrix.T) / 2
        self.kernel_matrix = cp.psd_wrap(self.kernel_matrix)

    def _calculating_cvx(self, ucb):
        """
        The inner minimization problem is min_ p^T ucb_x(c) s.t. sum_p=1, 0<=p<=1, p^T (M) p_t <= dist**2
        :param ucb: 'N_samples' dim tensor of ucb_x(c).
        :return: value of inner minimization problem.
        """
        ucb_np = ucb.detach().numpy().reshape(-1, )
        p = cp.Variable(ucb_np.shape[0])
        obj = cp.Minimize(ucb_np @ p)
        constraints = [p >= 0., cp.sum(p) == 1,
                       cp.quad_form(p - self.distribution, self.kernel_matrix) <= self.dist ** 2]
        problem = cp.Problem(obj, constraints)
        problem.solve(solver='SCS')
        if p.value is None:
            p = torch.tensor(self.distribution, dtype=ucb.dtype)
            print('p.value is None.')
        else:
            p = torch.tensor(p.value, dtype=ucb.dtype)
        return p @ ucb

    def forward(self, X):
        """
        :param X: 'batch_size x 1 x dim' decision variables.
        :return: The MMD DRO (worst expectation of distribution under the MMD ball) of the UCB value for each x in X.
        """
        contexts = self.discretized_contexts
        X = X.squeeze(dim=1)
        N_samples, contexts_dim = contexts.shape
        batch_size, dim = X.shape
        X_tmp = torch.tile(X, (1, N_samples))
        # This is 'batch x d*N_samples'
        X_tmp = torch.reshape(X_tmp, (batch_size, N_samples, dim))
        reshaped_contexts = contexts.reshape(1, N_samples, contexts_dim)
        Contexts = torch.tile(reshaped_contexts, (batch_size, 1, 1))
        # This is 'batch x N_samples x contexts_dim'
        X_Contexts = torch.cat((X_tmp, Contexts), dim=-1)
        # This is 'batch x N_samples x (dim+contexts_dim)
        X_Contexts = X_Contexts.reshape(-1, dim + contexts_dim)
        posterior = self.model.posterior(X_Contexts)
        mean = posterior.mean
        var = posterior.variance
        ucb = mean + self.beta * var.sqrt()
        ucb = ucb.reshape(batch_size, N_samples)
        # This is '(batch) x (N_samples)'
        res = []
        for ucb_x in ucb:
            res.append(self._calculating_cvx(ucb_x))
        return torch.stack(res)


class MMD_Minimax_Approx_UCB(AnalyticAcquisitionFunction):
    def __init__(
            self,
            model,
            beta,
            distribution,
            discretized_contexts,
            kernel_matrix,
            dist,
            posterior_transform=None,
            maximize=True,
            **kwargs,
    ) -> None:
        """
        The parameters are all the same as MMD.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.dist = dist
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("discretized_contexts", discretized_contexts)
        self.register_buffer("distribution", distribution.reshape(1, -1))
        self.maximize = maximize
        self.kernel_matrix = (kernel_matrix + kernel_matrix.T) / 2

    def forward(self, X):
        """
        :param X: 'batch_size x 1 x dim' decision variables.
        :return: The MMD DRO (worst expectation of distribution under the MMD ball) of the UCB value for each x in X using Minimax Approximation.
        """
        contexts = self.discretized_contexts
        X = X.squeeze(dim=1)
        N_samples, contexts_dim = contexts.shape
        batch_size, dim = X.shape
        X_tmp = torch.tile(X, (1, N_samples))
        # This is 'batch x d*N_samples'
        X_tmp = torch.reshape(X_tmp, (batch_size, N_samples, dim))
        reshaped_contexts = contexts.reshape(1, N_samples, contexts_dim)
        Contexts = torch.tile(reshaped_contexts, (batch_size, 1, 1))
        # This is 'batch x N_samples x contexts_dim'
        X_Contexts = torch.cat((X_tmp, Contexts), dim=-1)
        # This is 'batch x N_samples x (dim+contexts_dim)
        X_Contexts = X_Contexts.reshape(-1, dim + contexts_dim)
        posterior = self.model.posterior(X_Contexts)
        mean = posterior.mean
        var = posterior.variance
        ucb = mean + self.beta * var.sqrt()
        ucb = ucb.reshape(batch_size, N_samples)
        # This is '(batch) x (N_samples)'
        worst_g, min_index = torch.min(ucb, dim=1, keepdim=True)
        # worst_g: '(batch) x 1'
        # min_index: '(batch) x 1
        expectation_g = torch.mean(ucb, dim=1, keepdim=True)
        # This is '(batch) x 1'
        # Calculate Minimax Approx
        S_MMD_g = torch.sqrt(torch.sum(torch.pow(ucb, 2), dim=1, keepdim=True) - (expectation_g ** 2) * N_samples)
        worst_distribution = torch.zeros_like(ucb)
        # This is 'batch x N_samples'
        col_index = torch.arange(batch_size).to(X.device) * N_samples + min_index.reshape(-1, )
        worst_distribution.view(-1)[col_index] += 1.0
        epsilon_d_star_right = (self.kernel_matrix @ (worst_distribution - self.distribution).T).T
        # This is 'batch x N_samples'
        epsilon_d_star = torch.sqrt(torch.sum(torch.mul(worst_distribution - self.distribution, epsilon_d_star_right),
                                              dim=1, keepdim=True))
        # This is 'batch x 1'
        epsilon_d_prime = (worst_g - expectation_g) / S_MMD_g
        tau_d = (worst_g - expectation_g) / epsilon_d_star
        # This is 'batch x 1'
        res = torch.zeros_like(epsilon_d_prime)
        case1 = self.dist < epsilon_d_prime
        case2 = torch.logical_and(epsilon_d_prime <= self.dist, self.dist <= epsilon_d_star)
        case3 = self.dist > epsilon_d_star
        if torch.any(case1):
            res[case1] += (expectation_g[case1] + self.dist * (tau_d[case1] + S_MMD_g[case1]) / 2)
        if torch.any(case2):
            res[case2] += ((expectation_g[case2] + self.dist * tau_d[case2] + worst_g[case2]) / 2)
        else:
            res[case3] += worst_g[case3]
        return res.squeeze(1)


class KDE_DRBO_UCB(AnalyticAcquisitionFunction):
    def __init__(
            self,
            model,
            beta,
            distance,
            kde_samples,
            context_bound,
            posterior_transform=None,
            maximize=True,
            **kwargs,
    ) -> None:
        """
        :param model: GP model.
        :param beta: beta_t.
        :param distance: The distance/radius of Total Variation ball.
        :param kde_samples: 'N_samples x d' -dim tensors. samples from Kernel Density Estimation.
        :param context_bound: '2 x d' -dim tensors. The bound of the contexts.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("kde_samples", kde_samples)
        self.register_buffer("contexts_bound", context_bound)
        self.maximize = maximize
        self.N_samples = self.kde_samples.shape[0]
        self.distance = distance

    def _calculate_min_ucb(self, X):
        """
        Similar to Stable. Return the min value under the contexts_bound using 1024 samples.
        """
        # This is 'batch x d'
        sobol_eng = torch.quasirandom.SobolEngine(dimension=self.contexts_bound.shape[1], scramble=True)
        N_samples = 1024
        contexts = (sobol_eng.draw(N_samples).to(device=X.device))*(self.contexts_bound[0]-self.contexts_bound[1]) + self.contexts_bound[0]
        dim = X.shape[1]
        batch_size = X.shape[0]
        # This is 'batch x d'
        X_tmp = torch.tile(X, (1, N_samples))
        # This is 'batch x d*N_samples'
        X_tmp = torch.reshape(X_tmp, (batch_size, N_samples, dim))
        contexts_dim = self.contexts_bound.shape[1]
        reshaped_contexts = contexts.reshape(1, N_samples, contexts_dim)
        Contexts = torch.tile(reshaped_contexts, (batch_size, 1, 1))
        # This is 'batch x N_samples x contexts_dim'
        X_Contexts = torch.cat((X_tmp, Contexts), dim=-1)
        # This is 'batch x N_samples x (dim+contexts_dim)
        X_Contexts = X_Contexts.reshape(-1, dim+contexts_dim)
        posterior = self.model.posterior(X_Contexts)
        mean = posterior.mean
        var = posterior.variance
        ucb = mean + self.beta * var.sqrt()
        # This is '(batch * N_samples) x 1'
        ucb = ucb.reshape(batch_size, N_samples)
        min_ucb = torch.min(ucb, dim=1)[0]
        return min_ucb

    def _calculate_cvx(self, x, min_ucb):
        '''
        :param x:  'd'-dim tensor.
        :param min_ucb:   min_{c} of ucb(x,c)
        :return: Value of the two dimensional convex optimization.
        Optimization of \max_{\beta, \alpha>=0} {-\beta-\epsilon_t\alpha+E_{c}{ \min{ucb(x,c)+\beta,\alpha}}}
        '''
        x = x.reshape(1, -1)
        x = torch.tile(x, (self.N_samples, 1))
        x_contexts = torch.cat((x, self.kde_samples), dim=1)
        posterior = self.model.posterior(x_contexts)
        mean = posterior.mean
        var = posterior.variance
        ucb = mean + self.beta * var.sqrt()
        # This is 'N_samples x 1'
        ucb_np = ucb.detach().numpy().reshape(-1, )
        min_ucb = min_ucb.detach().numpy()
        alpha = cp.Variable()
        beta = cp.Variable()
        mean_samples = cp.sum(cp.minimum(ucb_np+beta,alpha))/self.N_samples
        obj = cp.Maximize(-beta-alpha*self.distance+mean_samples)
        constraints = [alpha + beta >= -min_ucb, alpha >= 0]
        problem = cp.Problem(obj, constraints)
        solution = problem.solve()
        alpha, beta = torch.tensor(alpha.value), torch.tensor(beta.value)
        return -beta-self.distance*alpha+torch.mean(torch.min(ucb+beta, alpha), dim=0)

    def forward(self, X):
        #先求min ucb
        X = X.squeeze(dim=1)
        min_ucbs = self._calculate_min_ucb(X)
        res = []
        for x, min_ucb in zip(X, min_ucbs):
            res.append(self._calculate_cvx(x, min_ucb))
        return torch.stack(res).reshape(-1,)