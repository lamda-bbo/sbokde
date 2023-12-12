import torch
import numpy as np
import math
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch.quasirandom import SobolEngine
from .portfolio_surrogate import PortfolioSurrogate
import pandas as pd
import pickle as pkl
from torch.distributions.cauchy import Cauchy
from scipy.stats import burr12


class Ackley(SyntheticTestFunction):

    # The last dimension follows a Gaussian Distribution.
    _check_grad_at_opt = False

    def __init__(
        self,
        mu=0.5,
        sigma=0.15,
        dim=3,
        noise_std=None,
        negate=False,
        bounds=None,
    ):
        self.dim = dim
        self.contexts_dim = 1
        #self._bounds = [(-32.768, 32.768) for _ in range(self.dim)]
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=False, bounds=bounds)
        self.a = 20
        self.b = 0.2
        self.c = 2 * math.pi
        self.negate_ = negate
        self.mu = mu
        self.sigma = sigma
        self.max_stochastic = 10.9523
        if negate:
            self.max_stochastic = -10.9523
        self.can_calculate_stochastic = True

    def evaluate_true(self, X):
        batch_size = X.shape[0]
        context = torch.normal(mean=self.mu, std=self.sigma, size=(batch_size, 1))
        context_clamp = torch.clamp(context, self.bounds[0,-1], self.bounds[1,-1])
        X = torch.cat((X, context_clamp), dim=1)
        X = X*65.536 - 32.768
        a, b, c = self.a, self.b, self.c
        part1 = -a * torch.exp(-b / math.sqrt(self.dim) * torch.norm(X, dim=-1))
        part2 = -(torch.exp(torch.mean(torch.cos(c * X), dim=-1)))
        f_X = part1 + part2 + a + math.e
        if self.negate_:
            f_X *= -1
        return f_X, context

    def evaluate_stochastic(self, X, sample_size=2097152):
        '''
        :param X: 1 x d Tensor.
        :param sample_size: QMC Sample Size for calculating stochastic value.
        :return:
        '''
        X = X.reshape(1, -1)
        engine = SobolEngine(1)
        sobol_samples = engine.draw(sample_size)
        std_samples = torch.erfinv(2 * sobol_samples - 1) * math.sqrt(2)
        context = self.mu + self.sigma * std_samples
        context_clamp = torch.clamp(context, self.bounds[0,-1], self.bounds[1,-1])
        X = torch.tile(X, (sample_size, 1))
        X = torch.cat((X, context_clamp), dim=1)
        X = X*65.536 - 32.768
        a, b, c = self.a, self.b, self.c
        part1 = -a * torch.exp(-b / math.sqrt(self.dim) * torch.norm(X, dim=-1))
        part2 = -(torch.exp(torch.mean(torch.cos(c * X), dim=-1)))
        f_X = part1 + part2 + a + math.e
        if self.negate_:
            f_X *= -1
        return f_X.mean()


class Hartmann(SyntheticTestFunction):
    def __init__(
        self,
        mu=0.5,
        sigma=0.1,
        dim=6,
        noise_std=None,
        negate=False,
        bounds=None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        if dim not in (6,):
            raise ValueError(f"Hartmann with dim {dim} not defined")
        self.dim = dim
        self.contexts_dim = 1
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=False, bounds=bounds)
        self.register_buffer("ALPHA", torch.tensor([1.0, 1.2, 3.0, 3.2]))
        self.can_calculate_stochastic = True
        if dim == 3:
            A = [[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]]
            P = [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        elif dim == 4:
            A = [
                [10, 3, 17, 3.5],
                [0.05, 10, 17, 0.1],
                [3, 3.5, 1.7, 10],
                [17, 8, 0.05, 10],
            ]
            P = [
                [1312, 1696, 5569, 124],
                [2329, 4135, 8307, 3736],
                [2348, 1451, 3522, 2883],
                [4047, 8828, 8732, 5743],
            ]
        elif dim == 6:
            A = [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
            P = [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        self.register_buffer("A", torch.tensor(A, dtype=torch.float))
        self.register_buffer("P", torch.tensor(P, dtype=torch.float))
        self.negate_ = negate
        self.mu = mu
        self.sigma = sigma
        self.max_stochastic = -2.6127565
        if negate:
            self.max_stochastic = 2.6127565


    def evaluate_true(self, X):
        self.to(device=X.device, dtype=X.dtype)
        context = torch.normal(mean=self.mu, std=self.sigma, size=(X.shape[0], 1))
        context_clamp = torch.clamp(context, self.bounds[0,-1], self.bounds[1,-1])
        X = torch.cat((X, context_clamp), dim=1)
        inner_sum = torch.sum(self.A * (X.unsqueeze(-2) - 0.0001 * self.P) ** 2, dim=-1)
        H = (torch.sum(self.ALPHA * torch.exp(-inner_sum), dim=-1))
        f_X = -H
        if self.negate_:
            f_X *= -1
        return f_X, context


    def evaluate_stochastic(self, X, sample_size=2097152):
        '''
        :param X: 1 x d Tensor.
        :param sample_size: QMC Sample Size for calculating stochastic value.
        :return:
        '''
        X = X.reshape(1, -1)
        engine = SobolEngine(1)
        sobol_samples = engine.draw(sample_size)
        std_samples = torch.erfinv(2 * sobol_samples - 1) * math.sqrt(2)
        context = self.mu + self.sigma * std_samples
        context_clamp = torch.clamp(context, self.bounds[0,-1], self.bounds[1,-1])
        X = torch.tile(X, (sample_size, 1))
        X = torch.cat((X, context_clamp), dim=1)
        inner_sum = torch.sum(self.A * (X.unsqueeze(-2) - 0.0001 * self.P) ** 2, dim=-1)
        H = (torch.sum(self.ALPHA * torch.exp(-inner_sum), dim=-1))
        f_X = -H
        if self.negate_:
            f_X *= -1
        return f_X.mean()

class Modified_Branin(SyntheticTestFunction):

    # The last dimension follows a Gaussian Distribution.
    _check_grad_at_opt = False

    def __init__(
        self,
        mu=0.5,
        sigma=0.1,
        dim=4,
        noise_std=None,
        negate=False,
        bounds=None,
    ):
        self.dim = 4
        self.contexts_dim = 2
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=False, bounds=bounds)
        self.negate_ = negate
        self.mu = mu
        self.sigma = sigma
        self.max_stochastic = 9.603
        if negate:
            self.max_stochastic = -9.603
        self.can_calculate_stochastic = True

    def evaluate_true(self, X):
        context = torch.normal(mean=self.mu, std=self.sigma, size=(X.shape[0], 2))
        context_clamp = torch.clamp(context, self.bounds[0, [1, 2]], self.bounds[1, [1, 2]])
        X = torch.cat((X[:,0].unsqueeze(dim=1), context_clamp[:,0].unsqueeze(dim=1),
                       context_clamp[:,1].unsqueeze(dim=1), X[:,1].unsqueeze(dim=1)), dim=1)
        u1, v1, u2, v2 = X[:, 0]*15-5, X[:, 1]*15, X[:, 2]*15-5, X[:, 3]*15,
        y1 = Branin(u1, v1)
        y2 = Branin(u2, v2)
        f_X = torch.sqrt(y1*y2)
        if self.negate_:
            f_X *= -1
        return f_X, context

    def evaluate_stochastic(self, X, sample_size=2097152):
        '''
        :param X: 1 x d Tensor.
        :param sample_size: QMC Sample Size for calculating stochastic value.
        :return:
        '''
        X = X.reshape(1, -1)
        engine = SobolEngine(2)
        sobol_samples = engine.draw(sample_size)
        std_samples = torch.erfinv(2 * sobol_samples - 1) * math.sqrt(2)
        context = self.mu + self.sigma * std_samples
        context_clamp = torch.clamp(context, self.bounds[0, [1, 2]], self.bounds[1, [1, 2]])
        X = torch.tile(X, (sample_size, 1))
        X = torch.cat((X[:,0].unsqueeze(dim=1), context_clamp[:,0].unsqueeze(dim=1),
                       context_clamp[:,1].unsqueeze(dim=1), X[:,1].unsqueeze(dim=1)), dim=1)
        u1, v1, u2, v2 = X[:, 0]*15-5, X[:, 1]*15, X[:, 2]*15-5, X[:, 3]*15,
        y1 = Branin(u1, v1)
        y2 = Branin(u2, v2)
        f_X = torch.sqrt(y1*y2)
        if self.negate_:
            f_X *= -1
        return f_X.mean()


def Branin(u, v):
    return torch.pow(v-5.1/(4*math.pi*math.pi) * u * u + 5.0/math.pi*u - 6.0, 2) +\
        10.0*(1-1/(8*math.pi))*torch.cos(u) + 10.0


class portfolio_optimization(SyntheticTestFunction):
    
    _check_grad_at_opt = False
    def __init__(
        self,
        mu=None,
        sigma=None,
        dim=5,
        noise_std=None,
        negate=False,
        bounds=None,
    ):
        self.dim = 5
        self.contexts_dim = 2
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=False, bounds=bounds)
        self.portfolio_surrogate = PortfolioSurrogate()
        self.portfolio_surrogate.fit_model()
        self.negate_ = negate
        self.mu = mu
        self.sigma = sigma
        self.max_stochastic = -20.0
        self.can_calculate_stochastic = True
        if negate:
            self.max_stochastic = 20.0

    def evaluate_true(self, X):
        context = torch.rand(size=(X.shape[0], self.contexts_dim))
        f_X = self.portfolio_surrogate(torch.cat((X, context), dim=1))
        if self.negate_:
            f_X = f_X * (-1)
        return f_X.reshape(-1,), context

    def evaluate_stochastic(self, X, sample_size=65536):
        '''
        :param X: 1 x d Tensor.
        :param sample_size: QMC Sample Size for calculating stochastic value.
        :return:
        '''
        X = X.reshape(1, -1)
        engine = SobolEngine(2)
        sobol_samples = engine.draw(sample_size)
        context = sobol_samples
        X = torch.tile(X, (sample_size, 1))
        X = torch.cat((X, context), dim=1)
        f_X = self.portfolio_surrogate(X)
        if self.negate_:
            f_X = f_X * (-1)
        return f_X.mean()


class portfolio_normal_optimization(SyntheticTestFunction):
    _check_grad_at_opt = False

    def __init__(
            self,
            mu=0.5,
            sigma=0.1,
            dim=5,
            noise_std=None,
            negate=False,
            bounds=None,
    ):
        self.dim = 5
        self.contexts_dim = 2
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=False, bounds=bounds)
        self.portfolio_surrogate = PortfolioSurrogate()
        self.portfolio_surrogate.fit_model()
        self.negate_ = negate
        self.mu = mu
        self.sigma = sigma
        self.max_stochastic = -22.0
        self.can_calculate_stochastic = True
        if negate:
            self.max_stochastic = 22.0

    def evaluate_true(self, X):
        context = torch.normal(mean=self.mu, std=self.sigma, size=(X.shape[0], 2))
        context = torch.clamp(context, self.bounds[0, [3, 4]], self.bounds[1, [3, 4]])
        f_X = self.portfolio_surrogate(torch.cat((X, context), dim=1))
        if self.negate_:
            f_X = f_X * (-1)
        return f_X.reshape(-1, ), context

    def evaluate_stochastic(self, X, sample_size=65536):
        '''
        :param X: 1 x d Tensor.
        :param sample_size: QMC Sample Size for calculating stochastic value.
        :return:
        '''
        X = X.reshape(1, -1)
        engine = SobolEngine(2)
        sobol_samples = engine.draw(sample_size)
        std_samples = torch.erfinv(2 * sobol_samples - 1) * math.sqrt(2)
        context = self.mu + self.sigma * std_samples
        context = torch.clamp(context, self.bounds[0, [3, 4]], self.bounds[1, [3, 4]])
        X = torch.tile(X, (sample_size, 1))
        X = torch.cat((X, context), dim=1)
        f_X = self.portfolio_surrogate(X)
        if self.negate_:
            f_X = f_X * (-1)
        return f_X.mean()


class Hartmann_complicated(SyntheticTestFunction):
    def __init__(
        self,
        mu=None,
        sigma=None,
        dim=6,
        noise_std=None,
        negate=False,
        bounds=None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        if dim not in (6,):
            raise ValueError(f"Hartmann with dim {dim} not defined")
        self.dim = dim
        self.contexts_dim = 1
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=False, bounds=bounds)
        self.register_buffer("ALPHA", torch.tensor([1.0, 1.2, 3.0, 3.2]))
        '''
        with open('kde_samples_for_test.pkl', 'rb') as f:
            context_qmc = pkl.load(f)
        context_qmc = torch.tensor(context_qmc).reshape(-1, 1)
        self.context_clamp = torch.clamp(context_qmc, self.bounds[0,-1], self.bounds[1,-1])
        '''
        self.can_calculate_stochastic = False
        if dim == 3:
            A = [[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]]
            P = [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        elif dim == 4:
            A = [
                [10, 3, 17, 3.5],
                [0.05, 10, 17, 0.1],
                [3, 3.5, 1.7, 10],
                [17, 8, 0.05, 10],
            ]
            P = [
                [1312, 1696, 5569, 124],
                [2329, 4135, 8307, 3736],
                [2348, 1451, 3522, 2883],
                [4047, 8828, 8732, 5743],
            ]
        elif dim == 6:
            A = [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
            P = [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        self.register_buffer("A", torch.tensor(A, dtype=torch.float))
        self.register_buffer("P", torch.tensor(P, dtype=torch.float))
        self.negate_ = negate
        self.mu = mu
        self.sigma = sigma
        self.max_stochastic = -2.1
        if negate:
            self.max_stochastic = 2.1
        self.center = np.array([0.1, 0.3, 0.4, 0.5, 0.7, 0.8])
        self.std = np.array([0.02, 0.075, 0.1, 0.1, 0.075, 0.03])

    def evaluate_true(self, X):
        self.to(device=X.device, dtype=X.dtype)
        context = torch.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            k = torch.randint(low=0, high=8, size=(1,)).item()
            if k == 7:
                m = Cauchy(torch.tensor([0.2]), torch.tensor(0.02))
                context[i] = m.sample().item()
            elif k == 6:
                m = Cauchy(torch.tensor([0.8]), torch.tensor(0.02))
                context[i] = m.sample().item()
            else:
                context[i] = torch.normal(mean=self.center[k], std=self.std[k], size=(1,)).item()
        context_clamp = torch.clamp(context, self.bounds[0,-1], self.bounds[1,-1])
        X = torch.cat((X, context_clamp), dim=1)
        inner_sum = torch.sum(self.A * (X.unsqueeze(-2) - 0.0001 * self.P) ** 2, dim=-1)
        H = (torch.sum(self.ALPHA * torch.exp(-inner_sum), dim=-1))
        f_X = -H
        if self.negate_:
            f_X *= -1
        return f_X, context


    def evaluate_stochastic(self, X, sample_size=65536):
        '''
        :param X: 1 x d Tensor.
        :param sample_size: QMC Sample Size for calculating stochastic value.
        :return:
        '''
        X = X.reshape(1, -1)
        context_clamp = self.context_clamp
        X = torch.tile(X, (sample_size, 1))
        X = torch.cat((X, context_clamp), dim=1)
        inner_sum = torch.sum(self.A * (X.unsqueeze(-2) - 0.0001 * self.P) ** 2, dim=-1)
        H = (torch.sum(self.ALPHA * torch.exp(-inner_sum), dim=-1))
        f_X = -H
        if self.negate_:
            f_X *= -1
        return f_X.mean()


class Continuous_Vendor(SyntheticTestFunction):
    _check_grad_at_opt = False

    def __init__(
        self,
        mu=2,
        sigma=20,
        dim=2,
        noise_std=None,
        negate=False,
        bounds=None,
    ):
        self.dim = dim
        self.contexts_dim = 1
        #self._bounds = [(0.0, 10.0) for _ in range(self.dim)]
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=False, bounds=bounds)
        self.negate_ = negate
        self.mu = mu
        self.sigma = sigma
        self.c = mu
        self.k = sigma
        self.can_calculate_stochastic = True
        self.max_stochastic = -0.464
        self.cost = 5.0
        self.price = 9.0
        self.salvage_price = 1.0
        if negate:
            self.max_stochastic = 0.464

    def evaluate_true(self, X):
        batch_size = X.shape[0]
        context = burr12.rvs(self.c, self.k, size=batch_size)
        context = torch.tensor(context).reshape(-1, 1)
        context_clamp = torch.clamp(context, self.bounds[torch.tensor(0), (self.dim-self.contexts_dim):],
                                    self.bounds[1, (self.dim-self.contexts_dim):])
        f_X = self.price*torch.min(context_clamp,X) - self.cost*X + self.salvage_price*torch.max(torch.tensor(0),X-context_clamp)
        f_X = f_X*(-1)
        if self.negate_:
            f_X *= -1
        return f_X.reshape(-1,), context

    def evaluate_stochastic(self, X, sample_size=2097152):
        '''
        :param X: 1 x d Tensor.
        :param sample_size: QMC Sample Size for calculating stochastic value.
        :return:
        '''
        X = X.reshape(1, -1)
        engine = SobolEngine(dimension=1)
        sobol_samples = engine.draw(sample_size)
        context = torch.pow(torch.pow(1-sobol_samples, -1/20) - torch.tensor(1.0), 1/2)
        context_clamp = torch.clamp(context, self.bounds[0,(self.dim-self.contexts_dim):],
                                    self.bounds[1,(self.dim-self.contexts_dim):])
        f_X = self.price*torch.min(context_clamp,X) - self.cost*X + self.salvage_price*torch.max(torch.tensor(0),X-context_clamp)
        f_X = f_X*(-1)
        if self.negate_:
            f_X *= -1
        return f_X.mean()
