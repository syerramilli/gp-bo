
import torch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition.analytic import _scaled_improvement, _ei_helper
from botorch.utils.transforms import t_batch_mode_transform
from botorch.sampling import SobolQMCNormalSampler
from .models import GPR
from .truncnorm_sample import sample_tmvnorm_sequential_qmc_torch

from typing import Union
from numbers import Number

class ExpectedImprovementWithPending(AnalyticAcquisitionFunction):
    """Expected improvement with pending evaulations

    This computes the expected improvement over the current best value given
    pending evaluations. This is done by drawing samples from the posterior
    over the pending evaluations, computing the expected improvement for each
    sample (with a possible update to the current best), and then averaging
    the expected improvements over the samples.

    Args:
        model: A fitted GPR model.
        best_f: The best function value observed so far (assumed noiseless).
        X_pending: A `q x d`-dim Tensor of `q` points with pending function evaluations.
        num_samples: The number of samples to draw from the posterior over the
            pending evaluations. Default: 64.
    """
    def __init__(
        self, model:GPR, best_f:Union[Number, torch.Tensor], 
        X_pending:torch.Tensor, num_samples:int=64
    ):
        with torch.no_grad():
            post_next_x = model.posterior(X_pending)
        
        sampler = SobolQMCNormalSampler(torch.Size([num_samples]))
        samples = sampler(post_next_x)

        new_model = model.condition_on_observations(
            X_pending.unsqueeze(0).repeat(num_samples, 1, 1).double(), samples.double()
        )

        super().__init__(model=new_model, posterior_transform=None)
        self.register_buffer(
            "best_f", 
            torch.maximum(torch.as_tensor(best_f), samples.max(-2).values).flatten()
        )
        self.register_buffer("num_samples", torch.as_tensor(num_samples))
        self.maximize = True
    
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        mean, sigma = self._mean_and_sigma(X.unsqueeze(1))
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        ei_batch = sigma * _ei_helper(u)
        return ei_batch.mean(dim=1)
    

class ExpectedImprovementWithPendingTrunc(AnalyticAcquisitionFunction):
    """Expected improvement with pending evaulations

    This computes the expected improvement over the current best value given
    pending evaluations. This is done by drawing samples from the posterior
    over the pending evaluations, computing the expected improvement for each
    sample (with a possible update to the current best), and then averaging
    the expected improvements over the samples.

    Args:
        model: A fitted GPR model.
        best_f: The best function value observed so far (assumed noiseless).
        X_pending: A `q x d`-dim Tensor of `q` points with pending function evaluations.
        num_samples: The number of samples to draw from the posterior over the
            pending evaluations. Default: 64.
    """
    def __init__(
        self, model:GPR, best_f:Union[Number, torch.Tensor], 
        X_pending:torch.Tensor, num_samples:int=64
    ):
        with torch.no_grad():
            post_next_x = model.posterior(X_pending)
        
        mu, Sigma = post_next_x.mvn.mean, post_next_x.mvn.covariance_matrix
        samples = sample_tmvnorm_sequential_qmc_torch(mu, Sigma, best_f, num_samples).unsqueeze(-1)

        new_model = model.condition_on_observations(
            X_pending.unsqueeze(0).repeat(num_samples, 1, 1).double(), samples
        )

        super().__init__(model=new_model, posterior_transform=None)
        self.register_buffer(
            "best_f", 
            torch.maximum(torch.as_tensor(best_f), samples.max(-2).values).flatten()
        )
        self.register_buffer("num_samples", torch.as_tensor(num_samples))
        self.maximize = True
    
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        mean, sigma = self._mean_and_sigma(X.unsqueeze(1))
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        ei_batch = sigma * _ei_helper(u)
        return ei_batch.mean(dim=1)