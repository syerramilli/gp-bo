import math
import torch
import gpytorch
from gpytorch import kernels
from gpytorch.models import ExactGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.outcome import Standardize

from gpytorch.constraints import GreaterThan, Positive
from gpytorch.priors import NormalPrior,LogNormalPrior, GammaPrior

from .priors import HalfCauchyPrior
from .warp import InputWarp

def exp_with_shift(x:torch.Tensor):
    return 1e-6+x.exp()

class GPR(ExactGP,GPyTorchModel):
    '''
    Gaussian Process Regression model with Matern kernel and Gaussian likelihood

    Args:
        train_x (torch.Tensor): training input data
        train_y (torch.Tensor): training output data
        warp_input (bool): whether to use input warping or not. If True, uses the 
            Kumaraswamy cdf as a monotonic transformation of the input data to
            account for any possible non-stationarities. Default is False.
    '''
    _num_outputs=1 # needed for botorch functions

    def __init__(
        self,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        warp_input:bool=False
    ) -> None:
    
        # initializing likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            constraints=GreaterThan(1e-6,transform=torch.exp,inv_transform=torch.log)
        )

        outcome_transform = Standardize(1, batch_shape=train_y.shape[:-2])
        train_y_sc,_ = outcome_transform(train_y)

        # initializing ExactGP
        super().__init__(train_x,train_y_sc.squeeze(-1), likelihood)

        # register outcome transform
        self.outcome_transform = outcome_transform

        # check if input warping is neeed
        self.register_buffer('warp_input',torch.tensor(warp_input))
        if self.warp_input:
            self.input_warping = InputWarp(input_dim = train_x.shape[-1])

        # Modules
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernels.ScaleKernel(
            base_kernel = kernels.MaternKernel(
                ard_num_dims=self.train_inputs[0].size(1),
                lengthscale_constraint=Positive(transform=torch.exp,inv_transform=torch.log),
                nu=2.5
            ),
            outputscale_constraint=Positive()
        )  

        # register priors
        self.likelihood.register_prior('noise_prior',HalfCauchyPrior(0.1,transform=exp_with_shift),'raw_noise')
        self.mean_module.register_prior('mean_prior',NormalPrior(0.,1.),'constant')
        self.covar_module.register_prior('outputscale_prior',LogNormalPrior(0.,1.),'outputscale')
        self.covar_module.base_kernel.register_prior(
            'lengthscale_prior',GammaPrior(3/2, 3.9/6),'lengthscale'
        )
    
    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(self.input_warping(x) if self.warp_input else x)
        return gpytorch.distributions.MultivariateNormal(mean_x,covar_x)
    
    def reset_parameters(self):
        # sample the hyperparameters from their respective priors
        # Note: samples in place
        for _,module,prior,closure,setting_closure in self.named_priors():
            if not closure(module).requires_grad:
                continue
            setting_closure(module,prior.expand(closure(module).shape).sample())

    def predict(self, x:torch.Tensor, return_std:bool=False):
        '''
        Convenience function that returns the posterior mean and 
        standard deviation of the response at locations specified by x.

        Args:
            x (torch.Tensor): input locations with shape (n_samples, input_dim)
            return_std (bool): whether to return the standard deviation or not. 
                If True, returns a tuple of (mean, std). Default is False.
        '''
        self.eval()

        out_dist = self.posterior(x).mvn
        out_mean = out_dist.loc

        if return_std:
            out_std = out_dist.stddev
            return out_mean,out_std

        return out_mean
        