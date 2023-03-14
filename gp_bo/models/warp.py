import torch

from torch.distributions import Kumaraswamy
from gpytorch import Module as GPyTorchModule
from gpytorch.constraints import Positive
from gpytorch.priors import LogNormalPrior

from typing import List,Optional, Union

def kumaraswamycdf(x,alpha,beta):
    return 1-(1-x**alpha)**beta

class InputWarp(GPyTorchModule):
    r"""A transform that uses learned input warping functions.
    Each specified input dimension is warped using the CDF of a
    Kumaraswamy distribution. Typically, MAP estimates of the
    parameters of the Kumaraswamy distribution, for each input
    dimension, are learned jointly with the GP hyperparameters.
    """

    def __init__(
        self,
        input_dim: int,
        eps: float = 1e-7,
        batch_shape: Optional[torch.Size] = None,
    ) -> None:
        r"""Initialize transform.
        Args:
            indices: The indices of the inputs to warp.
            eps: A small value used to clip values to be in the interval (0, 1).
            batch_shape: The batch shape.
        """
        super().__init__()
        self.batch_shape = batch_shape or torch.Size([])
        self._X_min = eps
        self._X_range = 1 - 2 * eps
        
        for i in (0,1): 
            # concentration0 -alpha
            # concentration1 -beta
            self.register_parameter(
                name='raw_concentration%d'%i, 
                parameter=torch.nn.Parameter(
                    # initial value corresponds to identity transform
                    #torch.full(batch_shape + self.indices.shape, 1.0)
                    torch.ones(*self.batch_shape,1,input_dim)
                )
            )

            self.register_constraint(
                param_name='raw_concentration%d'%i, 
                constraint=Positive(transform=torch.exp,inv_transform=torch.log,initial_value=1.0)
            )

            self.register_prior(
                name='concentration%d_prior'%i, 
                prior=LogNormalPrior(0.,0.75**0.5), # variance of 0.75 from the Snoek paper 
                param_or_closure='concentration%d'%i,
            )

    @property
    def concentration0(self):
        return self.raw_concentration0_constraint.transform(self.raw_concentration0)    
    
    @concentration0.setter
    def concentration0(self,v):
        self._set_concentration(0, v)
    
    @property
    def concentration1(self):
        return self.raw_concentration1_constraint.transform(self.raw_concentration1)

    @concentration1.setter
    def concentration1(self,v):
        self._set_concentration(1, v)

    def _set_concentration(self,i: int, value: Union[float, torch.Tensor]) -> None:
        raw_value = (
            self.raw_concentration0_constraint
            .inverse_transform(value.to(self.raw_concentration0))
        )
        self.initialize(**{'raw_concentration%d'%i:raw_value})
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Warp the inputs through the Kumaraswamy CDF.
        Args:
            X: A `input_batch_shape x (batch_shape) x n x d`-dim tensor of inputs.
                batch_shape here can either be self.batch_shape or 1's such that
                it is broadcastable with self.batch_shape if self.batch_shape is set.
        Returns:
            A `input_batch_shape x (batch_shape) x n x d`-dim tensor of transformed
                inputs.
        """
        # normalize to [eps, 1-eps], IDEA: could use Normalize and ChainedTransform.
        return kumaraswamycdf(
            torch.clamp(
                X * self._X_range + self._X_min,
                self._X_min,
                1.0 - self._X_min,
            ),self.concentration0,self.concentration1
        )