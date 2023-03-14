import torch

from torch.distributions import Kumaraswamy
from gpytorch import Module as GPyTorchModule
from gpytorch.constraints import GreaterThan
from gpytorch.priors import LogNormalPrior

from botorch.models.transforms.utils import subset_transform
from botorch.models.transforms.input import ReversibleInputTransform

from typing import List,Optional, Union

def exp_with_shift(x:torch.Tensor)-> torch.Tensor:
    return 1e-4 + x.exp()

class InputWarp(ReversibleInputTransform, GPyTorchModule):
    r"""A transform that uses learned input warping functions.
    Each specified input dimension is warped using the CDF of a
    Kumaraswamy distribution. Typically, MAP estimates of the
    parameters of the Kumaraswamy distribution, for each input
    dimension, are learned jointly with the GP hyperparameters.
    """

    def __init__(
        self,
        indices: List[int],
        transform_on_train: bool = True,
        transform_on_eval: bool = True,
        transform_on_fantasize: bool = True,
        reverse: bool = False,
        eps: float = 1e-7,
        batch_shape: Optional[torch.Size] = None,
    ) -> None:
        r"""Initialize transform.
        Args:
            indices: The indices of the inputs to warp.
            transform_on_train: A boolean indicating whether to apply the
                transforms in train() mode. Default: True.
            transform_on_eval: A boolean indicating whether to apply the
                transform in eval() mode. Default: True.
            transform_on_fantasize: A boolean indicating whether to apply the
                transform when called from within a `fantasize` call. Default: True.
            reverse: A boolean indicating whether the forward pass should untransform
                the inputs.
            eps: A small value used to clip values to be in the interval (0, 1).
            batch_shape: The batch shape.
        """
        super().__init__()
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.transform_on_train = transform_on_train
        self.transform_on_eval = transform_on_eval
        self.transform_on_fantasize = transform_on_fantasize
        self.reverse = reverse
        self.batch_shape = batch_shape or torch.Size([])
        self._X_min = eps
        self._X_range = 1 - 2 * eps
        if len(self.batch_shape) > 0:
            # Note: this follows the gpytorch shape convention for lengthscales
            # There is ongoing discussion about the extra `1`.
            # TODO: update to follow new gpytorch convention resulting from
            # https://github.com/cornellius-gp/gpytorch/issues/1317
            batch_shape = self.batch_shape + torch.Size([1])
        else:
            batch_shape = self.batch_shape
        
        for i in (0,1): 
            # concentration0 -alpha
            # concentration1 -beta
            self.register_parameter(
                name='raw_concentration%d'%i, 
                parameter=torch.nn.Parameter(
                    # initial value corresponds to identity transform
                    torch.ones(*batch_shape,1,len(indices))
                )
            )

            self.register_constraint(
                param_name='raw_concentration%d'%i, 
                constraint=GreaterThan(1e-4,transform=torch.exp,inv_transform=torch.log)
            )

            self.register_prior(
                name='raw_concentration%d_prior'%i, 
                prior=LogNormalPrior(0.,0.75**0.5,transform=exp_with_shift), # variance of 0.75 from the Snoek paper 
                param_or_closure='raw_concentration%d'%i,
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

    def _set_concentration(self, i: int, value: Union[float, torch.Tensor]) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.concentration0)
        self.initialize(**{f"concentration{i}": value})

    @subset_transform
    def _transform(self, X: torch.Tensor) -> torch.Tensor:
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
        return self._k.cdf(
            torch.clamp(
                X * self._X_range + self._X_min,
                self._X_min,
                1.0 - self._X_min,
            )
        )

    @subset_transform
    def _untransform(self, X: torch.Tensor) -> torch.Tensor:
        r"""Warp the inputs through the Kumaraswamy inverse CDF.
        Args:
            X: A `input_batch_shape x batch_shape x n x d`-dim tensor of inputs.
        Returns:
            A `input_batch_shape x batch_shape x n x d`-dim tensor of transformed
                inputs.
        """
        if len(self.batch_shape) > 0:
            if self.batch_shape != X.shape[-2 - len(self.batch_shape) : -2]:
                raise Botorchtorch.TensorDimensionError(
                    "The right most batch dims of X must match self.batch_shape: "
                    f"({self.batch_shape})."
                )
        # unnormalize from [eps, 1-eps] to [0,1]
        return ((self._k.icdf(X) - self._X_min) / self._X_range).clamp(0.0, 1.0)

    @property
    def _k(self) -> Kumaraswamy:
        """Returns a Kumaraswamy distribution with the concentration parameters."""
        return Kumaraswamy(
            concentration1=self.concentration1,
            concentration0=self.concentration0,
        )