import math
import torch
from torch.nn import Module as TModule
from gpytorch.priors import Prior
from torch.distributions import HalfCauchy
from torch.distributions.utils import broadcast_all

class HalfCauchyPrior(Prior, HalfCauchy):
    """
    Half-Cauchy prior.
    """

    def __init__(self, scale, validate_args=None, transform=None):
        TModule.__init__(self)
        HalfCauchy.__init__(self, scale=scale, validate_args=validate_args)
        self._transform = transform

    def expand(self, batch_shape):
        return HalfCauchyPrior(self.scale.expand(batch_shape))