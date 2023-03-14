import numpy as np
import torch
from ..models import GPR
from ..fit.mll_scipy import fit_model_scipy

from botorch.acquisition import ExpectedImprovement
from botorch.utils.transforms import normalize,unnormalize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import manual_seed

# optuna imports (works only if optuna is installed)
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler,QMCSampler, IntersectionSearchSpace
from optuna._transform import _SearchSpaceTransform
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

from typing import Optional,Callable,Dict,Any,Union,Sequence,OrderedDict


def gp_ei_candidates_func(
    train_x:torch.Tensor,train_y:torch.Tensor,bounds:torch.Tensor
) -> torch.Tensor:
    '''
    For the given data, fits a Gaussian Process model and then 
    computes the next candidate to try through optimizing 
    Expected Improvement
    '''
    train_x = normalize(train_x, bounds=bounds)
    
    model = GPR(train_x,train_y.squeeze(-1),warp_input=True).double()
    
    # fit model
    _ = fit_model_scipy(model,num_restarts=5)
    
    # acquistion function
    best_f = train_y.max()
    acq = ExpectedImprovement(model,best_f=best_f).double()
    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1
    
    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=standard_bounds,
        q=1,
        num_restarts=20,
        raw_samples=200
    )
    
    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


class UnconstrainedGPEISampler(BaseSampler):
    '''
    Optuna sampler interface that uses GP-based BO to generate the next trial
    for unconstrained single objective optimization problems. This is based on
    `optuna.integration.BoTorchSampler`.

    Parameters are transformed to continuous space before passing to the GP interface, and then
    transformed back to Optuna's representations. Categorical parameters are one-hot encoded.

    :param n_startup_trials: Number of initial trials drawn using QMC sampling
    :type n_startup_trials: int

    :param seed: Seed for random number generator
    :type seed: Optional[int]
    '''
    def __init__(
        self,
        n_startup_trials: int = 10,
        seed: Optional[int] = None,
    ):
        self._candidates_func = gp_ei_candidates_func
        self._constraints_func = None
        self._n_startup_trials = n_startup_trials
        self._independent_sampler = QMCSampler(seed=seed)
        self._seed = seed
        
        self._study_id = None
        self._search_space = IntersectionSearchSpace()
        self._device = torch.device('cpu')

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> Dict[str, BaseDistribution]:
        if self._study_id is None:
            self._study_id = study._study_id
        if self._study_id != study._study_id:
            # Note that the check below is meaningless when `InMemoryStorage` is used
            # because `InMemoryStorage.create_new_study` always returns the same study ID.
            raise RuntimeError("UnconstrainedGPEISampler cannot handle multiple studies.")

        search_space: Dict[str, BaseDistribution] = OrderedDict()
        for name, distribution in self._search_space.calculate(study, ordered_dict=True).items():
            if distribution.single():
                # built-in `candidates_func` cannot handle distributions that contain just a
                # single value, so we skip them. Note that the parameter values for such
                # distributions are sampled in `Trial`.
                continue
            search_space[name] = distribution

        return search_space
    
    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        assert isinstance(search_space, OrderedDict)

        if len(search_space) == 0:
            return {}

        trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

        n_trials = len(trials)
        if n_trials < self._n_startup_trials:
            return {}

        trans = _SearchSpaceTransform(search_space)
        n_objectives = len(study.directions)
        if n_objectives > 1:
            raise RuntimeError('UnconstrainedGPEISampler currently cannot handle multiple objectives')

        values: Union[np.ndarray, torch.Tensor] = np.empty(
            (n_trials, n_objectives), dtype=np.float64
        )
        params: Union[np.ndarray, torch.Tensor]
        bounds: Union[np.ndarray, torch.Tensor] = trans.bounds
        params = np.empty((n_trials, trans.bounds.shape[0]), dtype=np.float64)
        for trial_idx, trial in enumerate(trials):
            params[trial_idx] = trans.transform(trial.params)
            assert len(study.directions) == len(trial.values)

            for obj_idx, (direction, value) in enumerate(zip(study.directions, trial.values)):
                assert value is not None
                if direction == StudyDirection.MINIMIZE:  # BoTorch acquistion functopm assume maximization.
                    value *= -1
                values[trial_idx, obj_idx] = value


        values = torch.from_numpy(values).to(self._device)
        params = torch.from_numpy(params).to(self._device)
        bounds = torch.from_numpy(bounds).to(self._device)
        
        bounds.transpose_(0, 1)

        with manual_seed(self._seed):
            # `manual_seed` makes the default candidates functions reproducible.
            # `SobolQMCNormalSampler`'s constructor has a `seed` argument, but its behavior is
            # deterministic when the BoTorch's seed is fixed.
            candidates = self._candidates_func(params, values, bounds)
            if self._seed is not None:
                self._seed += 1
                
        candidates = candidates.squeeze(0)
        return trans.untransform(candidates.cpu().numpy())

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()
        if self._seed is not None:
            self._seed = np.random.RandomState().randint(2**60)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)
