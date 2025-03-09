import asyncio
import numpy as np
import os 
import time
import torch
import gpytorch
import warnings
import joblib

from ..models import GPR
from ..fit.mll_scipy import fit_model_scipy
from ..space.space import Space
from ..acquisition import ExpectedImprovementWithPending

from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples

from concurrent.futures import ProcessPoolExecutor

from typing import Callable,List,Union,Tuple,Optional,Dict
from collections import OrderedDict
from copy import deepcopy

class AsyncBayesOpt:
    def __init__(
        self,
        obj:Callable,
        config:Space,
        minimize:bool=True,
        n_initial:Optional[int]=None,
        n_parallel:int=1,
        execution_mode:str='local',
        refresh_rate:int=0.5,
        verbose:int=1
    ):
        self.obj = obj
        self.config = config
        self.minimize=minimize
        self.sign_mul = -1 if self.minimize else 1

        self.n_initial = n_initial if n_initial is not None else len(self.config.ndim)+1
        
        self.n_parallel = n_parallel
        self.execution_mode = execution_mode
        self.executor = ProcessPoolExecutor(max_workers=self.n_parallel) if self.execution_mode == 'local' else None
        self.refresh_rate = refresh_rate
        self.verbose=verbose

        # initialize modeling objects
        self.model = None
        self.model_lock = asyncio.Lock()
        self.train_x = torch.empty((0, config.ndim)).double()
        self.train_y = torch.empty((0, 1)).double()
        self.trial_meta = {} # stores metadata for each trial
        self.pending_trials = {}

        # initialize design
        self._x_init = draw_sobol_samples(
            bounds=torch.tensor([[0.0] * self.config.ndim, [1.0] * self.config.ndim]),
            n=self.n_initial, q=1
        ).squeeze(0)
        self._init_trial_idx = -1

    async def update_model(self):
        async with self.model_lock:
            if len(self.train_x) >= self.n_initial: # Ensure all initial confs are evaluated
                self.model = GPR(self.train_x, self.train_y).double()
                _ = fit_model_scipy(self.model, num_restarts=5)
                self.model.eval()
    
    async def suggest_next(self):
        if len(self.train_x) < self.n_initial:
            if self._init_trial_idx < self.n_initial-1:
                self._init_trial_idx += 1
                return self._x_init[self._init_trial_idx, :], "Sobol"
            return None, None # Wait until all initial confs are evaluated

        async with self.model_lock:
            best_f = self.train_y.max()
            if len(self.pending_trials) == 0:
                ei = ExpectedImprovement(self.model, best_f)
            else:
                ei = ExpectedImprovementWithPending(
                    self.model, best_f,
                    X_pending=torch.cat(list(self.pending_trials.values()), dim=-2),
                    num_samples=64
                )
            
            next_x, _ = optimize_acqf(
                acq_function=ei,
                bounds=torch.tensor([[0.0] * self.config.ndim, [1.0] * self.config.ndim]),
                q=1,
                num_restarts=20,
                raw_samples=200
            )
        
        return next_x, "EI"
    
    async def evaluate(self, x:torch.Tensor, trial_id:int):
        conf = self.config.get_dict_from_array(x.numpy().ravel())
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, self.obj, conf)

        if self.verbose >= 2:
            print(f"Trial {trial_id} with config {conf} completed with value={result}")
        self.train_x = torch.cat([self.train_x, x])
        self.train_y = torch.cat([self.train_y, self.sign_mul * torch.tensor([[result]])])
        del self.pending_trials[trial_id]

        # update meta
        self.trial_meta[trial_id].update({
            'conf': conf, 'value': result
        })

        if len(self.train_x) >= self.n_initial: # ensure all initial confs are evaluated
            await self.update_model()
    

    async def run(self, n_trials:int):
        trial_count = 0

        try:
            while trial_count < n_trials or self.pending_trials:
                while len(self.pending_trials) < self.n_parallel and trial_count < n_trials:
                    x_next, source = None, None
                    while x_next is None:
                        x_next, source = await self.suggest_next()
                        await asyncio.sleep(self.refresh_rate)
                    
                    trial_id = trial_count
                    self.pending_trials[trial_id] = x_next
                    self.trial_meta[trial_id] = {'source': source}
                    asyncio.create_task(self.evaluate(x_next, trial_id))
                    trial_count += 1
                
                await asyncio.sleep(self.refresh_rate)
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True)
        
        self.best_idx = self.train_y.argmax()
        self.best_conf = self.config.get_dict_from_array(self.train_x[self.best_idx].numpy().ravel())
        self.best_value = self.sign_mul * self.train_y[self.best_idx].item()
        if self.verbose >= 1:    
            print(f'Best configuration found: {self.best_conf}')
            print(f'Best value: {self.best_value}')