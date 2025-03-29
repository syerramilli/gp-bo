import asyncio
import time
import torch
import pandas as pd
from ..models import GPR
from ..fit.mll_scipy import fit_model_scipy
from ..space.space import Space
from ..acquisition import ExpectedImprovementWithPending

from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples

from concurrent.futures import ProcessPoolExecutor

from typing import Callable,List,Tuple,Optional,Dict
from numbers import Number

import nest_asyncio
nest_asyncio.apply()  # Ensures nested event loops work in Jupyter notebooks

class AsyncBayesOpt:
    def __init__(
        self,
        obj:Callable,
        config:Space,
        minimize:bool=True,
        n_initial:Optional[int]=None,
        n_batch:int=1,
        use_local_parallelism: bool = False,
        refresh_rate:int=0.5,
        verbose:int=1
    ):
        self.obj = obj
        self.config = config
        self.minimize=minimize
        self.sign_mul = -1 if self.minimize else 1

        self.n_initial = n_initial if n_initial is not None else len(self.config.ndim)+1
        
        self.n_batch = n_batch
        self.use_local_parallelism = use_local_parallelism
        self.executor = ProcessPoolExecutor(max_workers=self.n_batch) if self.use_local_parallelism else None
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

    def run_trials(self, n_trials:int) -> Tuple[Dict[str,Number],Number]:
        # Start overall timer
        self.start_time = time.time()
        try:
            loop = asyncio.get_running_loop()  # Get the current event loop (Jupyter case)
        except RuntimeError:
            loop = asyncio.new_event_loop()  # If no event loop exists, create one
            asyncio.set_event_loop(loop)

        best_conf, best_value = loop.run_until_complete(self._run(n_trials))
        return best_conf, best_value
    
    def get_trials_dataframe(self) -> pd.DataFrame:
        """Returns a pandas DataFrame with the trial metadata."""
        df = pd.DataFrame.from_dict(self.trial_meta, orient="index")
        conf_df = pd.json_normalize(df["conf"])  # Expands dictionary values into columns
        df = df.drop(columns=["conf"])
        df = pd.concat([df, conf_df], axis=1)
        return df 

    def get_incumbent_vs_trials(self) -> List[Number]:
        """Returns a list of the best incumbent values over the number of completed trials."""
        if not hasattr(self, "train_y") or len(self.train_y) == 0:
            raise ValueError("No completed trials yet.")

        incumbent_values = []
        best_so_far = float('-inf') if not self.minimize else float('inf')

        for i in range(len(self.train_y)):
            best_so_far = max(best_so_far, self.train_y[i].item()) if not self.minimize else min(best_so_far, -self.train_y[i].item())
            incumbent_values.append(best_so_far)

        return incumbent_values

    def get_incumbent_vs_time(self) -> Tuple[List[Number],List[Number]]:
        """Returns a list of timestamps and the best incumbent values over elapsed time.
        
        Ensures trials are processed in chronological order of completion.
        """
        if not hasattr(self, "train_y") or len(self.train_y) == 0:
            raise ValueError("No completed trials yet.")

        # Collect (end_time, value) for completed trials
        completed_trials = [
            (self.trial_meta[trial_id]["end_time"] - self.start_time, self.trial_meta[trial_id]["value"])
            for trial_id in self.trial_meta
            if "end_time" in self.trial_meta[trial_id]
        ]

        # Sort by end time to process in correct order
        completed_trials.sort(key=lambda x: x[0])

        # Track best incumbent value dynamically
        incumbent_values = []
        time_stamps = []
        best_so_far = float('-inf') if not self.minimize else float('inf')

        for elapsed_time, value in completed_trials:
            best_so_far = max(best_so_far, value) if not self.minimize else min(best_so_far, value)
            incumbent_values.append(best_so_far)
            time_stamps.append(elapsed_time)

        return time_stamps, incumbent_values

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

        self.trial_meta[trial_id]['start_time'] = time.time()
        result = await loop.run_in_executor(self.executor, self.obj, conf)
        # Log trial completion time
        self.trial_meta[trial_id]['end_time'] = time.time()
        self.trial_meta[trial_id]['duration'] = (
            self.trial_meta[trial_id]['end_time'] - self.trial_meta[trial_id]['start_time']
        )

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
    
    async def _run(self, n_trials:int):
        trial_count = 0
        
        try:
            while trial_count < n_trials or self.pending_trials:
                while len(self.pending_trials) < self.n_batch and trial_count < n_trials:
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

        return self.best_conf, self.best_value