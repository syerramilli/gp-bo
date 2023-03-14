import numpy as np
import os 
import time
import torch
import gpytorch
import warnings
import joblib

from ..models import GPR
from ..fit.mll_scipy import fit_model_scipy
from ..configspace import ConfigurationSpace

from botorch.acquisition import (
    ExpectedImprovement,qExpectedImprovement
)
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler

from typing import Callable,List,Union,Tuple,Optional,Dict
from collections import OrderedDict
from copy import deepcopy

class BayesOpt:
    def __init__(
        self,
        obj:Callable,
        config:ConfigurationSpace,
        minimize:bool=True,
        acq_function:str='EI',
        acq_function_options:Optional[Dict]={},
        batch_acquisition:bool=False,
        acquisition_q:int=1,
        verbose:int=1,
        save_iter:Optional[int]=None,
        save_dir:Optional[int]=None,
        n_jobs:int=1
    ):
        self.obj = obj
        self.config = config
        self.minimize=minimize
        self.sign_mul = -1 if self.minimize else 1
        self.acq_function = acq_function
        self.batch_acquisition = batch_acquisition
        self.acquisition_q = acquisition_q

        self.acq_function_options = acq_function_options
        self.verbose=verbose
        self.save_iter = save_iter
        self.save_dir = save_dir
        self.n_jobs = 1

        # initialize objects
        self.model = None
        self.train_confs = None
        self.train_x = []
        self.train_y = []
        self.confs_inc = []
        self.confs_cand = []
        self.f_inc_obs = []
        self.f_inc_est = []
        self.acq_vec = []
        self.fit_time = []
        self.acqopt_time = []
        self.obj_eval_time = []
        self.initial_params = None
    
    def run(self,n_iter:int,n_init:Optional[int]=None) -> Dict:

        # output_header = '%6s %10s %10s %10s %12s' % \
        #             ('iter', 'f_inc_obs', 'f_inc_est','acq_cand',"term_metric")
        output_header = '%6s %10s %10s %10s' % \
                    ('iter', 'f_inc_obs', 'f_inc_est','acq_cand')
        for i in range(n_iter):
            # either initialize observations or evaluate the next candidate
            self._initialize(n_init)

            # fit model and find incumbent
            self._fit_model_and_find_inc()
        
            # acquisition find next candidate
            self._acquisition()

            if self.save_iter is not None:
                if i % self.save_iter == 0:
                    self.save_to_file(self.save_dir)

            # update verbose statements
            if self.verbose >= 2:
                if i%10 == 0:
                    # print header every 19 iterations
                    print(output_header)
                #term_metric = (self.f_inc_est[-1]-self.acq_vec[-1])/self.sigma_vec[-1]/self.kappa
                # print('%6i %10.3e %10.3e %10.3e %12.4f' %\
                #       (i, self.f_inc_obs[-1],self.f_inc_est[-1],
                #       self.acq_vec[-1],term_metric))
                print('%6i %10.3e %10.3e %10.3e' %\
                      (i, self.f_inc_obs[-1],self.f_inc_est[-1],
                      self.acq_vec[-1]))
        
        if self.verbose >= 1:
            print('')
            print('Number of candidates evaluated.....: %g' % len(self.train_confs))
            print('Observed obj at incumbent..........: %g' % self.f_inc_obs[-1])
            print('Estimated obj at incumbent.........: %g' % self.f_inc_est[-1])
            print('')
            print('Incumbent at termination:')
            print(self.confs_inc[-1])
            print('')
            print('Candidate at termination:')
            print(self.confs_cand[-1])
        
        # return best configuration and other statistics
        results = OrderedDict()
        results['conf_inc'] = self.confs_inc[-1]
        results['f_inc_obs'] = self.f_inc_obs[-1]
        results['f_inc_est'] = self.f_inc_est[-1]

        return results

    def save_to_file(self,folder):
        #  optimization statistics
        stat_keys = [
            'f_inc_obs','f_inc_est','acq_vec',
            'confs_inc','confs_cand',
            'fit_time','acqopt_time','obj_eval_time',
        ]
        stats = {
            key:getattr(self,key) for key in stat_keys
        }
        joblib.dump(stats,os.path.join(folder,'stats.pkl'))
        # Observations
        _ = torch.save({
            key:getattr(self,key) for key in ['train_x','train_y']
        },os.path.join(folder,'model_train.pt'))
        # model state dict
        _ = torch.save(self.model.state_dict(),os.path.join(folder,'model_state.pth'))


    def _initialize(self,n_init:Optional[int]=None):
        if self.train_confs is None:
            if n_init is None:
                n_init = len(self.config.quant_index) + 1
            
            self.config.seed(np.random.randint(2e+4))
            self.train_confs = self.config.latinhypercube_sample(n_init)

            for conf in self.train_confs:
                x,y,eval_time = self._evaluate(conf)
                self.train_x.append(x)
                self.train_y.append(y)
                self.obj_eval_time.append(eval_time)
            
            self.train_x = torch.tensor(self.train_x).double()
            self.train_y = torch.tensor(self.train_y).double()
        else:
            # algorithm has been run previously
            # evaluate the next candidate 
            next_conf_list = self.confs_cand[-1]
            
            for next_conf in next_conf_list:
                # TODO: add support for parallel evaluations
                next_x,next_y,eval_time = self._evaluate(next_conf)
                self.train_confs.append(next_conf)
                self.train_y = torch.cat([self.train_y,torch.tensor([next_y]).to(self.train_y)])
                self.train_x = torch.cat([self.train_x,torch.tensor(next_x).to(self.train_x).reshape(1,-1)])
                self.obj_eval_time.append(eval_time)
    
    def _evaluate(self,conf,**kwargs):
        start_time = time.time()
        y = self.obj(conf.get_dictionary(),**kwargs)
        eval_time = time.time()-start_time
        return conf.get_array(),y,eval_time
    
    def _fit_model_and_find_inc(self) -> None:
        # construct model
        self.model = self._construct_model()

        start_time = time.time()
        if self.initial_params is not None:
            self.model.initialize(**self.initial_params)

        _ = fit_model_scipy(model = self.model,num_restarts = 5)

        self.initial_params = OrderedDict()
            
        self.fit_time.append(time.time()-start_time)

        # disable model gradients
        for name,parameter in self.model.named_parameters():
            parameter.requires_grad_(False)
            self.initial_params[name] = parameter

        # find incumbent
        with warnings.catch_warnings(),torch.no_grad(): 
            warnings.simplefilter(action='ignore',category=gpytorch.utils.warnings.GPInputWarning)
            train_pred = torch.tensor([
                self.model.predict(x.view(1,-1)) for x in self.train_x
            ])

        fmin_index = train_pred.argmax().item() 
        self.confs_inc.append(self.train_confs[fmin_index])
        self.f_inc_obs.append(self.train_y[fmin_index].item())
        self.f_inc_est.append(self.sign_mul*train_pred[fmin_index].item())
    
    def _construct_model(self):
        return GPR(
            train_x = self.train_x,
            train_y = self.sign_mul*self.train_y,
        ).double()
    
    def _acquisition(self) -> None:
        if self.acq_function == 'EI':
            best_f = -self.f_inc_est[-1] if self.minimize else self.f_inc_est[-1]

            if self.batch_acquisition:
                sampler = SobolQMCNormalSampler(512)
                acqobj = qExpectedImprovement(self.model,best_f,sampler)
            else:
                acqobj = ExpectedImprovement(self.model, best_f)

        start_time = time.time()
        new_x, max_acq = optimize_acqf(
            acqobj, 
            bounds=torch.tensor([[0.0] * self.train_x.shape[-1], [1.0] * self.train_x.shape[-1]]),
            q=1 if not self.batch_acquisition else self.acquisition_q,
            num_restarts=20,
            raw_samples=200
        )
        end_time = time.time()
        self.acqopt_time.append(end_time-start_time)
        self.confs_cand.append([self.config.get_conf_from_array(x.numpy()) for x in new_x])
        self.acq_vec.append(max_acq.item())

