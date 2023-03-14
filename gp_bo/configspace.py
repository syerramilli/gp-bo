import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from collections import OrderedDict

class ConfigurationSpace(CS.ConfigurationSpace):

    def generate_indices(self) -> None:
        self.quant_index = []
        self.quant_names = []
        self.cond_quant_index = []
        self.cond_qual_index = []
        self.qual_index = []
        self.qual_names = []
        self.num_levels = OrderedDict()
        self.ndim = 0

        conditionals = self.get_all_conditional_hyperparameters()

        for i,hyp in enumerate(self.get_hyperparameters()):
            self.ndim +=1
            if isinstance(hyp,CSH.UniformFloatHyperparameter) or isinstance(hyp,CSH.UniformIntegerHyperparameter):
                if hyp.name in conditionals:
                    self.cond_quant_index.append(i)
                else:
                    self.quant_index.append(i)
                    self.quant_names.append(hyp.name)
                
            elif isinstance(hyp,CSH.CategoricalHyperparameter):
                self.qual_index.append(i)
                self.qual_names.append(hyp.name)
                self.num_levels[i] = len(hyp.choices)
                
                if hyp.name in conditionals:
                    self.cond_qual_index.append(i)
                    # additional level for missing
                    self.num_levels[i] += 1
    
    def get_conf_from_array(self,x:np.ndarray) -> CS.Configuration:
        conf_dict = {}
        for idx,hyp in enumerate(self.get_hyperparameters()):
            if idx in self.cond_quant_index:
                if x[idx] > 1.0:
                    continue
            elif idx in self.cond_qual_index:
                if x[idx] == self.num_levels[idx]-1:
                    continue

            conf_dict[hyp.name] = hyp._transform(x[idx])
        
        return CS.Configuration(self,conf_dict)
    
    def latinhypercube_sample(self,size:int):
        # valid only for configurations with all numerical parameters 
        n_dim = len(self.quant_index)
        
        # generate row and column grids
        grid_bounds = np.stack([np.linspace(0., 1., size+1) \
                                for i in np.arange(n_dim)],axis=1)
        grid_lower = grid_bounds[:-1,:]
        grid_upper = grid_bounds[1:,:]
        
        # generate 
        grid = grid_lower + (grid_upper-grid_lower)*self.random.rand(size,n_dim)
        
        # shuffle and return
        for i in range(n_dim):
            self.random.shuffle(grid[:,i])
        
        return [self.get_conf_from_array(x) for x in grid]
    