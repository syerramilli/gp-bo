import numpy as np
import io
import json
from collections import OrderedDict
from typing import Optional,Union,List ,Dict,Callable

from .variables import Variable,Numerical, Categorical, Integer

class Space(object):
    '''
    Utility class containing a set of input variables along with their domains, and some
    additional methods for sampling and standardizing the inputs.

    When sampling, the inputs are always returned in array form. Categorical variables are
    encoded as ordinal integers - 0,1,...,L-1, where L is the number of levels. Numerical and 
    integer variables are scaled to the unit hypercube [0,1]. If the `logscale` attribute of 
    any such variable is `True`, the variable is first log-transformed and then scaled. 
    
    Use the `get_dict_from_array` method to get a dictionary of the actual input variable values
    for a single sample. For multiple samples, iteratively call this method over each sample.

    In array form, the variables are ordered in the order that are added to the input space object.
    Use the attributes :attr:`qual_index` and :attr:`quant_index` to obtain the list of qualitative
    or quantitative variables respectively.

    Useful attributes:
        :attr:`qual_index` (List[int])
            List containing the indices of the qualitative inputs in array form.
        :attr:`quant_index` (List[int])
            List containing the indices of the quantitative inputs in array form. This list also
            includes single level qualitative variables.
        :attr:`num_levels` (Dict[int,int])
            Ordered dictionary with the index of the qualitative variables as keys
            and their corresponding number of levels as values
        :attr:`n_dim` (int)
            Number of variables in the input space
    '''
    def __init__(self):
        super().__init__()

        self._variables = OrderedDict()
        self._variable_index = OrderedDict()
        self._variable_names = []
        
        self.ndim = 0
        self.quant_index = []
        self.qual_index = []
        self.num_levels = OrderedDict()
    
    def add_input(self,variable:Variable) -> None:
        '''
        Adds a variable to the input space and update the indices

        :param variable: The variable to add
        :type variable: Variable
        '''
        if not isinstance(variable,Variable):
            raise TypeError
        elif variable.name in self._variables:
            raise ValueError(
                'Variable %s is already in the space'%variable.name
            )
        
        self._variables[variable.name] = variable
        self._variable_names.append(variable.name)
        self._variable_index[variable.name] = self.ndim

        if isinstance(variable,Categorical):
            if len(variable.levels) > 1:
                # including only categorical variables with more than 1 level
                self.qual_index.append(self.ndim)
                self.num_levels[self.ndim] = len(variable.levels)
            else:
                self.quant_index.append(self.ndim)
        else:
            self.quant_index.append(self.ndim)
        # update number of parameters
        self.ndim += 1
    
    def add_inputs(self,var_list:List[Variable]) -> None:
        '''
        Adds variables in :attr:`var_list` to the input space

        :param var_list: List of variables to add.
        :type var_list: List[Variable]
        '''
        for var in var_list:
            self.add_input(var)
    
    def get_variables(self) -> List[Variable]:
        '''
        Return the list of variables in the input space (in order)
        '''
        return list(self._variables.values())
    
    def get_variable_names(self) -> List[str]:
        '''
        Return the list of names of the variables in the input space (in order)
        '''
        return self._variables.keys()

    def get_variable_by_name(self,name:str) -> Variable:
        '''
        Gets the variable from the input space from its name. Returns `None` if
        there is no variable with the given name

        :param name: Name of the searched variable
        :type name: str 
        '''
        return self._variables.get(name)
    
    def get_variable_by_idx(self,idx:int) -> Variable:
        '''
        Gets the variable from the input space from its index in array from. Throws an
        error if index is invalid. 

        :param idx: Index of the searched variable
        :type idx: int
        '''
        return self.get_variable_by_name(
            self._variable_names[idx]
        )

    def __repr__(self):
        s = io.StringIO()
        s.write('Input space with variables:\n')
        
        if len(self._variables) > 0:
            s.write('\n')
            s.write('\n'.join([
                str(var) for var in self.get_variables()
            ]))
        return s.getvalue()
    
    def __len__(self):
        return self.ndim

    def get_dict_from_array(
        self,x:np.ndarray
    ) -> Dict[str,Union[str,float,int]]:
        '''
        Given an input sample in array form, returns a dictionary of variable-value
        pairs. The values are returned in the original scale. 

        :param x: 1D array containing the input sample in the transformed scale
        :type x: np.ndarray
        '''
        conf_dict = {}
        for i,var in enumerate(self.get_variables()):
            conf_dict[var.name] = var._transform_scalar(x[i])
        return conf_dict

    def get_array_from_dict(
        self,values: Dict[str,Union[str,float,int]]
    ) -> np.ndarray:
        '''
        Given a dictionary with the variable-value pairs, return an array with 
        the transformed values

        :param values: dictionary containing variable value pairs. This needs to 
            contain all variables in the input space - missing variables will raise
            errors
        :type values: Dict[str,Union[str,float,int]]
        '''
        out = np.zeros(self.ndim)
        for i,var in enumerate(self.get_variables()):
            out[i] = var._inverse_transform_scalar(values[var.name])
        return out

    def random_sample(
        self,
        rng:Optional[Union[int,np.random.RandomState]]=None,
        size:int = 1,
    ) -> np.ndarray:
        '''
        Returns samples drawn uniformly at random in array form

        :param rng: The seed of the psuedo random number generator. This can either 
            be a RandomState instance or an integer that can be used to initialize one.
        :type rng: Union[int,np.random.RandomState], optional

        :param size: Number of samples to draw
        :type size: int

        :returns: A :attr:`size` x :attr:`ndim` array with the different values in 
            the transformed scale
        :rtype: np.ndarray
        '''
        rng = _check_random_state(rng)
        out = np.zeros((size,self.ndim))
        for i,var in enumerate(self.get_variables()):
            out[:,i] = var.sample(rng,size)
        
        return out

    @staticmethod
    def from_json(path:str) -> 'Space':
        '''
        Creates an input space object from a JSON file

        :param path: Path to the JSON file
        :type path: str
        '''
        with open(path,'r') as f:
            data = json.load(f)
        return Space.from_dict(data)
    
    @staticmethod
    def from_dict(config) ->  'Space':
        '''
        Creates an input space object from a dictionary

        :param config: Dictionary containing the input space configuration
        :type config: Dict
        '''
        space = Space()
        for var in config.values():
            if var['type'] == 'categorical':
                space.add_input(
                    Categorical(name = var['name'],levels = var['levels'])
                )
            elif var['type'] == 'numerical':
                space.add_input(
                    Numerical(
                        name=var['name'], 
                        lower=var['lower'], upper=var['upper'],
                        log= var.get('log',False)
                    )
                )
            elif var['type'] == 'integer':
                space.add_input(
                    Integer(
                        name=var['name'], 
                        lower=var['lower'], upper=var['upper'],
                        log= var.get('log',False)
                    )
                )
            else:
                raise ValueError(f'Unknown variable type { var["type"] }. Must be one of categorical, numerical, integer')
        return space

    def to_dict(self) -> Dict:
        '''
        Returns a dictionary representation of the input space
        '''
        if len(self._variables) == 0:
            raise ValueError('No variables in the input space')
        
        return {var.name:var._to_dict() for var in self.get_variables()}
    
    def to_json(self, path:str) -> None:
        '''
        Saves the input space to a JSON file

        :param path: Path to the JSON file
        :type path: str
        '''
        with open(path,'w') as f:
            json.dump(self.to_dict(), f)

    def latinhypercube_sample(
        self,
        rng:Optional[Union[int,np.random.RandomState]]=None,
        size:int=1,
    ) -> np.ndarray:
        '''
        Returns samples drawn from a "latin-hypercube" design in array form. 

        :param rng:  The seed of the psuedo random number generator. This can either 
            be a RandomState instance or an integer that can be used to initialize one.
        :type rng: Union[int,np.random.RandomState], optional
        
        :param size: Number of samples to draw
        :type size: int

        :returns: A :attr:`size` x :attr:`ndim` array with the different values in 
            the transformed scale
        :rtype: np.ndarray
        '''
        rng = _check_random_state(rng)
        out = np.zeros((size,self.ndim))
        if len(self.quant_index) > 0:
            out[:,self.quant_index] = _latinhypercube_sample(
                rng,len(self.quant_index),size
            )
        
        if len(self.qual_index) > 0:
            for idx in self.qual_index:
                out[:,idx] = self.get_variable_by_idx(idx).stratified_sample(rng,size)
        
        return out

def _check_random_state(rng):
    if rng is None or rng is np.random:
        return np.random.mtrand._rand
    elif isinstance(rng,int):
        return np.random.RandomState(rng)
    elif isinstance(rng,np.random.RandomState):
        return rng
    else:
        raise ValueError(
            "%r cannot be used to seed a numpy.random.RandomState instance" % rng
        )


def _latinhypercube_sample(
    rng:Union[int,Callable,np.random.RandomState],
    ndim:int,size:int)->np.ndarray:
    # generate row and column grids
    grid_bounds = np.stack([np.linspace(0., 1., size+1) \
                            for i in np.arange(ndim)],axis=1)
    grid_lower = grid_bounds[:-1,:]
    grid_upper = grid_bounds[1:,:]
    
    # generate 
    grid = grid_lower + (grid_upper-grid_lower)*rng.rand(size,ndim)
    
    # shuffle and return
    for i in range(ndim):
        rng.shuffle(grid[:,i])
    
    return grid