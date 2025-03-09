import numpy as np
import io
from typing import Optional,Union,List,Tuple,Callable, Dict

class Variable(object):
    def __init__(self,name:str) -> None:
        self.name = name
    
    def __repr__(self):
        raise NotImplementedError
    
    def sample(self,random_state:np.random.RandomState,size:int):
        '''
        Sample variable uniformly at random.
        '''
        raise NotImplementedError
    
    def _transform(self,vector:np.ndarray):
        raise NotImplementedError

    def _transform_scalar(self,x):
        return self._transform(x)

    def _inverse_transform(self,vector:np.ndarray):
        raise NotImplementedError

    def _inverse_transform_scalar(self,x):
        return self._inverse_transform(x)
    
    def _to_dict(self) -> Dict:
        pass

class Numerical(Variable):
    """Numerical variable with lower and upper bounds.

    Args:
        name: Name of the variable.
        lower: Lower bound.
        upper: Upper bound.
        log: If `True`, the variable is sampled on a logarithmic scale (for both
            random and latin-hypercube sampling). Default: `False`
    """
    def __init__(
        self, name:str,lower:Union[int,float],
        upper:Union[int,float],
        log:bool=False,
    )-> None:
        super().__init__(name=name)

        self.lower = float(lower)
        self.upper = float(upper)
        self.log = log

        if self.lower >= self.upper:
            raise ValueError(
                'Upper bound must be larger than lower bound for %s' % name
            )
        elif log and self.lower <= 0:
            raise ValueError(
                'Lower bound needs to be positive for log-scale'
            )

        if self.log:
            self._lower = np.log(self.lower)
            self._upper = np.log(self.upper)
        else:
            self._lower,self._upper = self.lower,self.upper
        
    def __repr__(self) -> str:
        repr_str = io.StringIO()
        repr_str.write(
            "%s, Type: Numerical, Range: [%s,%s]" %(
                self.name,repr(self.lower),repr(self.upper))
        )
        if self.log:
            repr_str.write(', on log-scale')
        
        repr_str.seek(0)
        return repr_str.getvalue()
    
    def sample(
        self,random_state:np.random.RandomState,size:int
    ) -> np.ndarray:
        return random_state.uniform(size=size)

    def _transform(self, vector:np.ndarray)->np.ndarray:
        out = self._lower + (self._upper-self._lower)*vector
        if self.log:
            out = np.exp(out)
        
        return np.clip(out,self.lower,self.upper)
    
    def _inverse_transform(self,vector:np.ndarray)->np.ndarray:
        if self.log:
            vector = np.log(vector)
        
        vector = (vector-self._lower)/(self._upper-self._lower)
        return np.clip(vector,0.,1.)
    
    def _to_dict(self):
        return {
            'name':self.name,
            'lower':self.lower,
            'upper':self.upper,
            'log':self.log,
            'type':'numerical'
        }

class Integer(Variable):
    """Integer variable with lower and upper bounds

    The only difference with :obj:`~.NumericalVariable` is that the samples are rounded
    to the closest integer.

    Args:
        name: Name of the variable.
        lower: Lower bound.
        upper: Upper bound.
        log: If `True`, the variable is sampled on a logarithmic scale (for both
            random and latin-hypercube sampling). Default: `False`
    """
    def __init__(
        self, name:str,lower:int,
        upper:int,
        log:bool=False,
    )-> None:
        super().__init__(name=name)
        self.lower = lower
        self.upper = upper
        self.log = log
        self.numvar = Numerical(name,self.lower,self.upper,self.log)
        
    def __repr__(self) -> str:
        repr_str = io.StringIO()
        repr_str.write(
            "%s, Type: Integer, Range: [%s,%s]" %(
                self.name,repr(self.lower),repr(self.upper))
        )
        if self.log:
            repr_str.write(', on log-scale')
        
        repr_str.seek(0)
        return repr_str.getvalue()
    
    def sample(
        self,random_state:np.random.RandomState,size:int
    ) -> np.ndarray:
        # first sample from [0,1]
        out = self.numvar.sample(random_state,size)
        # then convert back to the original scale
        # and rounding floats if any
        out = self._transform(out)
        # and then inverse-transform back to [0,1]
        return self._inverse_transform(out)

    def _transform(self, vector:np.ndarray)->np.ndarray:
        # transform from [0,1] to the correct scale
        out = self.numvar._transform(vector)
        return np.rint(out).astype(int)
    
    def _inverse_transform(self,vector:np.ndarray)->np.ndarray:
        return self.numvar._inverse_transform(vector)
    
    def _to_dict(self):
        return {
            'name':self.name,
            'lower':self.lower,
            'upper':self.upper,
            'log':self.log,
            'type':'integer'
        }
    

class Categorical(Variable):
    '''
    Categorical variable with pre-defined `levels`. 
    
    .. note::
        In the `transform` method, the levels are transformed to integers 
        in the range [0, `len(levels)` - 1]. 

    Args:
        name: Name of the variable.
        levels: List/tuple of levels. Each level is typically  one of 
            `str`, `float` or `int`.
    '''
    def __init__(
        self,name:str,levels:Union[List,Tuple]
    )-> None:
        
        super().__init__(name)
        self.levels = tuple(levels)
        self.num_levels = len(self.levels)
        self.levels_vector = list(range(self.num_levels))

    def __repr__(self) -> str:
        repr_str = io.StringIO()
        repr_str.write("%s, Type: Categorical, Levels: {" % (self.name))
        for idx, choice in enumerate(self.levels):
            repr_str.write(str(choice))
            if idx < len(self.levels) - 1:
                repr_str.write(", ")
        repr_str.write("}")

        repr_str.seek(0)
        return repr_str.getvalue()
    
    def sample(
        self,random_state:np.random.RandomState,size:int
    ) -> np.ndarray:
        return random_state.choice(self.num_levels,size=size,replace=True)

    def stratified_sample(
        self,random_state:np.random.RandomState,size:int
    ) -> np.ndarray:
        if size<self.num_levels:
            raise ValueError
        
        num_mult = size//self.num_levels
        out = np.array(self.levels_vector*num_mult)
        rem = size-num_mult*self.num_levels
        if rem>0:
            out = np.concatenate([
                out,random_state.choice(self.num_levels,size=rem,replace=False)
            ])
        
        random_state.shuffle(out)
        return out

    def _transform(self, vector):
        return np.array([
            self.levels[int(x)] for x in vector
        ])
    
    def _transform_scalar(self,x:int) -> Union[str,float,int]:
        return self.levels[int(x)]

    def _inverse_transform_scalar(self,x:Union[str,float,int]) -> int:
        return self.levels.index(x)
    
    def _inverse_transform(self,vector:np.ndarray) -> np.ndarray:
        return np.array([
            self.levels.index(x) for x in vector
        ])
    
    def _to_dict(self) -> Dict:
        return {
            'name':self.name,
            'levels':self.levels,
            'type':'categorical'
        }