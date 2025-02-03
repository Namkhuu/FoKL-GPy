import warnings
import numpy as np
from .utils import _str_to_bool, _process_kwargs, load
from .config import FoKLConfig
from .fokl_to_pyomo import fokl_to_pyomo
from .preprocessing.kernels import getKernels
from .preprocessing.dataFormat import dataFormat
from .sampler.samplers import fitSampler
from .postprocessing.postprocessing import postprocess
from .FoKL_Function.Functions import Functions


class FoKL:
    def __init__(self, **kwargs):
        self.config = FoKLConfig()
        self.dataFormat = dataFormat(self, self.config)
        self.functions = Functions(self, self.config, self.dataFormat)
        self.fitSampler = fitSampler(self, self.config, self.dataFormat, self.functions)
        self.postprocessing = postprocess(self, self.config, self.dataFormat, self.functions)

        current = _process_kwargs(self.config.DEFAULT, kwargs) # = default, but updated by any user kwargs
        for boolean in ['gimmie', 'way3', 'aic', 'UserWarnings', 'ConsoleOutput']:
            if not (current[boolean] is False or current[boolean] is True): 
                current[boolean] = _str_to_bool(current[boolean])

        # Load spline coefficients:
        phis = current['phis']  # in case advanced user is testing other splines
        if isinstance(current['kernel'], int):  # then assume integer indexing 'self.kernels'
            current['kernel'] = self.config.KERNELS[current['kernel']]  # update integer to string
        if current['phis'] is None: # if default
            if current['kernel'] == self.config.KERNELS[0]: # == 'Cubic Splines':
                current['phis'] = getKernels.sp500()
            elif current['kernel'] == self.config.KERNELS[1]:   # == 'Bernoulli Polynomials':
                current['phis'] = getKernels.bernoulli()
            elif isinstance(current['kernel'], str):    # confirm string before printing to console
                raise ValueError(f"The user-provided kernel '{current['phis']}' is not supported.")
            else:
                raise ValueError(f"The user-provided kernel is not supported.")
            
        if current['UserWarnings']:
            warnings.filterwarnings("default", category=UserWarning)
        else: 
            warnings.filterwarnings("ignore", category=UserWarning)

        for key, value, in current.items():
            setattr(self, key, value)
                
    def coverage3(self, **kwargs): 
        return self.postprocessing.coverage3(**kwargs)
    
    def fit(self, inputs=None, data=None, **kwargs):
        self.inputs, self.data, self.betas, self.minmax, self.mtx, evs = self.fitSampler.fit(inputs, data, **kwargs)
        return self.betas, self.mtx, self.minmax, evs
    
    def fitupdate(self, inputs=None, data=None, **kwargs):
        self.inputs, self.data, self.betas, self.minmax, self.mtx, evs = self.fitSampler.fitupdate(inputs, data)
        return self.betas, self.mtx, self.minmax, evs
    
    
    # need to do more examination 
    def clear(self, keep=None, clear=None, all=False):
        """
        Delete all attributes from the FoKL class except for hyperparameters and settings by default, but user may
        specify otherwise. If an attribute is listed in both 'clear' and 'keep', then the attribute is cleared.

        Optional Inputs:
            keep (list of strings)  == additional attributes to keep, e.g., ['mtx']
            clear (list of strings) == hyperparameters to delete, e.g., ['kernel', 'phis']
            all (boolean)           == if True then all attributes (including hyperparameters) get deleted regardless

        Tip: To remove all attributes, simply call 'self.clear(all=1)'.
        """

        if all is not False:  # if not default
            all = str_to_bool(all)  # convert to boolean if all='on', etc.

        if all is False:
            attrs_to_keep = self.config.KEEP  # default
            if isinstance(keep, list) or isinstance(keep, str):  # str in case single entry (e.g., keep='mtx')
                attrs_to_keep += keep  # add user-specified attributes to list of ones to keep
                attrs_to_keep = list(np.unique(attrs_to_keep))  # remove duplicates
            if isinstance(clear, list) or isinstance(clear, str):
                for attr in clear:
                    attrs_to_keep.remove(attr)  # delete attribute from list of ones to keep
        else:  # if all=True
            attrs_to_keep = []  # empty list so that all attributes get deleted

        attrs = list(vars(self).keys())  # list of all currently defined attributes
        for attr in attrs:
            if attr not in attrs_to_keep:
                delattr(self, attr)  # delete attribute from FoKL class if not keeping

        return
    
    def to_pyomo(self, xvars, yvars, m=None, xfix=None, yfix=None, truescale=True, std=True, draws=None):
        return fokl_to_pyomo(self, xvars, yvars, m, xfix, yfix, truescale, std, draws)
    
    def save(self, filename=None, **kwargs):
        return self.functions.save(filename, **kwargs)