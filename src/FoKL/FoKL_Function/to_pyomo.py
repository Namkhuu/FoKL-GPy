def to_pyomo(self, xvars, yvars, m=None, xfix=None, yfix=None, truescale=True, std=True, draws=None):
    """Passes arguments to external function. See 'fokl_to_pyomo' for more documentation."""
    return fokl_to_pyomo(self, xvars, yvars, m, xfix, yfix, truescale, std, draws)