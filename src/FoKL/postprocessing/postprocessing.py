from ..utils import process_kwargs, str_to_bool, merge_dicts
import matplotlib.pyplot as plt
import numpy as np
import warnings

class postprocess:
    def __init__(self, fokl, config, dataFormat, functions):
        self.fokl = fokl
        self.config = config
        self.dataFormat = dataFormat
        self.functions = functions
         
    def coverage3(self, **kwargs):
        """
        For validation testing of FoKL model. Default functionality is to evaluate all inputs (i.e., train+test sets).
        Returned is the predicted output 'mean', confidence bounds 'bounds', and root mean square error 'rmse'. A plot
        may be returned by calling 'coverage3(plot=1)'; or, for a potentially more meaningful plot in terms of judging
        accuracy, 'coverage3(plot='sorted')' plots the data in increasing value.
        Optional inputs for numerical evaluation of model:
            inputs == normalized and properly formatted inputs to evaluate              == self.inputs (default)
            data   == properly formatted data outputs to use for validating predictions == self.data (default)
            draws  == number of beta terms used                                         == self.draws (default)
        Optional inputs for basic plot controls:
            plot              == binary for generating plot, or 'sorted' for plot of ordered data == False (default)
            bounds            == binary for plotting bounds                                       == True (default)
            xaxis             == integer indexing the input variable to plot along the x-axis     == indices (default)
            labels            == binary for adding labels to plot                                 == True (default)
            xlabel            == string for x-axis label                                          == 'Index' (default)
            ylabel            == string for y-axis label                                          == 'Data' (default)
            title             == string for plot title                                            == 'FoKL' (default)
            legend            == binary for adding legend to plot                                 == True (default)
            LegendLabelFoKL   == string for FoKL's label in legend                                == 'FoKL' (default)
            LegendLabelData   == string for Data's label in legend                                == 'Data' (default)
            LegendLabelBounds == string for Bounds's label in legend                              == 'Bounds' (default)
        Optional inputs for detailed plot controls:
            PlotTypeFoKL   == string for FoKL's color and line type  == 'b' (default)
            PlotSizeFoKL   == scalar for FoKL's line size            == 2 (default)
            PlotTypeBounds == string for Bounds' color and line type == 'k--' (default)
            PlotSizeBounds == scalar for Bounds' line size           == 2 (default)
            PlotTypeData   == string for Data's color and line type  == 'ro' (default)
            PlotSizeData   == scalar for Data's line size            == 2 (default)
        Return Outputs:
            mean   == predicted output values for each indexed input
            bounds == confidence interval for each predicted output value
            rmse   == root mean squared deviation (RMSE) of prediction versus known data
        """
        # Process keywords:
        default = {
            # For numerical evaluation of model:
            'inputs': None, 'data': None, 'draws': self.config.DEFAULT['draws'],
            # For basic plot controls:
            'plot': False, 'bounds': True, 'xaxis': False, 'labels': True, 'xlabel': 'Index', 'ylabel': 'Data',
            'title': 'FoKL', 'legend': True, 'LegendLabelFoKL': 'FoKL', 'LegendLabelData': 'Data',
            'LegendLabelBounds': 'Bounds',
            # For detailed plot controls:
            'PlotTypeFoKL': 'b', 'PlotSizeFoKL': 2, 'PlotTypeBounds': 'k--', 'PlotSizeBounds': 2, 'PlotTypeData': 'ro',
            'PlotSizeData': 2
        }
        current = process_kwargs(default, kwargs)
        if isinstance(current['plot'], str):
            if current['plot'].lower() in ['sort', 'sorted', 'order', 'ordered']:
                current['plot'] = 'sorted'
                if current['xlabel'] == 'Index':  # if default value
                    current['xlabel'] = 'Index (Sorted)'
            else:
                warnings.warn("Keyword input 'plot' is limited to True, False, or 'sorted'.", category=UserWarning)
                current['plot'] = False
        else:
            current['plot'] = str_to_bool(current['plot'])
        for boolean in ['bounds', 'labels', 'legend']:
            current[boolean] = str_to_bool(current[boolean])
        if current['labels']:
            for label in ['xlabel', 'ylabel', 'title']:  # check all labels are strings
                if current[label] and not isinstance(current[label], str):
                    current[label] = str(current[label])  # convert numbers to strings if needed (e.g., title=3)
        # Check for and warn about potential issues with user's 'input'/'data' combinations:
        if current['plot']:
            warn_plot = ' and ignoring plot.'
        else:
            warn_plot = '.'
        flip = [1, 0]
        flop = ['inputs', 'data']
        for i in range(2):
            j = flip[i]  # [i, j] = [[0, 1], [1, 0]]
            if current[flop[i]] is not None and current[flop[j]] is None:  # then data does not correspond to inputs
                warnings.warn(f"Keyword argument '{flop[j]}' should be defined to align with user-defined '{flop[i]}'. "
                              f"Ignoring RMSE calculation{warn_plot}", category=UserWarning)
                current['data'] = False  # ignore data when plotting and calculating RMSE
        if current['data'] is False and current['plot'] == 'sorted':
            warnings.warn("Keyword argument 'data' must correspond with 'inputs' if requesting a sorted plot. "
                          "Returning a regular plot instead.", category=UserWarning)
            current['plot'] = True  # regular plot
        # Define 'inputs' and 'data' if default (defined here instead of in 'default' to avoid lag for large datasets):
        if current['inputs'] is None:
            current['inputs'] = self.fokl.inputs # in fit
        if current['data'] is None:
            current['data'] = self.fokl.data
        def check_xaxis(current):
            """If plotting, check if length of user-defined x-axis aligns with length of inputs."""
            if current['xaxis'] is not False and not isinstance(current['xaxis'], int):  # then assume vector
                warn_xaxis = []
                l_xaxis = len(current['xaxis'])
                try:  # because shape any type of inputs is unknown, try lengths of different orientations
                    if l_xaxis != np.shape(current['inputs'])[0] and l_xaxis != np.shape(current['inputs'])[1]:
                        warn_xaxis.append(True)
                except:
                    warn_xaxis = warn_xaxis  # do nothing
                try:
                    if l_xaxis != np.shape(current['inputs'])[0]:
                        warn_xaxis.append(True)
                except:
                    warn_xaxis = warn_xaxis  # do nothing
                try:
                    if l_xaxis != len(current['inputs']):
                        warn_xaxis.append(True)
                except:
                    warn_xaxis = warn_xaxis  # do nothing
                if any(warn_xaxis):  # then vectors do not align
                    warnings.warn("Keyword argument 'xaxis' is limited to an integer indexing the input variable to "
                                  "plot along the x-axis (e.g., 0, 1, 2, etc.) or to a vector corresponding to 'data'. "
                                  "Leave blank (i.e., False) to plot indices along the x-axis.", category=UserWarning)
                    current['xaxis'] = False
            return current['xaxis']
        # Define local variables:
        normputs = current['inputs']  # assumes inputs are normalized and formatted correctly
        data = current['data']
        draws = current['draws']
        mean, bounds = self.evaluate(normputs, draws=draws, ReturnBounds=1, _suppress_normalization_warning=True)
        n, mputs = np.shape(normputs)  # Size of normalized inputs ... calculated in 'evaluate' but not returned
        if current['plot']:  # if user requested a plot
            current['xaxis'] = check_xaxis(current)  # check if custom xaxis can be plotted, else plot indices
            if current['xaxis'] is False:  # if default then plot indices
                plt_x = np.linspace(0, n - 1, n)  # indices
            elif isinstance(current['xaxis'], int):  # if user-defined but not a vector
                try:
                    normputs_np = np.array(normputs)  # in case list
                    min = self.minmax[current['xaxis']][0] # in clean
                    max = self.minmax[current['xaxis']][1]
                    plt_x = normputs_np[:, current['xaxis']] * (max - min) + min  # un-normalized vector for x-axis
                except:
                    warnings.warn(f"Keyword argument 'xaxis'={current['xaxis']} failed to index 'inputs'. Plotting indices instead.",
                                  category=UserWarning)
                    plt_x = np.linspace(0, n - 1, n)  # indices
            else:
                plt_x = current['xaxis']  # user provided vector for xaxis
            if current['plot'] == 'sorted':  # if user requested a sorted plot
                sort_id = np.argsort(np.squeeze(data))
                plt_mean = mean[sort_id]
                plt_bounds = bounds[sort_id]
                plt_data = data[sort_id]
            else:  # elif current['plot'] is True:
                plt_mean = mean
                plt_data = data
                plt_bounds = bounds
            plt.figure()
            plt.plot(plt_x, plt_mean, current['PlotTypeFoKL'], linewidth=current['PlotSizeFoKL'],
                     label=current['LegendLabelFoKL'])
            if data is not False:
                plt.plot(plt_x, plt_data, current['PlotTypeData'], markersize=current['PlotSizeData'],
                         label=current['LegendLabelData'])
            if current['bounds']:
                plt.plot(plt_x, plt_bounds[:, 0], current['PlotTypeBounds'], linewidth=current['PlotSizeBounds'],
                         label=current['LegendLabelBounds'])
                plt.plot(plt_x, plt_bounds[:, 1], current['PlotTypeBounds'], linewidth=current['PlotSizeBounds'])
            if current['labels']:
                if current['xlabel']:
                    plt.xlabel(current['xlabel'])
                if current['ylabel']:
                    plt.ylabel(current['ylabel'])
                if current['title']:
                    plt.title(current['title'])
            if current['legend']:
                plt.legend()
            plt.show()
        if data is not False:
            rmse = np.sqrt(np.mean(mean - data) ** 2)
        else:
            rmse = []
        return mean, bounds, rmse
    
    def evaluate(self, inputs=None, betas=None, mtx=None, **kwargs):
        """
        Evaluate the FoKL model for provided inputs and (optionally) calculate bounds. Note 'evaluate_fokl' may be a
        more accurate name so as not to confuse this function with 'evaluate_basis', but it is unexpected for a user to
        call 'evaluate_basis' so this function is simply named 'evaluate'.
        Input:
            inputs == input variable(s) at which to evaluate the FoKL model == self.inputs (default)
        Optional Inputs:
            betas        == coefficients defining FoKL model                       == self.betas (default)
            mtx          == interaction matrix defining FoKL model                 == self.mtx (default)
            minmax       == [min, max] of inputs used for normalization            == None (default)
            draws        == number of beta terms used                              == self.draws (default)
            clean        == boolean to automatically normalize and format 'inputs' == False (default)
            ReturnBounds == boolean to return confidence bounds as second output   == False (default)
        """
        # Process keywords:
        default = {'minmax': None, 'draws': self.config.DEFAULT['draws'], 'clean': False, 'ReturnBounds': False,  # for evaluate
                   '_suppress_normalization_warning': False}                                    # if called from coverage3
        default_for_clean = {'train': 1, 
                             # For '_format':
                             'AutoTranspose': True, 'SingleInstance': False, 'bit': 64,
                             # For '_normalize':
                             'normalize': True, 'minmax': None, 'pillow': None, 'pillow_type': 'percent'}
        current = process_kwargs(merge_dicts(default, default_for_clean), kwargs)
        for boolean in ['clean', 'ReturnBounds']:
            current[boolean] = str_to_bool(current[boolean])
        kwargs_to_clean = {}
        for kwarg in default_for_clean.keys():
            kwargs_to_clean.update({kwarg: current[kwarg]})  # store kwarg for clean here
            del current[kwarg]  # delete kwarg for clean from current
        if current['draws'] < 40 and current['ReturnBounds']:
            current['draws'] = 40
            warnings.warn("'draws' must be greater than or equal to 40 if calculating bounds. Setting 'draws=40'.")
        draws = current['draws']  # define local variable
        if betas is None:  # default
            if draws > self.fokl.betas.shape[0]:
                draws = self.fokl.betas.shape[0]  # more draws than models results in inf time, so threshold
                self.draws = draws
                warnings.warn("Updated attribute 'self.draws' to equal number of draws in 'self.betas'.",
                              category=UserWarning)
            betas = self.fokl.betas[-draws::, :]  # use samples from last models
        else:  # user-defined betas may need to be formatted
            betas = np.array(betas)
            if betas.ndim == 1:
                betas = betas[np.newaxis, :]  # note transpose would be column of beta0 terms, so not expected
            if draws > betas.shape[0]:
                draws = betas.shape[0]  # more draws than models results in inf time, so threshold
            betas = betas[-draws::, :]  # use samples from last models
        if mtx is None:  # default
            mtx = self.fokl.mtx
        else:  # user-defined mtx may need to be formatted
            if isinstance(mtx, int):
                mtx = [mtx]
            mtx = np.array(mtx)
            if mtx.ndim == 1:
                mtx = mtx[np.newaxis, :]
                warnings.warn("Assuming 'mtx' represents a single model. If meant to represent several models, then "
                              "explicitly enter a 2D numpy array where rows correspond to models.")
        phis = self.config.DEFAULT['phis']
        # Automatically normalize and format inputs:
        if inputs is None:  # default
            inputs = self.fokl.inputs
            if current['clean']:
                warnings.warn("Cleaning was already performed on default 'inputs', so overriding 'clean' to False.",
                              category=UserWarning)
                current['clean'] = False
        else:  # user-defined 'inputs'
            if not current['clean']:  # assume provided inputs are already formatted and normalized
                normputs = inputs
                if current['_suppress_normalization_warning'] is False:  # to suppress warning when evaluate called from coverage3
                    warnings.warn("User-provided 'inputs' but 'clean=False'. Subsequent errors may be solved by enabling automatic formatting and normalization of 'inputs' via 'clean=True'.", category=UserWarning)
        if current['clean']:
            normputs = self.dataFormat.clean(inputs, kwargs_from_other=kwargs_to_clean)
        elif inputs is None:
            normputs = self.inputs
        else:
            normputs = np.array(inputs)
        m, mbets = np.shape(betas)  # Size of betas
        n, mputs = np.shape(normputs)  # Size of normalized inputs
        setnos_p = np.random.randint(m, size=(1, draws))  # Random draws  from integer distribution
        i = 1
        while i == 1:
            setnos = np.unique(setnos_p)
            if np.size(setnos) == np.size(setnos_p):
                i = 0
            else:
                setnos_p = np.append(setnos, np.random.randint(m, size=(1, draws - np.shape(setnos)[0])))
        X = np.zeros((n, mbets))
        normputs = np.asarray(normputs)
        if self.config.KERNELS[0] == self.config.DEFAULT['kernel']:  # == 'Cubic Splines':
            _, phind, xsm = self.dataFormat.inputs_to_phind(normputs)  # ..., phis=self.phis, kernel=self.kernel) already true
        elif self.config.KERNELS[1] == self.config.DEFAULT['kernel']:  # == 'Bernoulli Polynomials':
            phind = None
            xsm = normputs
        for i in range(n):
            for j in range(1, mbets):
                phi = 1
                for k in range(mputs):
                    num = mtx[j - 1, k]
                    if num > 0:
                        nid = int(num - 1)
                        if self.config.KERNELS[0] == self.config.DEFAULT['kernel']:  # == 'Cubic Splines':
                            coeffs = [phis[nid][order][phind[i, k]] for order in range(4)]  # coefficients for cubic
                        elif self.config.KERNELS[1] == self.config.DEFAULT['kernel']:  # == 'Bernoulli Polynomials':
                            coeffs = phis[nid]  # coefficients for bernoulli
                        phi *= self.functions.evaluate_basis(coeffs, xsm[i, k])  # multiplies phi(x0)*phi(x1)*etc.
                X[i, j] = phi
        X[:, 0] = np.ones((n,))
        modells = np.zeros((n, draws))  # note n == np.shape(data)[0] if data != 'ignore'
        for i in range(draws):
            modells[:, i] = np.transpose(np.matmul(X, np.transpose(np.array(betas[setnos[i], :]))))
        mean = np.mean(modells, 1)
        if current['ReturnBounds']:
            bounds = np.zeros((n, 2))  # note n == np.shape(data)[0] if data != 'ignore'
            cut = int(np.floor(draws * 0.025) + 1)
            for i in range(n):  # note n == np.shape(data)[0] if data != 'ignore'
                drawset = np.sort(modells[i, :])
                bounds[i, 0] = drawset[cut]
                bounds[i, 1] = drawset[draws - cut]
            return mean, bounds
        else:
            return mean
        
    def GP_Integrate(betas, matrix, b, norms, phis, start, stop, y0, h, used_inputs):
        """
        betas is a list of arrays in which each entry to the list contains a specific row of the betas matrix,
        or the mean of the the betas matrix for each model being integrated

        matrix is a list of arrays containing the interaction matrix of each model

        b is an array of of the values of all the other inputs to the model(s) (including
        any forcing functions) over the time period we integrate over. The length of b
        should be equal to the number of points in the final time series (end-start)/h
        All values in b need to be normalized with respect to the min and max values
        of their respective values in the training dataset

        h is the step size with respect to time

        norms is a matrix of the min and max values of all the inputs being
        integrated (in the same order as y0). min values are in the top row, max values in the bottom.

        Start is the time at which integration begins. Stop is the time to
        end integration.

        y0 is an array of the inital conditions for the models being integrated

        Used inputs is a list of arrays containing the information as to what inputs
        are used in what model. Each array should contain a vector corresponding to a different model.
        Inputs should be referred to as those being integrated first, followed by
        those contained in b (in the same order as they appear in y0 and b
        respectively)
        For example, if two models were being integrated, with 3 other inputs total
        and the 1st model used both models outputs as inputs and the 1st and 3rd additional
        inputs, while the 2nd model used its own output as an input and the 2nd
        and 3rd additional inputs, used_inputs would be equal to
        [[1,1,1,0,1],[0,1,0,1,0]].
        If the models created do not follow this ordering scheme for their inputs
        the inputs can be rearranged based upon an alternate
        numbering scheme provided to used_inputs. E.g. if the inputs need to breordered the the 1st input should have a '1' in its place in the
        used_inputs vector, the 2nd input should have a '2' and so on. Using the
        same example as before, if the 1st models inputs needed rearranged so that
        the 3rd additional input came first, followed by the two model outputs in
        the same order as they are in y0, and ends with the 1st additional input,
        then the 1st cell in used_inputs would have the form [2,3,4,0,1]

        T an array of the time steps the models are integrated at.

        Y is an array of the models that have been integrated, at the time steps
        contained in T.
    """

        def prediction(inputs):
            f = []
            for kk in range(len(inputs)):
                if len(f) == 0:
                    f = [bss_eval(inputs[kk], betas[kk], phis, matrix[kk])]
                else:
                    f = np.append(f, bss_eval(inputs[kk], betas[kk], phis, matrix[kk]))
            return f

        def reorder(used, inputs):
            order = used[used != 0]
            reinputs = np.array((inputs.shape))
            for i in range(len(inputs)):
                reinputs[order[i] - 1] = inputs[i]
            return reinputs

        def normalize(v, minim, maxim):
            norm = np.zeros((1, 1))
            norm[0] = (v - minim) / (maxim - minim)
            if norm[0] > 1:
                norm[0] = 1
            if norm[0] < 0:
                norm[0] = 0
            return norm

        def bss_eval(x, betas, phis, mtx, Xin=[]):
            """
            x are normalized inputs
            betas are coefficients. If using 'Xin' include all betas (include constant beta)
        
            phis are the spline coefficients for the basis functions (cell array)
        
            mtx is the 'interaction matrix' -- a matrix each row of which corresponds
            to a term in the expansion, each column corresponds to an input. if the
            column is zero there's no corresponding basis function in the term; if
            it's greater than zero it corresponds to the order of the basis function
        
            Xin is an optional input of the chi matrix. If this was pre-computed with xBuild,
            one may use it to improve performance.
            """
        
            if Xin == []:
                m, n = np.shape(mtx)  # getting dimensions of the matrix 'mtx'
        
                mx = 1
        
                mbet = 1
        
                delta = np.zeros([mx, mbet])
        
                phind = []
        
                for j in range(len(x)):
                    phind.append(math.floor(x[j] * 498))
        
                phind_logic = []
                for k in range(len(phind)):
                    if phind[k] == 498:
                        phind_logic.append(1)
                    else:
                        phind_logic.append(0)
        
                phind = np.subtract(phind, phind_logic)
        
                r = 1 / 498
                xmin = r * np.array(phind)
                X = (x - xmin) / r
                for ii in range(mx):
                    for i in range(m):
                        phi = 1
        
                        for j in range(n):
                            
                            num = mtx[i][j]
        
                            if num != 0:
                                phi = phi * (phis[int(num) - 1][0][phind[j]] + phis[int(num) - 1][1][phind[j]] * X[j] \
                                             + phis[int(num) - 1][2][phind[j]] * X[j] ** 2 + phis[int(num) - 1][3][
                                                 phind[j]] *
                                             X[j] ** 3)
        
                        delta[ii, :] = delta[ii, :] + betas[i + 1] * phi
                        mmm = 1
                delta[ii, :] = delta[ii, :] + betas[0]
            else:
                if np.ndim(betas) == 1:
                    betas = np.array([betas])
                elif np.ndim(betas) > 2:
                    print("The \'betas\' parameter has %d dimensions, but needs to have only 2." % (np.ndim(betas)))
                    print("The current shape is:", np.shape(betas))
                    print("Attempting to get rid of unnecessary dimensions of size 1...")
                    betas = np.squeeze(betas)
        
                    if np.ndim(betas) == 1:
                        betas = np.array([betas])
                        print("Success! New shape is", np.shape(betas))
                    elif np.ndim(betas) == 2:
                        print("Success! New shape is", np.shape(betas))
        
                delta = Xin.dot(betas.T)
            dc = 1
            return delta
        
        T = np.arange(start, stop + h, h)
        y = y0
        Y = np.array([y0])
        Y = Y.reshape(len(y0), 1)
        
        ind = 1
        for t in range(len(T) - 1):
            inputs1 = list()
            othinputs = list()
            inputs2 = list()
            inputs3 = list()
            inputs4 = list()
            for i in range(len(y)):  # initialize inputs1 and othinputs to contain empty arrays
                inputs1.append([])
                othinputs.append([])
                inputs2.append([])
                inputs3.append([])
                inputs4.append([])
            for i in range(len(y)):
                for j in range(len(y)):
                    if used_inputs[i][j] != 0:
                        if len(inputs1[i]) == 0:
                            inputs1[i] = normalize(y[j], norms[0, j], norms[1, j])
                        else:
                            inputs1[i] = np.append(inputs1[i], normalize(y[j], norms[0, j], norms[1, j]), 1)
        
            nnn = int(b.size / b.shape[0])
            if b.size > 0:
                for ii in range(len(y0)):
                    for jj in range(len(y), nnn + len(y)):
                        
                        if used_inputs[ii][jj] != 0:
                            if len(othinputs[ii]) == 0:
                                if ind - 1 == 0:
                                    othinputs[ii] = b[ind - 1]
                                else:
                                    
                                    othinputs[ii] = b[ind - 1]
        
                                    ttt = 1
                            else:
                                othinputs[ii] = np.append(othinputs[ii], b[ind - 1, jj - len(y0)], 1)
                for k in range(len(y)):
                    inputs1[k] = np.append(inputs1[k], othinputs[k])
            for ii in range(len(y0)):
                if np.amax(used_inputs[ii]) > 1:
                    inputs1[ii] = reorder(used_inputs[ii], inputs1[ii])
        
            dy1 = prediction(inputs1) * h
        
            for p in range(len(y)):
                if y[p] >= norms[1, p] and dy1[p] > 0:
                    dy1[p] = 0
                else:
                    if y[p] <= norms[0, p] and dy1[p] < 0:
                        dy1[p] = 0
        
            for i in range(len(y)):
                for j in range(len(y)):
                    if used_inputs[i][j] != 0:
                        if len(inputs2[i]) == 0:
                            inputs2[i] = normalize(y[j] + dy1[j] / 2, norms[0, j], norms[1, j])
                        else:
                            inputs2[i] = np.append(inputs2[i], normalize(y[j] + dy1[j] / 2, norms[0, j], norms[1, j]),
                                                   1)
        
            for k in range(len(y)):
                inputs2[k] = np.append(inputs2[k], othinputs[k])
            for ii in range(len(y0)):
                if np.amax(used_inputs[ii]) > 1:
                    inputs2[ii] = reorder(used_inputs[ii], inputs2[ii])
            dy2 = prediction(inputs2) * h
            for p in range(len(y)):
                if (y[p] + dy1[p] / 2) >= norms[1, p] and dy2[p] > 0:
                    dy2[p] = 0
                if (y[p] + dy1[p] / 2) <= norms[0, p] and dy2[p] < 0:
                    dy2[p] = 0
        
            for i in range(len(y)):
                for j in range(len(y)):
                    if used_inputs[i][j] != 0:
                        if len(inputs3[i]) == 0:
                            inputs3[i] = normalize(y[j] + dy2[j] / 2, norms[0, j], norms[1, j])
                        else:
                            inputs3[i] = np.append(inputs3[i], normalize(y[j] + dy2[j] / 2, norms[0, j], norms[1, j]),
                                                   1)
            if b.size > 0:
                for k in range(len(y)):
                    inputs3[k] = np.append(inputs3[k], othinputs[k])
            for ii in range(len(y0)):
                if np.amax(used_inputs[ii]) > 1:
                    inputs3[ii] = reorder(used_inputs[ii], inputs3[ii])
            dy3 = prediction(inputs3) * h
            for p in range(len(y)):
                if (y[p] + dy2[p] / 2) >= norms[1, p] and dy3[p] > 0:
                    dy3[p] = 0
                if (y[p] + dy2[p] / 2) <= norms[0, p] and dy3[p] < 0:
                    dy3[p] = 0
        
            for i in range(len(y)):
                for j in range(len(y)):
                    if used_inputs[i][j] != 0:
                        if len(inputs4[i]) == 0:
                            inputs4[i] = normalize(y[j] + dy3[j], norms[0, j], norms[1, j])
                        else:
                            inputs4[i] = np.append(inputs4[i], normalize(y[j] + dy3[j], norms[0, j], norms[1, j]), 1)
            if b.size > 0:
                for k in range(len(y)):
                    inputs4[k] = np.append(inputs4[k], othinputs[k])
            for ii in range(len(y0)):
                if np.amax(used_inputs[ii]) > 1:
                    inputs4[ii] = reorder(used_inputs[ii], inputs4[ii])
            dy4 = prediction(inputs4) * h
            for p in range(len(y)):
                if (y[p] + dy3[p]) >= norms[1, p] and dy4[p] > 0:
                    dy4[p] = 0
                if (y[p] + dy3[p]) <= norms[0, p] and dy4[p] < 0:
                    dy4[p] = 0
        
            y += (dy1 + 2 * dy2 + 2 * dy3 + dy4) / 6
            yt = np.reshape(y, [2, 1])
            Y = np.append(Y, yt, 1)
            ind += 1
        
            return T, Y