#Author: Rajat Sen
# general MF function object for doing tree search on scikit-learn classifier/regressor object



from __future__ import print_function
from __future__ import division

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
	sys.path.append(module_path)

import numpy as np 
from sklearn.metrics import *
from MFTreeSearchCV.converters import *
from sklearn.model_selection import cross_val_score

from copy import deepcopy
from scipy.stats import norm
from scipy import integrate
# Local imports
from mf.mf_func import MFOptFunction
import warnings
from sklearn.model_selection import cross_val_score
import pandas as pd






def return_scoring_function(tag):
	'''
	Given a scoring tag like 'accuracy' returns the 
	corresponding scoring function. For example given
	the string 'accuracy', this will return the function accuracy_score
	from sklearn.model_selection
	'''
	if tag == 'accuracy':
		f = accuracy_score
	elif tag == 'balanced_accuracy':
		f = balanced_accuracy_score
	elif tag == 'average_precision':
		f = average_precision_score
	elif tag == 'brier_score_loss':
		f = brier_score_loss
	elif tag == 'f1':
		f = f1_score
	elif tag == 'neg_log_loss':
		f = log_loss
	elif tag == 'precision':
		f = precision_score
	elif tag == 'recall':
		f = recall_score
	elif tag == 'roc_auc':
		f = roc_auc_score

	elif tag == 'explained_variance':
		f = explained_variance_score
	elif tag == 'neg_mean_absolute_error':
		f = mean_absolute_error
	elif tag == 'neg_mean_squared_error':
		f = mean_squared_error
	elif tag == 'neg_mean_squared_log_error':
		f = mean_squared_log_error
	elif tag == 'neg_median_absolute_error':
		f = median_absolute_error	 
	elif tag == 'r2':
		f = r2_score
	else:
		raise ValueError('Unrecognized scorer tag!')

	return f

def merge_two_dicts(x, y):
	'''
	merges the two disctionaries x and y and returns the merged dictionary
	'''
	z = x.copy()   # start with x's keys and values
	z.update(y)    # modifies z with y's keys and values & returns None
	return z


class MFTreeFunction(MFOptFunction):
	'''
	A multi-fidelity function class which can be queried at 'x' at different 
	fidelity levels 'z in [0,1]'.
	----------
	X: training data features
	y: training laabel features
	estimator : estimator object.
		This is assumed to implement the scikit-learn estimator interface.
		Unlike grid search CV, estimator need not provide a ``score`` function.
		Therefore ``scoring`` must be passed. 
	param_dict : Dictionary with parameters names (string) as keys and and the value is another dictionary. The value dictionary has
	the keys 'range' that specifies the range of the hyper-parameter, 'type': 'int' or 'cat' or 'real' (integere, categorical or real),
	and 'scale': 'linear' or 'log' specifying whether the search is done on a linear scale or a logarithmic scale. An example for param_dict
	for scikit-learn SVC is as follows:
		eg: param_dict = {'C' : {'range': [1e-2,1e2], 'type': 'real', 'scale': 'log'}, \
		'kernel' : {'range': [ 'linear', 'poly', 'rbf', 'sigmoid'], 'type': 'cat'}, \
		'degree' : {'range': [3,10], 'type': 'int', 'scale': 'linear'}}
	scoring : string, callable, list/tuple, dict or None, default: None
		A single string (see :ref:`scoring_parameter`). this must be specified as a string. See scikit-learn metrics 
		for more details. 
	fixed_params: dictionary of parameter values other than the once in param_dict, that should be held fixed at the supplied value.
	For example, if fixed_params = {'nthread': 10} is passed with estimator as XGBoost, it means that all
	XGBoost instances will be run with 10 parallel threads
	cv : int, cross-validation generator or an iterable, optional
		Determines the cross-validation splitting strategy.
		Possible inputs for cv are:
		- None, to use the default 3-fold cross validation,
		- integer, to specify the number of folds in a `(Stratified)KFold`,
	debug : Binary
		Controls the verbosity: True means more messages, while False only prints critical messages
	refit : True means the best parameters are fit into an estimator and trained, while False means the best_estimator is not refit

	fidelity_range : range of fidelity to use. It is a tuple (a,b) which means lowest fidelity means a samples are used for training and 
	validation and b samples are used when fidelity is the highest. We recommend setting b to be the total number of training samples
	available and a to bea reasonable value. 
	
	n_jobs : number of parallel runs for the CV. Note that njobs * (number of threads used in the estimator) must be less than the number of threads 
	allowed in your machine. default value is 1. 
	
	Attributes and functions
	----------
	_mf_func : returns the value of the function at point 'x' evaluated at fidelity 'z'
	For other methods see the specifications in mf/mf_func. 

	
	'''
	def __init__(self, X,y,estimator, param_dict,fidelity_range, \
		scoring='accuracy', greater_is_better = True, fixed_params = {},\
				 n_jobs=1, cv = 3):

		self.base_estimator = estimator 
		self.param_dict = param_dict
		self.scoring = scoring
		self.fixed_params = fixed_params
		self.n_jobs = n_jobs
		self.fidelity_range = fidelity_range
		self.cv = cv
		self.fidelity_range = fidelity_range
		self.X = X
		self.y = y
		self.greater_is_better = greater_is_better

		self.scorer = return_scoring_function(self.scoring)
		self.problem_bounds, self.keys = convert_dict_to_bounds(self.param_dict)
		self.max_data = self.fidelity_range[1]
		mf_func = self._mf_func
		fidel_cost_func = self._fidel_cost
		fidel_bounds = np.array([self.fidelity_range])
		domain_bounds = np.array(self.problem_bounds)
		opt_fidel_unnormalised = np.array([self.max_data])
		super(MFTreeFunction, self).__init__(mf_func, fidel_cost_func, fidel_bounds,
											  domain_bounds, opt_fidel_unnormalised,
											  vectorised=False)

	def _fidel_cost(self, z):
		return 0.01 + (float(z[0])/self.max_data)


	def _mf_func(self, z, x):
		pgrid = convert_values_to_dict(list(x),self.problem_bounds,self.keys, self.param_dict)
		grid = merge_two_dicts(pgrid,self.fixed_params)
		gbm = self.base_estimator
		gbm.set_params(**grid)
		r,c = self.X.shape
		num_data_curr = int(z[0])
		inds = np.random.choice(r,num_data_curr)
		feat_curr = self.X[inds]
		label_curr = self.y[inds]
		return self.get_kfold_val_score(gbm, feat_curr, label_curr)

	def get_kfold_val_score(self,clf, X, Y, num_folds=None,random_seed = 512):
		st0 = np.random.get_state()
		if random_seed is None:
			np.random.seed()
		else:
			np.random.seed(random_seed)
		num_folds = self.cv
		acc = cross_val_score(clf,X = X,y = Y,cv=num_folds,n_jobs=self.n_jobs,scoring=self.scoring)
		np.random.set_state(st0)
		if self.greater_is_better:
			return acc.mean()
		else:
			return -acc.mean()
