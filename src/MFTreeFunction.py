#Author: Rajat Sen
# general MF function object for doing tree search on scikit-learn classifier/regressor object



from __future__ import print_function
from __future__ import division


import numpy as np 
from sklearn.metrics import *
from src.converters import *
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
	z = x.copy()   # start with x's keys and values
	z.update(y)    # modifies z with y's keys and values & returns None
	return z


class MFTreeFunction(MFOptFunction):
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
