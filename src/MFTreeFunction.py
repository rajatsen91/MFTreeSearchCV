#Author: Rajat Sen
# general MF function object for doing tree search on scikit-learn classifier/regressor object


import numpy as np 
from sklearn.metrics import *
from converters import *
from sklearn.model_selection import cross_val_score

from copy import deepcopy
from scipy.stats import norm
from scipy import integrate
# Local imports
from mf.mf_func import MFOptFunction
import warnings
from sklearn.model_selection import cross_val_score
import pandas as pd


'''
Scoring Functions:
Scoring	Function	Comment
Classification	 	 
‘accuracy’	metrics.accuracy_score	 
‘balanced_accuracy’	metrics.balanced_accuracy_score	for binary targets
‘average_precision’	metrics.average_precision_score	 
‘brier_score_loss’	metrics.brier_score_loss	 
‘f1’	metrics.f1_score	for binary targets
‘f1_micro’	metrics.f1_score	micro-averaged
‘f1_macro’	metrics.f1_score	macro-averaged
‘f1_weighted’	metrics.f1_score	weighted average
‘f1_samples’	metrics.f1_score	by multilabel sample
‘neg_log_loss’	metrics.log_loss	requires predict_proba support
‘precision’ etc.	metrics.precision_score	suffixes apply as with ‘f1’
‘recall’ etc.	metrics.recall_score	suffixes apply as with ‘f1’
‘roc_auc’	metrics.roc_auc_score	 
 	 
Regression	 	 
‘explained_variance’	metrics.explained_variance_score	 
‘neg_mean_absolute_error’	metrics.mean_absolute_error	 
‘neg_mean_squared_error’	metrics.mean_squared_error	 
‘neg_mean_squared_log_error’	metrics.mean_squared_log_error	 
‘neg_median_absolute_error’	metrics.median_absolute_error	 
‘r2’	metrics.r2_score
'''

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
	def __init__(self, X,y,estimator, param_dict, scoring='accuracy', greater_is_better = True, fixed_params = None,\
				 n_jobs=1, cv = 3, n_jobs = 1, \
				 fidelity_range):

		self.base_estimator = estimator 
		self.param_dict = param_dict
		self.scoring = scoring
		self.fixed_params = fixed_params
		self.n_jobs = self.n_jobs
		self.fidelity_range = fidelity_range
		self.cv = cv
		self.fidelity_range = fidelity_range
		self.X = X
		self.y = y

		self.scorer = return_scoring_function(self.scoring)
		self.problem_bounds, self.keys = convert_dict_to_bounds(self.param_dict)
		self.max_data = self.fidelity_range[1]
		mf_func = self._mf_func
		fidel_cost_func = self._fidel_cost
		domain_bounds = np.array(self.problem_bounds)
		opt_fidel_unnormalised = np.array([self.max_data])
		super(MFTreeFunction, self).__init__(mf_func, fidel_cost_func, fidel_bounds,
											  domain_bounds, opt_fidel_unnormalised,
											  vectorised=False)

	def _fidel_cost(self, z):
	""" cost function """
		return 0.01 + (z[0]/self.max_data)


	def _mf_func(self, z, x):
		pgrid = convert_values_to_dict(list(x),self.problem_bounds,self.keys, self.param_dict)
		grid = merge_two_dicts(pgrid,self.fixed_params)
		gbm.set_params(**grid)
		num_data_curr = int(z[0])
		feat_curr = self.X[1:num_data_curr]
		label_curr = self.y[1:num_data_curr]
		return get_kfold_val_score(gbm, feat_curr, label_curr)

	def get_kfold_val_score(clf, X, Y, num_folds=None,random_seed = 512):
		st0 = np.random.get_state()
		np.random.seed(random_seed)
		num_folds = self.cv
		acc = cross_val_score(clf,X = X,y = Y,cv=num_folds,n_jobs=self.n_jobs,scoring=self.scoring)
		np.random.set_state(st0)
		if self.greater_is_better:
			return acc.mean()
		else:
			return -acc.mean()
