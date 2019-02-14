#Author: Rajat Sen
# general MF function object for doing tree search on scikit-learn classifier/regressor object


import numpy as np 
from sklearn.metrics import *


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
Clustering	 	 
‘adjusted_mutual_info_score’	metrics.adjusted_mutual_info_score	 
‘adjusted_rand_score’	metrics.adjusted_rand_score	 
‘completeness_score’	metrics.completeness_score	 
‘fowlkes_mallows_score’	metrics.fowlkes_mallows_score	 
‘homogeneity_score’	metrics.homogeneity_score	 
‘mutual_info_score’	metrics.mutual_info_score	 
‘normalized_mutual_info_score’	metrics.normalized_mutual_info_score	 
‘v_measure_score’	metrics.v_measure_score	 
Regression	 	 
‘explained_variance’	metrics.explained_variance_score	 
‘neg_mean_absolute_error’	metrics.mean_absolute_error	 
‘neg_mean_squared_error’	metrics.mean_squared_error	 
‘neg_mean_squared_log_error’	metrics.mean_squared_log_error	 
‘neg_median_absolute_error’	metrics.median_absolute_error	 
‘r2’	metrics.r2_score
'''

def return_scoring_function(tag,greater_is_better):
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

	if greater_is_better:
		return f
	else:
		return -f 