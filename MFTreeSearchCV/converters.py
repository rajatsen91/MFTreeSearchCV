# Author: Rajat Sen
### This file is meant for functions that interchange various methods of representing the range and values for various hyper-parameters. 




from __future__ import print_function
from __future__ import division
import numpy as np

def convert_dict_to_bounds(param_dict):
	'''
	convert param_dict to list of parameters
	Returned values:---------------------
	problem_bounds: list of ranges for different parameters of dimensions
	keys: list of keys in the same order as problem bounds
	'''
	problem_bounds = []
	keys = []
	for key in param_dict:
		param = param_dict[key]
		if param['type'] == 'cat':
			bound = [0,1]
			scale = 'linear'
		else:
			if param['scale'] == 'linear':
				bound = param['range']
			else:
				bound = [np.log(param['range'][0]),np.log(param['range'][1])]
			scale = param['scale']


		problem_bounds = problem_bounds + [bound]
		keys = keys + [key]

	return problem_bounds,keys


def indexify(v,r):
	'''
	Helper Function
	'''
	for i in range(r):
		if float(i)/r <= v < float(i+1)/r:
			return i
		else:
			continue

	return r-1





def convert_values_to_dict(values,problem_bounds,keys, param_dict):
	'''
	Function to convert a vector of values for different hyper-parameters to a dict
	that can be used to set parameters of the base estimator object
	'''
	vdict = {}
	n = len(values)
	for i in range(n):
		v = values[i]
		k = keys[i]
		param = param_dict[k]

		if param['type'] == 'cat':
			r = len(param['range'])
			index = indexify(v,r)
			vdict[k] = param['range'][index]
		else:
			if param['scale'] == 'log':
				nv = np.exp(v)
			else:
				nv = v

			if param['type'] == 'int':
				nv = int(nv)

			vdict[k] = nv

	return vdict





