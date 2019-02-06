import numpy as np
import Queue
from mf.mf_func import MFOptFunction
from utils.general_utils import map_to_cube
import sys
from examples.synthetic_functions import *
from mf.mf_func import get_noisy_mfof_from_mfof
from letters.letters_classifier import *
import time

from MFTree.MFHOO import *

import synthetic_functions


NUM_EXP = 5
#EXP_NAME = 'Branin'
#EXP_NAME = 'CurrinExp'
#EXP_NAME = 'Hartmann3'
#EXP_NAME = 'Hartmann6'
EXP_NAME = 'Borehole'


def run_one_experiment(mfobject,nu,rho,times,sigma,C,t0,filname):
	R = []
	T = []
	for t in times:
		budget = t*mfobject.opt_fidel_cost
		t1 = time.time()
		MP = MFPOO(mfobject=mfobject, nu_max=nu, rho_max=rho, total_budget=budget, sigma=sigma, C=C, mult=0.5, tol = 1e-3, Randomize = False, Auto = True, unit_cost=t0 )
		MP.run_all_MFHOO()
		X, E = MP.get_point()
		t2 = time.time()	

		R = R + [E]
		T = T + [MP.cost]
		print str(MP.cost) + ' : ' + str(E)
		#print 'Total HOO Queries: ' + str(MP.t) 

	np.save(filename,R)
	return np.array(R),np.array(T)


if __name__ == '__main__':


	if EXP_NAME == 'Hartmann3':
		mfof = synthetic_functions.get_mf_hartmann_as_mfof(1, 3)
		noise_var = 0.01
		sigma = np.sqrt(noise_var)

	elif EXP_NAME == 'Hartmann6':
		mfof = synthetic_functions.get_mf_hartmann_as_mfof(1, 6)
		max_capital = 200 * mfof.opt_fidel_cost
		noise_var = 0.05
		sigma = np.sqrt(noise_var)
	elif EXP_NAME == 'CurrinExp':
		mfof = synthetic_functions.get_mf_currin_exp_as_mfof()
		max_capital = 200 * mfof.opt_fidel_cost
		noise_var = 0.5
		sigma = np.sqrt(noise_var)
	elif EXP_NAME == 'Branin':
		mfof = synthetic_functions.get_mf_branin_as_mfof(1)
		max_capital = 200 * mfof.opt_fidel_cost
		noise_var = 0.05
		sigma = np.sqrt(noise_var)
	elif EXP_NAME == 'Borehole':
		mfof = synthetic_functions.get_mf_borehole_as_mfof()
		max_capital = 200 * mfof.opt_fidel_cost
		noise_var = 5
		sigma = np.sqrt(noise_var)


	times = [10,20,50,75,100,150,175, 200]
	mfobject = get_noisy_mfof_from_mfof(mfof, noise_var)
	nu = 1.0
	rho = 0.95
	C = 0.1
	t0 = mfobject.opt_fidel_cost

	NT = str(time.time())
	print 'Running Experiment 1: '
	filename = 'MFHOO' + EXP_NAME + '_' + NT + '_' + '1.npy'
	R,T = run_one_experiment(mfobject,nu,rho,times,sigma,C,t0,filename)
	result = R

	for i in range(1,NUM_EXP):
		print 'Running Experiment' + str(i+1) + ': '
		filename = 'MFHOO' + EXP_NAME + '_' + NT + '_' + str(i+1) + '.npy'
		R,T = run_one_experiment(mfobject,nu,rho,times,sigma,C,t0,filename)
		result = np.vstack([result,R])


	mu = np.mean(result,axis = 0)
	std = np.std(result,axis = 0)
	result = mfobject.opt_val - mu
	filename = './examples/results/MFHOO_' + EXP_NAME + '_' + NT + '_' + '.csv'
	dfdic = {}
	dfdic['Capital'] = np.array(times)
	dfdic['Value'] = result
	dfdic['Std'] = std
	df = pd.DataFrame(dfdic)
	df.to_csv(filename) 




