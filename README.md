# MFTreeSearchCV

This is a package for fast hyper-parameter tuning using noisy multi-fidelity tree-search for scikit-learn estimators (classifiers/regressors). Given ranges and types (categorical, integer, reals) for several hyper-parameters, this package is desgined to search for a good configuartion by treating the k-fold cross-validation errors and different setting under different fidelities (levels of approximation based on amount od data used), as a multi-fidelity noisy black-box function. This work is based on the publications:

1. [A deterministic version of MF Tree Seach](http://proceedings.mlr.press/v80/sen18a/sen18a.pdf)
2. [A version that can hadle noise -- which is there in tuning](https://arxiv.org/pdf/1810.10482)

Please cite the above papers, if using this software in any publications/reports. 


### Prerequisites

1. You will need a C and a Fortran compiler such as gnu95. Please install them by following the correct steps for your machine and OS. 

2. You will need the following python packages: sklearn, numpy, brewer2mpl, scipy, 

### Installing

1. Go to the location of your choice and clone the repo.

```
git clone https://github.com/rajatsen91/MFTreeSearchCV.git
```

2. Now the next steps are to setup the multi-fidelity enviroment. 
- We will assume that the location of the project is `/home/MFTreeSearchCV/`
- You first need to build the direct fortran library. For this `cd` into
  `/home/MFTreeSearchCV/utils/direct_fortran` and run `bash make_direct.sh`. You will need a fortran compiler
  such as gnu95. Once this is done, you can run `simple_direct_test.py` to make sure that
  it was installed correctly.
- Edit the `set_up_gittins` file to change the `GITTINS_PATH` to the local location of the repo. 
```
GITTINS_PATH=/home/MFTreeSearchCV
``` 
- Run `source set_up_gittins` to set up all environment variables.
- To test the installation, run `bash run_all_tests.sh`. Some of the tests are
  probabilistic and could fail at times. Some tests that require a scratch directory will fail,
  but nothing to worry about. 


### Usage 

The usage of this repository is designed to be similar to parameter tuning functions in sklearn.model_selection, like `gridsearchcv`. The main function is `MFTreeSearchCV.MFTreeSearchCV`. The arguments and methods of this function are as follows,

```
	"""Multi-Fidelity  Tree Search over specified parameter ranges for an estimator.
	Important members are fit, predict.
	MFTreeSearchCV implements a "fit" and a "score" method.
	It also implements "predict", "predict_proba" is they are present in the base-estimator.
	The parameters of the estimator used to apply these methods are optimized
	by cross-validated Tree Search over a parameter search space.
	----------
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

	nu_max : automatically set, but can be give a default values in the range (0,2]
	rho_max : rho_max in the paper. Default value is 0.95 and is recommended
	sigma : sigma in the paper. Default value is 0.02, adjust according to the believed noise standard deviation in the system
	C : default is 1.0, which is overwritten if Auto = True, which is the recommended setting
	Auto : If True then the bias function parameter C is auto set. This is recommended. 
	tol : default values is 1e-3. All fidelities z_1, z_2 such that |z_1 - z_2| < tol are assumed to yield the same bias value

	total_budget : total budget for the search in seconds. This includes the time for automatic parameter C selection and does not include refit time. 
	total_budget should ideally be more than 5X the unit_cost which is the time taken to run one experiment at the highest fidelity
	
	unit_cost : time in seconds required to fit the base estimator at the highest fidelity. This should be estimated by the user and then supplied. 
	
	Attributes
	----------
	cv_results_ : dictionary showing the scores attained under a few parameters setting. Each
	parameter setting is the best parameter obtained from a tree-search call. 
	best_estimator_ : estimator or dict
		Estimator that was chosen by the search, i.e. estimator
		which gave highest score (or smallest loss if specified)
		on the left out data. Not available if ``refit=False``.
		See ``refit`` parameter for more information on allowed values.
	best_score_ : float
		Mean cross-validated score of the best_estimator
	best_params_ : dict
		Parameter setting that gave the best results on the hold out data.
	refit_time_ : float
		Seconds used for refitting the best model on the whole dataset.
		This is present only if ``refit`` is not False.
	fit_time_ : float
		Seconds taken to find the best parameters. Should be close to the budget given. 
	

	
	"""

```

A functional example is provided in the ipython notebook `Illustrate.ipynb`. 


### Example

```python

estimator = LogisticRegression() #base estimator
param_dict = {'C':{'range':[1e-5,1e5],'scale':'log','type':'real'},\
              'penalty':{'range':['l1','l2'],'scale':'linear','type':'cat'}} #parameter space
fidelity_range = [500,15076] # fidelity range, lowest fidelity uses 500 samples while the highest one uses 
#the whole dataset  
n_jobs = 3 # number of jobs
cv = 3 # cv level
fixed_params = {}
scoring = 'accuracy'

t1 = time.time()
estimator = estimator.fit(X_train,y_train)
t2 = time.time()
total_budget = 500 # total budget in seconds
print('Time without CV: ', t2 - t1)

model = MFTreeSearchCV(estimator=estimator,param_dict=param_dict,scoring=scoring,\
                      fidelity_range=fidelity_range,unit_cost=None,\
                    cv=cv,  n_jobs = n_jobs,total_budget=total_budget,debug = True,fixed_params=fixed_params)

## running in debug mode will display certain outputs


m = model.fit(X_train,y_train) # actual tree search will be done here 

y_pred = m.predict(X_test) # the returned model m will have the best parameters fitted as 'refit = True'

accuracy = accuracy_score(y_pred,y_test) # this is the best accuracy obtained

print(m.best_params_) # will print the best params 

print(m.cv_results_) # will print cv scores for some other parameters as well, that were close




```



## Authors

* **Rajat Sen** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* We acknowledge the support from [@kirthevasank](https://github.com/kirthevasank) for providing the original multi-fidelity black-box function code base.
