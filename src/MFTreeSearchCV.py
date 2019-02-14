# Author: Rajat Sen

# This is the main source file that implements the methods MFTreeSearchCV


from sklearn.model_selection import BaseSearchCV

class MFTreeSearchCV():
	"""Multi-Fidelity  Tree Search over specified parameter ranges for an estimator.
	Important members are fit, predict.
	MFTreeSearchCV implements a "fit" and a "score" method.
	It also implements "predict", "predict_proba"
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
		'kernel' : {'range': [ ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’], 'type': 'cat'}, \
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
	cv_results_ :
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

	
	"""

	def __init__(self, estimator, param_dict, scoring='accuracy', greater_is_better = True, fixed_params = None,\
				 n_jobs=1, refit=True, cv = 3, debug = True, n_jobs = 1, nu_max = 1.0, rho_max = 0.95, sigma = 0.02, C = 1.0, tol = 1e-3, \
				 Randomize = False, Auto = True, unit_cost = 1.0,\
				 fidelity_range,total_budget):

		self.base_estimator = estimator 
		self.param_dict = param_dict
		self.scoring = scoring
		self.greater_is_better = greater_is_better
		self.fixed_params = fixed_params
		self.n_jobs = self.n_jobs
		self.fidelity_range = fidelity_range
		self.refit = refit
		self.cv = cv 
		self.debug = debug 
		self.nu_max = nu_max
		self.rho_max = rho_max
		self.sigma = sigma
		self.C = C
		self.tol = tol 
		self.fidelity_range = fidelity_range
		self.total_budget = total_budget



	def _create_mfobject(self):



	def _run_tree_search(self):



	def _populate_cv_results(self):


	def _refit(self):


	def predict(self,X):


	def predict_proba(self,X):


	def fit(self,X,y):
		










		

