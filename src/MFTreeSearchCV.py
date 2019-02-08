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
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.
        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.
        See :ref:`multimetric_grid_search` for an example.
        If None, the estimator's default scorer (if available) is used.
    fixed_params: dictionary of parameter values other than the once in param_dict, that should be held fixed at the supplied value.
    For example, if fixed_params = {'nthread': 10} is passed with estimator as XGBoost, it means that all
    XGBoost instances will be run with 10 parallel threads
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
    degub : Binary
        Controls the verbosity: True means more messages, while False only prints critical messages
    refit : True means the best parameters are fit into an estimator and trained, while False means the best_estimator is not refit

    fidelity_range : range of fidelity to use. It is a tuple (a,b) which means lowest fidelity means a samples are used for training and 
    validation and b samples are used when fidelity is the highest. We recommend setting b to be the total number of training samples
    available and a to bea reasonable value. 
    
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
        For multi-metric evaluation, this is present only if ``refit`` is
        specified.
    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.
        This is present only if ``refit`` is not False.
    Notes
    ------
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.
    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.
    See Also
    ---------
    :class:`ParameterGrid`:
        generates all the combinations of a hyperparameter grid.
    :func:`sklearn.model_selection.train_test_split`:
        utility function to split the data into a development set usable
        for fitting a GridSearchCV instance and an evaluation set for
        its final evaluation.
    :func:`sklearn.metrics.make_scorer`:
        Make a scorer from a performance metric or loss function.
    """

    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=None, iid='warn', refit=True, cv='warn', verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise-deprecating',
                 return_train_score="warn"):
        super(GridSearchCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=fit_params,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))
