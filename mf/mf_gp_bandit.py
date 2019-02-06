"""
  Implements Multi-fidelity GP Bandit Optimisaiton.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=super-on-old-class

from argparse import Namespace
from copy import deepcopy
import time
import numpy as np

# Local imports
from mf_func import MFOptFunction
from mf_gp import all_mf_gp_args, MFGPFitter
from mf_gpb_utils import acquisitions, fidelity_choosers
from mf_gpb_utils import is_an_opt_fidel_query, latin_hc_sampling
from utils.optimisers import direct_ft_maximise, random_maximise
from utils.option_handler import get_option_specs, load_options
from utils.reporters import get_reporter

mf_gp_bandit_args = [
  get_option_specs('capital_type', False, 'given',
    ('The type of capital to be used. If \'given\', it will use the cost specified. '
     'Could be one of given, cputime, or realtime')),
  get_option_specs('max_iters', False, 1e5,
    'The maximum number of iterations, regardless of capital.'),
  get_option_specs('gamma_0', False, '1',
    ('The multiplier in front of the default threshold value for switching. Should be',
     'a scalar or the string \'adapt\'.')),
  get_option_specs('acq', False, 'mf_gp_ucb',
    'Which acquisition to use. Should be one of mf_gp_ucb, gp_ucb or gp_ei'),
  get_option_specs('acq_opt_criterion', False, 'rand',
    'Which optimiser to use when maximising the acquisition function.'),
  get_option_specs('acq_opt_max_evals', False, -1,
    'Number of evaluations when maximising acquisition. If negative uses default value.'),
  get_option_specs('gpb_init', False, 'random_lower_fidels',
    'How to initialise. Should be either random_lower_fidels or random.'),
  get_option_specs('gpb_init_capital', False, -1.0,
    ('The amount of capital to be used for initialisation. If negative, will use',
     'init_capital_frac fraction of the capital for optimisation.')),
  get_option_specs('gpb_init_capital_frac', False, 0.1,
    'The percentage of the capital to use for initialisation.'),
  # The following are perhaps not so important.
  get_option_specs('shrink_kernel_with_time', False, 1,
    'If True, shrinks the kernel with time so that we don\'t get stuck.'),
  get_option_specs('perturb_thresh', False, 1e-4,
    ('If the next point chosen is too close to an exisiting point by this times the '
     'diameter, then we will perturb the point a little bit before querying. This is '
     'mainly to avoid numerical stability issues.')),
  get_option_specs('build_new_gp_every', False, 20,
    'Updates the GP via a suitable procedure every this many iterations.'),
  get_option_specs('report_results_every', False, 20,
    'Report results every this many iterations.'),
  get_option_specs('monitor_progress_every', False, 9,
    ('Performs some simple sanity checks to make sure we are not stuck every this many',
     ' iterations.')),
  get_option_specs('monitor_domain_kernel_shrink', False, 0.9,
    ('If the optimum has not increased in a while, shrinks the kernel smoothness by this',
     ' much to increase variance.')),
  get_option_specs('monitor_mf_thresh_increase', False, 1.5,
    ('If we have not queried at the highest fidelity in a while, increases the leading',
     'constant by this much')),
  get_option_specs('track_every_time_step', False, 0,
    ('If 1, it tracks every time step.')),
  # TODO: implement code for next_pt_std_thresh
  get_option_specs('next_pt_std_thresh', False, 0.005,
    ('If the std of the queried point queries below this times the kernel scale ',
     'frequently we will reduce the bandwidth range')),
  ]
# All of them including what is needed for fitting GP.
all_mf_gp_bandit_args = all_mf_gp_args + mf_gp_bandit_args


# The MFGPBandit Class
# ========================================================================================
class MFGPBandit(object):
  """ MFGPBandit Class. """
  # pylint: disable=attribute-defined-outside-init

  # Methods needed for construction -------------------------------------------------
  def __init__(self, mf_opt_func, options=None, reporter=None):
    """ Constructor. """
    self.reporter = get_reporter(reporter)
    if options is None:
      options = load_options(all_mf_gp_bandit_args, reporter=reporter)
    self.options = options
    # Set up mfgp and mfof attributes
    self.mfof = mf_opt_func   # mfof refers to an MFOptFunction object.
    self.mfgp = None
    # Other set up
    self._set_up()

  def _set_up(self):
    """ Some additional set up routines. """
    # Check for legal parameter values
    self._check_options_vals('capital_type', ['given', 'cputime', 'realtime'])
    self._check_options_vals('acq', ['mf_gp_ucb', 'gp_ucb', 'gp_ei', 'mf_gp_ucb_finite',
                                     'mf_sko'])
    self._check_options_vals('acq_opt_criterion', ['rand', 'direct'])
    if isinstance(self.options.gpb_init, str):
      self._check_options_vals('gpb_init', ['random', 'random_lower_fidels'])
    # Set up some book keeping parameters
    self.available_capital = 0.0
    self.time_step = 0
    self.num_opt_fidel_queries = 0
    # Copy some stuff over from mfof
    copyable_params = ['fidel_dim', 'domain_dim']
    for param in copyable_params:
      setattr(self, param, getattr(self.mfof, param))
    # Set up acquisition optimisation
    self._set_up_acq_opt()
    # set up variables for monitoring
    self.monit_kernel_shrink_factor = 1
    self.monit_thresh_coeff = 1
    # Set initial history
    self.history = Namespace(query_fidels=np.zeros((0, self.fidel_dim)),
                             query_points=np.zeros((0, self.domain_dim)),
                             query_vals=np.zeros(0),
                             query_costs=np.zeros(0),
                             curr_opt_vals=np.zeros(0),
                             query_at_opt_fidel=np.zeros(0).astype(bool),
                            )

  @classmethod
  def _check_arg_vals(cls, arg_val, arg_name, allowed_vals):
    """ Checks if arg_val is in allowed_vals. """
    if arg_val not in allowed_vals:
      err_str = '%s should be one of %s.'%(arg_name,
                 ' '.join([str(x) for x in allowed_vals]))
      raise ValueError(err_str)

  def _check_options_vals(self, option_name, allowed_vals):
    """ Checks if the option option_name has taken a an allowed value. """
    return self._check_arg_vals(getattr(self.options, option_name),
                                option_name, allowed_vals)


  # Methods for setting up optimisation of acquisition ----------------------------------
  def _set_up_acq_opt(self):
    """ Sets up acquisition optimisation. """
    # First set up function to get maximum evaluations.
    if isinstance(self.options.acq_opt_max_evals, int):
      if self.options.acq_opt_max_evals > 0:
        self.get_acq_opt_max_evals = lambda t: self.options.acq_opt_max_evals
      else:
        self.get_acq_opt_max_evals = None
    else:
      # In this case, the user likely passed a function here.
      self.get_acq_opt_max_evals = self.options.acq_opt_max_evals
    # Now based on the optimisation criterion, do additional set up
    if self.options.acq_opt_criterion == 'direct':
      self._set_up_acq_opt_direct()
    elif self.options.acq_opt_criterion == 'rand':
      self._set_up_acq_opt_rand()
    else:
      raise NotImplementedError('Not implemented acq opt for %s yet!'%(
                                self.options.acq_opt_criterion))

  def _set_up_acq_opt_direct(self):
    """ Sets up acquisition optimisation with direct. """
    def _direct_wrap(*args):
      """ A wrapper so as to only return optimal value. """
      _, opt_pt, _ = direct_ft_maximise(*args)
      return opt_pt
    direct_lower_bounds = [0] * self.domain_dim
    direct_upper_bounds = [1] * self.domain_dim
    self.acq_optimise = lambda obj, max_evals: _direct_wrap(obj,
      direct_lower_bounds, direct_upper_bounds, max_evals)
    # Set up function for obtaining number of function evaluations.
    if self.get_acq_opt_max_evals is None:
      lead_const = 15 * min(5, self.domain_dim)**2
      self.get_acq_opt_max_evals = lambda t: lead_const * np.sqrt(min(t, 1000))
    # Acquisition function should be evaluated via single evaluations.
    self.acq_query_type = 'single'


  def _set_up_acq_opt_rand(self):
    """ Sets up acquisition optimisation with direct. """
    def _random_max_wrap(*args):
      """ A wrapper so as to only return optimal value. """
      _, opt_pt = random_maximise(*args)
      return opt_pt
    rand_bounds = np.array([[0, 1]] * self.domain_dim)
    self.acq_optimise = lambda obj, max_evals: _random_max_wrap(obj,
      rand_bounds, max_evals)
    if self.get_acq_opt_max_evals is None:
      lead_const = 7 * min(5, self.domain_dim)**2
      self.get_acq_opt_max_evals = lambda t: np.clip(
        lead_const * np.sqrt(min(t, 1000)), 1000, 2e4)
    # Acquisition function should be evaluated via multiple evaluations
    self.acq_query_type = 'multiple'

  # Book keeping methods ------------------------------------------------------------
  def _update_history(self, pts_fidel, pts_domain, pts_val, pts_cost, at_opt_fidel):
    """ Adds a query point to the history and discounts the capital etc. """
    pts_fidel = pts_fidel.reshape(-1, self.fidel_dim)
    pts_domain = pts_domain.reshape(-1, self.domain_dim)
    pts_val = pts_val if hasattr(pts_val, '__len__') else [pts_val]
    pts_cost = pts_cost if hasattr(pts_cost, '__len__') else [pts_cost]
    # Append to history
    self.history.query_fidels = np.append(self.history.query_fidels, pts_fidel, axis=0)
    self.history.query_points = np.append(self.history.query_points, pts_domain, axis=0)
    self.history.query_vals = np.append(self.history.query_vals, pts_val, axis=0)
    self.history.query_costs = np.append(self.history.query_costs, pts_cost, axis=0)
    self.history.curr_opt_vals = np.append(self.history.curr_opt_vals, self.gpb_opt_val)
    self.history.query_at_opt_fidel = np.append(self.history.query_at_opt_fidel,
                                                at_opt_fidel)

  def _get_min_distance_to_opt_fidel(self):
    """ Computes the minimum distance to the optimal fidelity. """
    dists_to_of = np.linalg.norm(self.history.query_fidels - self.mfof.opt_fidel, axis=1)
    return dists_to_of.min()

  def _report_current_results(self):
    """ Writes the current results to the reporter. """
    cost_frac = self.spent_capital / self.available_capital
    report_str = '  '.join(['%s-%03d::'%(self.options.acq, self.time_step),
                            'cost: %0.3f,'%(cost_frac),
                            '#hf_queries: %03d,'%(self.num_opt_fidel_queries),
                            'optval: %0.4f'%(self.gpb_opt_val)
                           ])
    if self.num_opt_fidel_queries == 0:
      report_str = report_str + '. min-to-of: %0.4f'%(
                    self._get_min_distance_to_opt_fidel())
    self.reporter.writeln(report_str)

  # Methods for managing the GP -----------------------------------------------------
  def _build_new_gp(self):
    """ Builds the GP with the data in history and stores in self.mfgp. """
    if hasattr(self.mfof, 'init_mfgp') and self.mfof.init_mfgp is not None:
      self.mfgp = deepcopy(self.mfof.init_mfgp)
      self.mfgp.add_mf_data(self.history.query_fidels, self.history.query_points,
                            self.history.query_vals)
      mfgp_prefix_str = 'Using given gp: '
    else:
      # Set domain bandwidth bounds
      if self.options.shrink_kernel_with_time:
        bw_ub = max(0.2, 2/(1+self.time_step)**0.25)
        domain_bw_log_bounds = [[0.05, bw_ub]] * self.domain_dim
        self.options.domain_bandwidth_log_bounds = np.array(domain_bw_log_bounds)
      else:
        self.options.domain_bandwidth_log_bounds = np.array([[0, 4]] * self.domain_dim)
      # Set fidelity bandwidth bounds
      self.options.fidel_bandwidth_log_bounds = np.array([[0, 4]] * self.fidel_dim)
      # Call the gp fitter
      mfgp_fitter = MFGPFitter(self.history.query_fidels, self.history.query_points,
        self.history.query_vals, options=self.options, reporter=self.reporter)
      self.mfgp, _ = mfgp_fitter.fit_gp()
      mfgp_prefix_str = 'Fitting GP (t=%d): '%(self.time_step) # increase bandwidths
    mfgp_str = '          -- %s%s.'%(mfgp_prefix_str, str(self.mfgp))
    self.reporter.writeln(mfgp_str)

  def _add_data_to_mfgp(self, fidel_pt, domain_pt, val_pt):
    """ Adds data to self.mfgp. """
    self.mfgp.add_mf_data(fidel_pt.reshape((-1, self.fidel_dim)),
                       domain_pt.reshape((-1, self.domain_dim)),
                       np.array(val_pt).ravel())


  # Methods needed for initialisation -----------------------------------------------
  def perform_initial_queries(self):
    """ Performs an initial set of queries to initialise optimisation. """
    if not isinstance(self.options.gpb_init, str):
      raise NotImplementedError('Not implemented taking given initialisation yet.')
    # First determine the initial budget.
    gpb_init_capital = (self.options.gpb_init_capital if self.options.gpb_init_capital > 0
                        else self.options.gpb_init_capital_frac * self.available_capital)
    if self.options.acq in ['gp_ucb', 'gp_ei']:
      num_sf_init_pts = np.ceil(float(gpb_init_capital)/self.mfof.opt_fidel_cost)
      fidel_init_pts = np.repeat(self.mfof.opt_fidel.reshape(1, -1), num_sf_init_pts,
                                 axis=0)
    elif self.options.acq in ['mf_gp_ucb', 'mf_gp_ucb_finite', 'mf_sko']:
      fidel_init_pts = self._mf_method_random_initial_fidels_random(gpb_init_capital)
    num_init_pts = len(fidel_init_pts)
    domain_init_pts = latin_hc_sampling(self.domain_dim, num_init_pts)
    for i in range(num_init_pts):
      self.query(fidel_init_pts[i], domain_init_pts[i])
      if self.spent_capital >= gpb_init_capital:
        break
    self.reporter.writeln('Initialised %s with %d queries, %d at opt_fidel.'%(
      self.options.acq, len(self.history.query_vals), self.num_opt_fidel_queries))

  def _mf_method_random_initial_fidels_interweaved(self):
    """Gets initial fidelities for a multi-fidelity method. """
    rand_fidels = self.mfof.get_candidate_fidelities()
    np.random.shuffle(rand_fidels)
    num_rand_fidels = len(rand_fidels)
    opt_fidels = np.repeat(self.mfof.opt_fidel.reshape(1, -1), num_rand_fidels, axis=0)
    fidel_init_pts = np.empty((2*num_rand_fidels, self.fidel_dim), dtype=np.float64)
    fidel_init_pts[0::2] = rand_fidels
    fidel_init_pts[1::2] = opt_fidels
    return fidel_init_pts

  def _mf_method_random_initial_fidels_random(self, gpb_init_capital):
    """Gets initial fidelities for a multi-fidelity method. """
    cand_fidels = self.mfof.get_candidate_fidelities()
    cand_costs = self.mfof.cost(cand_fidels)
    not_too_expensive_fidel_idxs = cand_costs <= (gpb_init_capital / 3.0)
    fidel_init_pts = cand_fidels[not_too_expensive_fidel_idxs, :]
    np.random.shuffle(fidel_init_pts)
    return np.array(fidel_init_pts)

  def initialise_capital(self):
    """ Initialises capital. """
    self.spent_capital = 0.0
    if self.options.capital_type == 'cputime':
      self.cpu_time_stamp = time.clock()
    elif self.options.capital_type == 'realtime':
      self.real_time_stamp = time.time()

  def optimise_initialise(self):
    """ Initialisation for optimisation. """
    self.gpb_opt_pt = None
    self.gpb_opt_val = -np.inf
    self.initialise_capital() # Initialise costs
    self.perform_initial_queries() # perform initial queries
    self._build_new_gp()

  # Methods needed for monitoring -------------------------------------------------
  def _monitor_progress(self):
    """ Monitors progress. """
#     self._monitor_opt_val()
    self._monitor_opt_fidel_queries()

  def _monitor_opt_val(self):
    """ Monitors progress of the optimum value. """
    # Is the optimum increasing over time.
    if (self.history.curr_opt_vals[-self.options.monitor_progress_every] * 1.01 >
        self.gpb_opt_val):
      recent_queries = self.history.query_points[-self.options.monitor_progress_every:, :]
      recent_queries_mean = recent_queries.mean(axis=0)
      dispersion = np.linalg.norm(recent_queries - recent_queries_mean, ord=2, axis=1)
      dispersion = dispersion.mean() / np.sqrt(self.domain_dim)
      lower_dispersion = 0.05
      upper_dispersion = 0.125
      if dispersion < lower_dispersion:
        self.monit_kernel_shrink_factor *= self.options.monitor_domain_kernel_shrink
      elif dispersion > upper_dispersion:
        self.monit_kernel_shrink_factor /= self.options.monitor_domain_kernel_shrink
      if not lower_dispersion < dispersion < upper_dispersion:
        self.mfgp.domain_kernel.change_smoothness(self.monit_kernel_shrink_factor)
        self.mfgp.build_posterior()
        self.reporter.writeln('%s--monitor: Kernel shrink set to %0.4f.'%(' '*10,
                              self.monit_kernel_shrink_factor))

  def _monitor_opt_fidel_queries(self):
    """ Monitors if we querying at higher fidelities too much or too little. """
    # Are we querying at higher fidelities too much or too little.
    if self.options.acq in ['mf_gp_ucb', 'mf_gp_ucb_finite']:
      of_start_query = max(0, (len(self.history.query_vals) -
                               2*self.options.monitor_progress_every))
      of_recent_query_idxs = range(of_start_query, len(self.history.query_vals))
      recent_query_at_opt_fidel = self.history.query_at_opt_fidel[of_recent_query_idxs]
      recent_query_at_opt_fidel_mean = recent_query_at_opt_fidel.mean()
      if not 0.25 <= recent_query_at_opt_fidel_mean <= 0.75:
        if recent_query_at_opt_fidel_mean < 0.25:
          self.monit_thresh_coeff *= self.options.monitor_mf_thresh_increase
        else:
          self.monit_thresh_coeff /= self.options.monitor_mf_thresh_increase
        self.reporter.writeln(('%s-- monitor: Changing thresh_coeff  to %0.3f, ' + 
                               'recent-query-frac: %0.3f.')%(
                               ' '*10, self.monit_thresh_coeff,
                               recent_query_at_opt_fidel_mean))

  # Methods needed for optimisation -------------------------------------------------
  def _terminate_now(self):
    """ Returns true if we should terminate now. """
    if self.time_step >= self.options.max_iters:
      return True
    return self.spent_capital >= self.available_capital

  def add_capital(self, capital):
    """ Adds capital. """
    self.available_capital += capital

  def _determine_next_query_point(self):
    """ Obtains the next query point according to the acquisition. """
    # Construction of acquisition function ------
    if self.options.acq in ['mf_gp_ucb', 'gp_ucb', 'mf_gp_ucb_finite']:
      def _acq_max_obj(x):
        """ A wrapper for the mf_gp_ucb acquisition. """
        ucb, _ = acquisitions.mf_gp_ucb(self.acq_query_type, x, self.mfgp,
                                        self.mfof.opt_fidel, self.time_step)
        return ucb
    elif self.options.acq in ['gp_ei', 'mf_sko']:
      def _acq_max_obj(x):
        """ A wrapper for the gp_ei acquisition. """
        return acquisitions.gp_ei(self.acq_query_type, x, self.mfgp, self.mfof.opt_fidel,
                                  self.gpb_opt_val)
    else:
      raise NotImplementedError('Only implemented %s yet!.'%(self.options.acq))
    # Maximise -----
    next_pt = self.acq_optimise(_acq_max_obj, self.get_acq_opt_max_evals(self.time_step))
    # Store results -----
    acq_params = Namespace()
    if self.options.acq in ['mf_gp_ucb', 'gp_ucb', 'mf_gp_ucb_finite']:
      max_acq_val, beta_th = acquisitions.mf_gp_ucb_single(next_pt, self.mfgp,
                              self.mfof.opt_fidel, self.time_step)
      acq_params.beta_th = beta_th
      acq_params.thresh_coeff = self.monit_thresh_coeff
    else:
      max_acq_val = acquisitions.gp_ei_single(next_pt, self.mfgp, self.mfof.opt_fidel,
                                              self.gpb_opt_val)
    acq_params.max_acq_val = max_acq_val
    return next_pt, acq_params

  def _determine_next_fidel(self, next_pt, acq_params):
    """ Determines the next fidelity. """
    if self.options.acq in ['mf_gp_ucb', 'mf_gp_ucb_finite']:
      next_fidel = fidelity_choosers.mf_gp_ucb(next_pt, self.mfgp, self.mfof, acq_params)
    elif self.options.acq in ['mf_sko']:
      next_fidel = fidelity_choosers.mf_sko(self.mfof, next_pt, self.mfgp, acq_params)
    elif self.options.acq in ['gp_ucb', 'gp_ei']:
      next_fidel = deepcopy(self.mfof.opt_fidel)
    return next_fidel

  @classmethod
  def _process_next_fidel_and_pt(cls, next_fidel, next_pt):
    """ Processes next point and fidel. Will do certiain things such as perturb it if its
        too close to an existing point. """
    return next_fidel, next_pt

  def _update_capital(self, fidel_pt):
    """ Updates the capital according to the cost of the current query. """
    if self.options.capital_type == 'given':
      pt_cost = self.mfof.cost_single(fidel_pt)
    elif self.options.capital_type == 'cputime':
      new_cpu_time_stamp = time.clock()
      pt_cost = new_cpu_time_stamp - self.cpu_time_stamp
      self.cpu_time_stamp = new_cpu_time_stamp
    elif self.options.capital_type == 'realtime':
      new_real_time_stamp = time.time()
      pt_cost = new_real_time_stamp - self.real_time_stamp
      self.real_time_stamp = new_real_time_stamp
    self.spent_capital += pt_cost
    return pt_cost

  # The actual querying happens here
  def query(self, fidel_pt, domain_pt):
    """ The querying happens here. It also calls functions to update history and the
        maximum value/ points. But it does *not* update the GP. """
    val_pt = self.mfof.eval_single(fidel_pt, domain_pt)
    cost_pt = self._update_capital(fidel_pt)
    # Update the optimum point
    if (np.linalg.norm(fidel_pt - self.mfof.opt_fidel) < 1e-5 and
        val_pt > self.gpb_opt_val):
      self.gpb_opt_val = val_pt
      self.gpb_opt_pt = domain_pt
    # Add to history
    at_opt_fidel = is_an_opt_fidel_query(fidel_pt, self.mfof.opt_fidel)
    self._update_history(fidel_pt, domain_pt, val_pt, cost_pt, at_opt_fidel)
    if at_opt_fidel:
      self.num_opt_fidel_queries += 1
    return val_pt, cost_pt

  def _time_keeping(self, reset=0):
    """ Used to keep time by _track_time_step. """
    curr_keep_time = time.time()
    curr_keep_clock = time.clock()
    if reset:
      self.last_keep_time = curr_keep_time
      self.last_keep_clock = curr_keep_clock
    else:
      time_diff = curr_keep_time - self.last_keep_time
      clock_diff = curr_keep_clock - self.last_keep_clock
      self.last_keep_time = curr_keep_time
      self.last_keep_clock = curr_keep_clock
      return round(time_diff, 3), round(clock_diff, 3)

  def _track_time_step(self, msg=''):
    """ Used to track time step. """
    if not self.options.track_every_time_step:
      return
    if not msg:
      self._time_keeping(0)
      self.reporter.writeln('')
    else:
      self.reporter.write('%s: t%s, '%(msg, self._time_keeping()))


  # Main optimisation function ------------------------------------------------------
  def optimise(self, max_capital):
    """ This executes the sequential optimisation routine. """
    # Preliminaries
    self.add_capital(max_capital)
    self.optimise_initialise()

    # Main loop --------------------------
    while not self._terminate_now():
      self.time_step += 1 # increment time
      if self.time_step % self.options.build_new_gp_every == 0: # Build GP if needed
        self._build_new_gp()
      if self.time_step % self.options.monitor_progress_every == 0:
        self._monitor_progress()

      # Determine next query
      self._track_time_step()
      next_pt, acq_params = self._determine_next_query_point()
      self._track_time_step('#%d, next point'%(self.time_step))
      next_fidel = self._determine_next_fidel(next_pt, acq_params)
      next_fidel, next_pt = self._process_next_fidel_and_pt(next_fidel, next_pt)
      self._track_time_step('next fidel')
      next_val, _ = self.query(next_fidel, next_pt)
      self._track_time_step('querying')
      # update the gp
      self._add_data_to_mfgp(next_fidel, next_pt, next_val)
      self._track_time_step('gp-update')

      if self.time_step % self.options.report_results_every == 0: # report results
        self._report_current_results()
    return self.gpb_opt_pt, self.gpb_opt_val, self.history

# MFGPBandit Class ends here ========================================================


# APIs for MF GP Bandit optimisation -----------------------------------------------------
# Optimisation from a mf_Func.MFOptFunction object
def mfgpb_from_mfoptfunc(mf_opt_func, max_capital, acq=None, options=None,
                         reporter='default'):
  """ MF GP Bandit optimisation with an mf_func.MFOptFunction object. """
#   if not isinstance(mf_opt_func, MFOptFunction):
#     raise ValueError('mf_opt_func should be a mf_func.MFOptFunction instance.')
  if acq is not None:
    if options is None:
      reporter = get_reporter(reporter)
      options = load_options(all_mf_gp_bandit_args, reporter=reporter)
    options.acq = acq
  return (MFGPBandit(mf_opt_func, options, reporter)).optimise(max_capital)


# Main API
def mfgpb(mf_func, fidel_cost_func, fidel_bounds, domain_bounds, opt_fidel, max_capital,
  acq=None, options=None, reporter=None, vectorised=True, true_opt_pt=None,
  true_opt_val=None):
  # pylint: disable=too-many-arguments
  """ This function executes GP Bandit (Bayesian Optimisation)
    Input Arguments:
      - mf_func: The multi-fidelity function to be optimised.
      - fidel_cost_func: The function which describes the cost for each fidelity.
      - fidel_bounds, domain_bounds: The bounds for the fidelity space and domain.
      - opt_fidel: The point in the fidelity space at which to optimise mf_func.
      - max_capital: The maximum capital for optimisation.
      - options: A namespace which gives other options.
      - reporter: A reporter object to write outputs.
      - vectorised: If true, it means mf_func and fidel_cost_func take matrix inputs. If
          false, they take only single point inputs.
      - true_opt_pt, true_opt_val: The true optimum point and value (if known). Mostly for
          experimenting with synthetic problems.
    Returns: (gpb_opt_pt, gpb_opt_val, history)
      - gpb_opt_pt, gpb_opt_val: The optimum point and value.
      - history: A namespace which contains a history of all the previous queries.
  """
  mf_opt_func = MFOptFunction(mf_func, fidel_cost_func, fidel_bounds, domain_bounds,
                              opt_fidel, vectorised, true_opt_pt, true_opt_val)
  return mfgpb_from_mfoptfunc(mf_opt_func, max_capital, acq, options, reporter)

