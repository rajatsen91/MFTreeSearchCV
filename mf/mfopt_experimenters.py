"""
  Harness for conducting experiments for MF Optimisation.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=relative-import
# pylint: disable=abstract-class-not-used

from argparse import Namespace
from datetime import datetime
import numpy as np
import os
# Local imports
from mf_gp_bandit import mfgpb_from_mfoptfunc
from mf_func import NoisyMFOptFunction
from utils.experimenters import BasicExperimenter
from utils.optimisers import direct_maximise_from_mfof
from utils.reporters import get_reporter
# scratch
from scratch.get_finite_fidel_mfof import get_finite_mfof_from_mfof


class MFOptExperimenter(BasicExperimenter):
  """ Base class for running experiments. """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, experiment_name, mfof, max_capital, methods, num_experiments,
               save_dir, save_file_prefix='', method_options=None, method_reporter=None,
               *args, **kwargs):
    """ Constructor for MFOptExperiment. See BasicExperimenter for more args.
          mfof: A MFOptFunction Object.
          methods: are the methods we will use for MF optimisation.
          method_options: a dictionary which gives the options for each option.
    """
    save_file_name = self._get_save_file_name(save_dir, experiment_name, save_file_prefix)
    super(MFOptExperimenter, self).__init__(experiment_name, num_experiments,
                                            save_file_name, *args, **kwargs)
    self.mfof = mfof
    self.max_capital = max_capital
    self.methods = methods
    self.num_methods = len(self.methods)
    self.method_options = (method_options if method_options else
                           {key: None for key in method_options})
    self.method_reporter = get_reporter(method_reporter)
    self.noisy_observations = isinstance(mfof, NoisyMFOptFunction)
    self._set_up_saving() # Set up for saving results.

  @classmethod
  def _get_save_file_name(cls, save_dir, experiment_name, save_file_prefix):
    """ Gets the save file name. """
    save_file_prefix = save_file_prefix if save_file_prefix else experiment_name
    save_file_name = '%s-%s.mat'%(save_file_prefix,
                                  datetime.now().strftime('%m%d-%H%M%S'))
    save_file_name = os.path.join(save_dir, save_file_name)
    return save_file_name

  def _set_up_saving(self):
    """ Runs some routines to set up saving. """
    # Store methods and the options in to_be_saved
    self.to_be_saved.methods = self.methods
    self.to_be_saved.method_options = self.method_options
    # Data about the problem
    self.to_be_saved.true_opt_val = (self.mfof.opt_val if self.mfof.opt_val is not None
                                     else -np.inf)
    self.to_be_saved.true_opt_pt = (self.mfof.opt_pt if self.mfof.opt_pt is not None
                                    else np.zeros((1)))
    self.to_be_saved.opt_fidel = self.mfof.opt_fidel
    self.to_be_saved.opt_fidel_unnormalised = self.mfof.opt_fidel_unnormalised
    self.to_be_saved.fidel_dim = self.mfof.fidel_dim
    self.to_be_saved.domain_dim = self.mfof.domain_dim
    self.fidel_dim = self.mfof.fidel_dim
    self.domain_dim = self.mfof.domain_dim
    # For the results
    self.data_to_be_extracted = ['query_fidels', 'query_points', 'query_vals',
                                 'query_costs', 'curr_opt_vals', 'query_at_opt_fidel']
    self.data_to_be_saved = self.data_to_be_extracted + ['true_curr_opt_vals']
    for data_type in self.data_to_be_saved:
      setattr(self.to_be_saved, data_type, self._get_new_empty_results_array())

  def _get_new_empty_results_array(self):
    """ Returns a new empty array to be used for storing results. """
    return np.empty((self.num_methods, 0), dtype=np.object)

  def _get_new_iter_results_array(self):
    """ Returns a new empty array for saving results of the current iteration. """
    return np.empty((self.num_methods, 1), dtype=np.object)

  def _print_method_header(self, method):
    """ Prints a header for the current method. """
    experiment_header = '--Exp %d/%d.  Method: %s with capital %0.4f'%(
      self.experiment_iter, self.num_experiments, method, self.max_capital)
    experiment_header = '\n' + experiment_header + '\n' + '-' * len(experiment_header)
    self.reporter.writeln(experiment_header)

  def _print_method_result(self, method, comp_opt_val, num_opt_fidel_evals):
    """ Prints the result for this method. """
    result_str = 'Method: %s achieved max-val %0.5f in %d opt-fidel queries.'%(method,
                  comp_opt_val, num_opt_fidel_evals)
    self.reporter.writeln(result_str)

  def run_experiment_iteration(self):
    """ Runs each method in self.methods once and stores the results to to_be_saved."""
    # pylint: disable=too-many-branches
    curr_iter_results = Namespace()
    for data_type in self.data_to_be_saved:
      setattr(curr_iter_results, data_type, self._get_new_iter_results_array())

    # We will go through each method in this loop ----------------------------------
    for meth_iter in range(self.num_methods):
      method = self.methods[meth_iter]
      self._print_method_header(method)
      # Create arrays for storing.
      # Run the method.
      if method in ['mf_gp_ucb_finite', 'mf_sko']:
        curr_mfof = get_finite_mfof_from_mfof(self.mfof,
                      self.method_options[method].finite_fidels,
                      self.method_options[method].finite_fidels_is_normalised)
      else:
        curr_mfof = self.mfof
      if method in ['gp_ucb', 'gp_ei', 'mf_gp_ucb', 'mf_gp_ucb_finite', 'mf_sko']:
        _, _, opt_hist = mfgpb_from_mfoptfunc(curr_mfof, self.max_capital,
                                              acq=method,
                                              options=self.method_options[method],
                                              reporter=self.method_reporter)
        if method in ['gp_ucb', 'gp_ei']:
          # save some parameters for DiRect because I can't control each evaluation in the
          # fortran library
          direct_num_evals = len(opt_hist.query_fidels)
          direct_av_cost = opt_hist.query_costs.mean()

      elif method == 'direct':
        # As this is deterministic, just run it once.
        if self.experiment_iter == 1:
          _, _, opt_hist = direct_maximise_from_mfof(self.mfof, direct_num_evals)
          num_actual_direct_evals = len(opt_hist.curr_opt_vals)
          opt_hist.query_fidels = np.repeat(self.mfof.opt_fidel.reshape(1, -1),
                                            num_actual_direct_evals, axis=0)
          opt_hist.query_points = np.zeros((num_actual_direct_evals, self.mfof.fidel_dim))
          opt_hist.query_vals = np.zeros((num_actual_direct_evals))
          opt_hist.query_costs = direct_av_cost * np.ones((num_actual_direct_evals))
          opt_hist.query_at_opt_fidel = np.ones((num_actual_direct_evals), dtype=bool)
        else:
          self.reporter.writeln('Not running %s this iteration as it is deterministic.'%(
                                method))
          opt_hist = Namespace()
          for data_type in self.data_to_be_extracted:
            data_pointer = getattr(self.to_be_saved, data_type)
            setattr(opt_hist, data_type, data_pointer[meth_iter, 0])

      else:
        raise ValueError('Unknown method %s!'%(method))

      # Save noiseless function values results
      if self.noisy_observations:
        num_evals = len(opt_hist.curr_opt_vals)
        curr_best = -np.inf
        opt_hist.true_curr_opt_vals = np.zeros((num_evals))
        for i in range(num_evals):
          if opt_hist.query_at_opt_fidel[i]:
            curr_value = self.mfof.eval_single_noiseless(self.mfof.opt_fidel,
                                                         opt_hist.query_points[i])
            if curr_value > curr_best:
              curr_best = curr_value
          opt_hist.true_curr_opt_vals[i] = curr_best
      else:
        opt_hist.true_curr_opt_vals = opt_hist.curr_opt_vals

      # Save the results.
      for data_type in self.data_to_be_saved:
        data = getattr(opt_hist, data_type)
        data_pointer = getattr(curr_iter_results, data_type)
        data_pointer[meth_iter, 0] = data
      # Print out the results
      comp_opt_val = opt_hist.true_curr_opt_vals[-1]
      self._print_method_result(method, comp_opt_val, opt_hist.query_at_opt_fidel.sum())
    # for meth_iter ends here -------------------------------------------------------

    # Now save the results of this experiment in to_be_saved
    for data_type in self.data_to_be_saved:
      data = getattr(curr_iter_results, data_type)
      curr_data_to_be_saved = getattr(self.to_be_saved, data_type)
      updated_data_to_be_saved = np.append(curr_data_to_be_saved, data, axis=1)
      setattr(self.to_be_saved, data_type, updated_data_to_be_saved)

  def get_iteration_header(self):
    """ Header for iteration. """
    noisy_str = ('Noiseless' if not self.noisy_observations else
                 'noisy (var=%0.4f)'%(self.mfof.noise_var))
    opt_val_str = '?' if self.mfof.opt_val is None else '%0.4f'%(self.mfof.opt_val)
    ret = '%s(p=%d,d=%d), max=%s, max-capital %0.3f, %s'%(self.experiment_name,
      self.mfof.fidel_dim, self.mfof.domain_dim, opt_val_str,
      self.max_capital, noisy_str)
    return ret

