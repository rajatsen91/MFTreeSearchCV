"""
  Unit tests for MF-GP-Bandits
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=maybe-no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=super-on-old-class

import numpy as np
# Local
from examples.synthetic_functions import get_mf_hartmann_as_mfof
from gen_mfgp_sample import gen_simple_mfgp_as_mfof
from mf_func import get_noisy_mfof_from_mfof
import mf_gp_bandit
from mf_gpb_utils import are_opt_fidel_queries
from unittest_mf_func import get_mf_func_data
from utils.ancillary_utils import is_non_decreasing_sequence
from utils.base_test_class import BaseTestClass, execute_tests
import utils.reporters as reporters
from utils import option_handler


# Generate data
def _get_gpb_instances():
  """ Generates some GPB problems and MFGPBandit instances. """
  instances = get_mf_func_data()
  for inst in instances:
    inst.mfgpb = mf_gp_bandit.MFGPBandit(inst.mfof)
  return instances

def _get_gpb_problem():
  """ Generates one bandit problem and returns. """
  problems = get_mf_func_data()
  ret = problems[0]
  ret.reporter = reporters.SilentReporter()
  return ret


class MFGPBanditTestCase(BaseTestClass):
  """ Unit tests for mf_gpb_utils.py """

  def setUp(self):
    """ Sets up unit tests. """
    pass

  def test_initial_sampling(self):
    """ Test for initialisation sampling. """
    self.report('Testing sample initialisation.')
    prob = _get_gpb_problem()
    acquisitions = ['mf_gp_ucb', 'gp_ucb', 'gp_ei']
    options = option_handler.load_options(mf_gp_bandit.all_mf_gp_bandit_args,
                                          reporter=prob.reporter)
    for acq in acquisitions:
      options.acq = acq
      options.gpb_init_capital = prob.mfof.opt_fidel_cost * 23.2
      mfgpb = mf_gp_bandit.MFGPBandit(prob.mfof, options, prob.reporter)
      mfgpb.optimise_initialise()
      hf_idxs = are_opt_fidel_queries(mfgpb.history.query_fidels, prob.mfof.opt_fidel)
      hf_vals = mfgpb.history.query_vals[hf_idxs]
      num_hf_queries = len(hf_vals)
      self.report(('Initialised %s with %d queries (%d at opt_fidel). Init capital = ' +
                   '%0.4f (%0.4f used) ')%(acq, len(mfgpb.history.query_vals),
                   num_hf_queries, options.gpb_init_capital, mfgpb.spent_capital),
                   'test_result')
      assert mfgpb.spent_capital <= 1.1 * options.gpb_init_capital
      assert mfgpb.history.curr_opt_vals[-1] == mfgpb.gpb_opt_val
      assert is_non_decreasing_sequence(mfgpb.history.curr_opt_vals)
      assert num_hf_queries == 0 or hf_vals.max() == mfgpb.gpb_opt_val
      assert mfgpb.num_opt_fidel_queries == num_hf_queries
      assert mfgpb.history.query_at_opt_fidel.sum() == num_hf_queries

  def test_gpb_opt_1(self):
    """ Tests the optimisaiton routine. """
    # pylint: disable=bad-whitespace
    self.report('Tests mf-gp-ucb using a sample from gen_mfgp_sample_as_mfof.')
    mfof = gen_simple_mfgp_as_mfof(random_seed=np.random.randint(1000))
    mfof.init_mfgp = mfof.mfgp
    # Also get the noisy mfof
    nmfof = get_noisy_mfof_from_mfof(mfof, mfof.mfgp.noise_var)
    method_data = [('gp_ucb',    20 * mfof.opt_fidel_cost),
                   ('gp_ei',     20 * mfof.opt_fidel_cost),
                   ('mf_gp_ucb', 20 * mfof.opt_fidel_cost)]

    for meth in method_data:
      opt_pt, opt_val, _ = mf_gp_bandit.mfgpb_from_mfoptfunc(mfof, meth[1], acq=meth[0],
                                                             reporter='silent')
      report_str = ('%s::  capital: %0.1f, opt_pt: %s, opt_val: %0.4f, ' +
                    'true_opt: %s, true opt_val: %0.4f')%(meth[0], meth[1], str(opt_pt),
                    opt_val, str(mfof.opt_pt), mfof.opt_val)
      self.report(report_str, 'test_result')
      # Now do the noisy version
      noisy_opt_pt, noisy_opt_val, _ = mf_gp_bandit.mfgpb_from_mfoptfunc(nmfof, meth[1],
                                         acq=meth[0], reporter='silent')
      if noisy_opt_val < np.inf:
        fval_at_noisy_opt_pt = mfof.eval_single(mfof.opt_fidel, noisy_opt_pt)
      else:
        fval_at_noisy_opt_pt = np.inf
      noisy_report_str = ('Noisy %s::  noise: %0.4f, opt_pt: %s, opt_val: %0.4f, ' +
                    'fval_at_noisy_opt_pt: %0.4f')%(meth[0], nmfof.noise_var,
                    str(noisy_opt_pt), noisy_opt_val, fval_at_noisy_opt_pt)
      self.report(noisy_report_str, 'test_result')

  def test_gpb_opt_2(self):
    """ Tests the optimisaiton routine. """
    # pylint: disable=bad-whitespace
    self.report('Tests mf-gp-ucb using the hartmann function while learning kernel.')
    mfof = get_mf_hartmann_as_mfof(2, 3)
    noise_var = 0.1
    nmfof = get_noisy_mfof_from_mfof(mfof, noise_var)
    method_data = [('gp_ucb',    20 * mfof.opt_fidel_cost),
                   ('gp_ei',     20 * mfof.opt_fidel_cost),
                   ('mf_gp_ucb', 20 * mfof.opt_fidel_cost)]

    for meth in method_data:
      _, opt_val, history = mf_gp_bandit.mfgpb_from_mfoptfunc(mfof, meth[1],
                                   acq=meth[0], reporter='silent')
      num_opt_fidel_queries = history.query_at_opt_fidel.sum()
      total_num_queries = len(history.query_at_opt_fidel)
      report_str = ('%s::  capital: %0.4f, opt_val: %0.4f, true opt_val: %0.4f, ' +
                    'queries(at-opt_fidel): %d(%d).')%(meth[0], meth[1], opt_val,
                    mfof.opt_val, total_num_queries, num_opt_fidel_queries)
      self.report(report_str, 'test_result')
      # Now do the noisy version
      noisy_opt_pt, noisy_opt_val, noisy_history = mf_gp_bandit.mfgpb_from_mfoptfunc(
        nmfof, meth[1], acq=meth[0], reporter='silent')
      noisy_num_opt_fidel_queries = noisy_history.query_at_opt_fidel.sum()
      noisy_total_num_queries = len(history.query_at_opt_fidel)
      if noisy_opt_val < np.inf:
        fval_at_noisy_opt_pt = mfof.eval_single(mfof.opt_fidel, noisy_opt_pt)
      else:
        fval_at_noisy_opt_pt = np.inf
      noisy_report_str = ('Noisy %s::  noise: %0.4f, opt_val: %0.4f, ' +
                          'fval_at_noisy_opt_pt: %0.4f, ' + 
                          'queries(at-opt_fidel): %d(%d).')%(meth[0], nmfof.noise_var,
                          noisy_opt_val, fval_at_noisy_opt_pt, noisy_total_num_queries,
                          noisy_num_opt_fidel_queries)
      self.report(noisy_report_str, 'test_result')

if __name__ == '__main__':
  execute_tests()

