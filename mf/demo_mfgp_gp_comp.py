"""
  A simple demo for mf gps. We will use the same data and see if the GPs learned are
  the same.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=superfluous-parens

import numpy as np
# Local
from gp import gp_instances
import mf_gp
from utils.reporters import BasicReporter
from utils.option_handler import load_options
from utils.general_utils import compute_average_sq_prediction_error
from unittest_mf_gp import gen_data_from_func


def get_data():
  """ Generates data for the demo. """
  fzx = lambda z, x: (z**2).sum(axis=1) + (x**2).sum(axis=1)
  dim_z = 1
  dim_x = 1
  N = 100
  Z_tr, X_tr, Y_tr = gen_data_from_func(fzx, N, dim_z, dim_x)
  ZX_tr = np.concatenate((Z_tr, X_tr), axis=1)
  Z_te, X_te, Y_te = gen_data_from_func(fzx, N, dim_z, dim_x)
  ZX_te = np.concatenate((Z_te, X_te), axis=1)
  return Z_tr, X_tr, Y_tr, ZX_tr, Z_te, X_te, Y_te, ZX_te

def _print_str_results(reporter, descr, sgp_result, mfgp_result):
  """ Prints the result out. """
  print_str = '%s:: S-GP: %s, MF-GP: %s'%(descr, sgp_result, mfgp_result)
  reporter.writeln(print_str)

def _print_float_results(reporter, descr, sgp_result, mfgp_result):
  """ Prints float results. """
  sgp_result = '%0.4f'%(sgp_result)
  mfgp_result = '%0.4f'%(mfgp_result)
  _print_str_results(reporter, descr, sgp_result, mfgp_result)

def main():
  """ Main function. """
  # pylint: disable=too-many-locals
  # pylint: disable=maybe-no-member
  np.random.seed(0)
  reporter = BasicReporter()
  Z_tr, X_tr, Y_tr, ZX_tr, Z_te, X_te, Y_te, ZX_te = get_data()
  sgp_options = load_options(gp_instances.all_simple_gp_args, 'GP', reporter=reporter)
  mfgp_options = load_options(mf_gp.all_mf_gp_args, 'MFGP', reporter=reporter)
  mfgp_options.mean_func_type = 'median'
  # Fit the GPs.
  sgp_fitter = gp_instances.SimpleGPFitter(ZX_tr, Y_tr, sgp_options, reporter=reporter)
  sgp, opt_s = sgp_fitter.fit_gp()
  mfgp_fitter = mf_gp.MFGPFitter(Z_tr, X_tr, Y_tr, mfgp_options, reporter=reporter)
  mfgp, opt_mf = mfgp_fitter.fit_gp()
  opt_s = (np.array(opt_s).round(4))
  opt_mf = (np.array(opt_mf).round(4))
  s_bounds = sgp_fitter.hp_bounds.round(3)
  mf_bounds = mfgp_fitter.hp_bounds.round(3)
  # Print out some fitting statistics
  _print_str_results(reporter, 'Opt-pts', str(opt_s), str(opt_mf))
  _print_str_results(reporter, 'Opt-bounds', str(s_bounds), str(mf_bounds))

  # The marginal likelihoods
  sgp_lml = sgp.compute_log_marginal_likelihood()
  mfgp_lml = mfgp.compute_log_marginal_likelihood()
  _print_float_results(reporter, 'Log_Marg_Like', sgp_lml, mfgp_lml)
  # Train errors
  s_pred, _ = sgp.eval(ZX_tr)
  mf_pred, _ = mfgp.eval_at_fidel(Z_tr, X_tr)
  sgp_tr_err = compute_average_sq_prediction_error(Y_tr, s_pred)
  mfgp_tr_err = compute_average_sq_prediction_error(Y_tr, mf_pred)
  _print_float_results(reporter, 'Train Error', sgp_tr_err, mfgp_tr_err)
  # Test errors
  s_pred, _ = sgp.eval(ZX_te)
  mf_pred, _ = mfgp.eval_at_fidel(Z_te, X_te)
  sgp_te_err = compute_average_sq_prediction_error(Y_te, s_pred)
  mfgp_te_err = compute_average_sq_prediction_error(Y_te, mf_pred)
  _print_float_results(reporter, 'Test Error', sgp_te_err, mfgp_te_err)


if __name__ == '__main__':
  main()

