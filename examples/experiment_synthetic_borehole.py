"""
  Running experiments for the synthetic functions.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=abstract-class-not-used


from argparse import Namespace
import numpy as np
# Local imports
from mf.mf_gp_bandit import all_mf_gp_bandit_args
from mf.mfopt_experimenters import MFOptExperimenter
from mf.mf_func import get_noisy_mfof_from_mfof
from mf.gen_mfgp_sample import gen_simple_mfgp_as_mfof
import synthetic_functions
from utils.option_handler import load_options
from utils.reporters import get_reporter

NOISY = True
#NOISY = False

# Debug or not
#IS_DEBUG = True
IS_DEBUG = False

# Choose experiment
# EXP_NAME = 'GP'
# EXP_NAME = 'GP-Bad-Approx'
# EXP_NAME = 'Hartmann3'
# EXP_NAME = 'Hartmann3b'
# EXP_NAME = 'Hartmann3c'
#EXP_NAME = 'Hartmann6'
# EXP_NAME = 'Hartmann6b'
#EXP_NAME = 'CurrinExp'
EXP_NAME = 'Branin'
#EXP_NAME = 'Borehole'

# Set parameters
# NONFINITE_METHODS = ['mf_gp_ucb', 'gp_ucb', 'gp_ei', 'direct']
NONFINITE_METHODS = ['mf_gp_ucb', 'gp_ucb', 'gp_ei']
# NONFINITE_METHODS = ['gp_ucb']
# NONFINITE_METHODS = ['gp_ucb', 'direct']
FINITE_METHODS = ['mf_gp_ucb_finite', 'mf_sko']
# FINITE_METHODS = ['mf_gp_ucb_finite']
# FINITE_METHODS = []
NUM_EXPERIMENTS = 10
SAVE_RESULTS_DIR = './examples/results'

def get_problem_parameters(options):
  """ Returns the problem parameters. """
  prob = Namespace()
  if EXP_NAME == 'GP':
    mfof = gen_simple_mfgp_as_mfof(fidel_bw=1)
    mfof.init_mfgp = mfof.mfgp
    max_capital = 20 * mfof.opt_fidel_cost
    noise_var = 0.05
  elif EXP_NAME == 'GP-Bad-Approx':
    mfof = gen_simple_mfgp_as_mfof(fidel_bw=0.01)
    mfof.init_mfgp = mfof.mfgp
    max_capital = 20 * mfof.opt_fidel_cost
    noise_var = 0.01
  elif EXP_NAME == 'Hartmann3':
    mfof = synthetic_functions.get_mf_hartmann_as_mfof(1, 3)
    max_capital = 200 * mfof.opt_fidel_cost
    noise_var = 0.01
  elif EXP_NAME == 'Hartmann3b':
    mfof = synthetic_functions.get_mf_hartmann_as_mfof(2, 3)
    max_capital = 200 * mfof.opt_fidel_cost
    noise_var = 0.05
  elif EXP_NAME == 'Hartmann3c':
    mfof = synthetic_functions.get_mf_hartmann_as_mfof(4, 3)
    max_capital = 200 * mfof.opt_fidel_cost
    noise_var = 0.05
  elif EXP_NAME == 'Hartmann6':
    mfof = synthetic_functions.get_mf_hartmann_as_mfof(1, 6)
    max_capital = 200 * mfof.opt_fidel_cost
    noise_var = 0.05
  elif EXP_NAME == 'Hartmann6b':
    mfof = synthetic_functions.get_mf_hartmann_as_mfof(2, 6)
    max_capital = 200 * mfof.opt_fidel_cost
    noise_var = 0.05
  elif EXP_NAME == 'CurrinExp':
    mfof = synthetic_functions.get_mf_currin_exp_as_mfof()
    max_capital = 200 * mfof.opt_fidel_cost
    noise_var = 0.5
  elif EXP_NAME == 'Branin':
    mfof = synthetic_functions.get_mf_branin_as_mfof(1)
    max_capital = 200 * mfof.opt_fidel_cost
    noise_var = 0.05
  elif EXP_NAME == 'Borehole':
    mfof = synthetic_functions.get_mf_borehole_as_mfof()
    max_capital = 200 * mfof.opt_fidel_cost
    noise_var = 5

  # Add finite fidels
  options.finite_fidels = np.array([[0.333] * mfof.fidel_dim, [0.667] * mfof.fidel_dim])
  options.finite_fidels_is_normalised = True

  # If NOISY, get noisy version
  if NOISY:
    mfof = get_noisy_mfof_from_mfof(mfof, noise_var)

  # is debug
  if IS_DEBUG:
    max_capital = 20 * mfof.opt_fidel_cost
    num_experiments = 3
    experiment_name = 'debug-%s'%(EXP_NAME)
  else:
    experiment_name = EXP_NAME
    num_experiments = NUM_EXPERIMENTS

  # Return everything in this namespace
  prob = Namespace(mfof=mfof, max_capital=max_capital, noisy=NOISY,
                   num_experiments=num_experiments, experiment_name=experiment_name)
  return prob, options


def main():
  """ Main function. """
  options = load_options(all_mf_gp_bandit_args)
  prob, options = get_problem_parameters(options)

  # Set other variables
  all_methods = NONFINITE_METHODS + FINITE_METHODS
  method_options = {key: options for key in all_methods}
  noisy_str = 'noiseless' if not NOISY else 'noisy%0.3f'%(prob.mfof.noise_var)
  save_file_prefix = '%s-%s-p%d-d%d'%(prob.experiment_name, noisy_str,
                                      prob.mfof.fidel_dim,
                                      prob.mfof.domain_dim)
  reporter = get_reporter('default')

  experimenter = MFOptExperimenter(prob.experiment_name, prob.mfof, prob.max_capital,
                         all_methods, prob.num_experiments, SAVE_RESULTS_DIR,
                         save_file_prefix=save_file_prefix,
                         method_options=method_options,
                         method_reporter=reporter,
                         reporter=reporter)
  experimenter.run_experiments()


if __name__ == '__main__':
  main()
