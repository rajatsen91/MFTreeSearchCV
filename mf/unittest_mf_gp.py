"""
  Unit tests for mf_gp.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=superfluous-parens
# pylint: disable=maybe-no-member
# pylint: disable=abstract-class-not-used

from argparse import Namespace
from copy import deepcopy
import numpy as np
# Local
from gp import kernel
from gp.unittest_gp_instances import fit_gp_with_dataset
import mf_gp
from utils.base_test_class import BaseTestClass, execute_tests
from utils.general_utils import compute_average_sq_prediction_error
from utils.option_handler import load_options


# Functions to create data ---------------------------------------------------------------
def _get_mf_gp_options(tune_noise):
  """ Gets the options for the dataset. """
  options = load_options(mf_gp.all_mf_gp_args)
  options.noise_var_type = 'tune' if tune_noise else options.noise_var_type
  return options

def gen_data_from_func(fzx, N, dim_z, dim_x):
  """ Generates data from the function. """
  Z = np.random.random((N, dim_z))
  X = np.random.random((N, dim_x))
  Y = fzx(Z, X)
  return Z, X, Y

def _gen_datasets_from_func(fzx, N, dim_z, dim_x):
  """ Generates train and test datasets from the function. """
  Z_tr, X_tr, Y_tr = gen_data_from_func(fzx, N, dim_z, dim_x)
  ZX_tr = np.concatenate((Z_tr, X_tr), axis=1)
  Z_te, X_te, Y_te = gen_data_from_func(fzx, 2 * N, dim_z, dim_x)
  ZX_te = np.concatenate((Z_te, X_te), axis=1)
  return Z_tr, X_tr, Y_tr, ZX_tr, Z_te, X_te, Y_te, ZX_te

def gen_mf_gp_test_data():
  """ Generates test data. """
  # pylint: disable=too-many-locals
  # dataset 1
  fzx = lambda z, x: (x**2).sum(axis=1) + (z**2).sum(axis=1)
  dim_z = 2
  dim_x = 3
  N = 20
  Z_tr, X_tr, Y_tr, ZX_tr, Z_te, X_te, Y_te, ZX_te = _gen_datasets_from_func(
    fzx, N, dim_z, dim_x)
  fidel_kernel = kernel.SEKernel(dim=dim_z, scale=1, dim_bandwidths=0.5)
  domain_kernel = kernel.SEKernel(dim=dim_x, scale=1, dim_bandwidths=0.5)
  kernel_scale = 2
  tune_noise = True
  dataset_1 = Namespace(Z_tr=Z_tr, X_tr=X_tr, Y_tr=Y_tr, ZX_tr=ZX_tr,
                        Z_te=Z_te, X_te=X_te, Y_te=Y_te, ZX_te=ZX_te,
                        fidel_kernel=fidel_kernel, domain_kernel=domain_kernel,
                        kernel_scale=kernel_scale, tune_noise=tune_noise)
  # dataset 2
  fx = lambda x: -70 * (x + 0.01) * (x - 0.31) * (x + 0.51) * (x - 0.71) * (x - 0.98)
  fzx = lambda z, x: (np.exp((z - 0.8)**2) * fx(x)).sum(axis=1)
  N = 100
  Z_tr, X_tr, Y_tr, ZX_tr, Z_te, X_te, Y_te, ZX_te = _gen_datasets_from_func(fzx, N, 1, 1)
  fidel_kernel = kernel.SEKernel(dim=1, scale=1, dim_bandwidths=1.0)
  domain_kernel = kernel.SEKernel(dim=1, scale=1, dim_bandwidths=0.25)
  kernel_scale = 2
  tune_noise = True
  dataset_2 = Namespace(Z_tr=Z_tr, X_tr=X_tr, Y_tr=Y_tr, ZX_tr=ZX_tr,
                        Z_te=Z_te, X_te=X_te, Y_te=Y_te, ZX_te=ZX_te,
                        fidel_kernel=fidel_kernel, domain_kernel=domain_kernel,
                        kernel_scale=kernel_scale, tune_noise=tune_noise)
  # return all datasets
  return [dataset_1, dataset_2]

def get_init_and_post_gp(fidel_dim, domain_dim, num_data, kernel_scale, dim_bw_power=0.5):
  """ Generates a GP and data, constructs posterior and returns everything. """
  # pylint: disable=too-many-locals
  kernel_bw_scaling = float(fidel_dim + domain_dim) ** dim_bw_power
  fidel_kernel = kernel.SEKernel(fidel_dim, 1,
                    (1 + np.random.random(fidel_dim)) * kernel_bw_scaling)
  domain_kernel = kernel.SEKernel(domain_dim, 1,
                    (0.1 + 0.2*np.random.random(domain_dim)) * kernel_bw_scaling)
  mean_func_const_val = (2 + np.random.random()) * (1 + np.random.random())
  mean_func = lambda x: np.array([mean_func_const_val] * len(x))
  noise_var = 0.05 * np.random.random()
  Z_init = np.zeros((0, fidel_dim))
  X_init = np.zeros((0, domain_dim))
  Y_init = np.zeros((0))
  init_gp = mf_gp.get_mfgp_from_fidel_domain(Z_init, X_init, Y_init, kernel_scale,
              fidel_kernel, domain_kernel, mean_func, noise_var)
  # Now construct the data
  post_gp = deepcopy(init_gp)
  Z_data = np.random.random((num_data, fidel_dim))
  X_data = np.random.random((num_data, domain_dim))
  Y_wo_noise = post_gp.draw_mf_samples(1, Z_data, X_data).ravel()
  Y_data = Y_wo_noise + np.random.normal(0, np.sqrt(noise_var), Y_wo_noise.shape)
  post_gp.add_mf_data(Z_data, X_data, Y_data)
  # Put everything in a namespace
  ret = Namespace(init_gp=init_gp, post_gp=post_gp, Z_data=Z_data, X_data=X_data,
    Y_data=Y_data, fidel_kernel=fidel_kernel, domain_kernel=domain_kernel,
    mean_func=mean_func, noise_var=noise_var, kernel_scale=kernel_scale,
    fidel_dim=fidel_dim, domain_dim=domain_dim, num_data=num_data)
  return ret

def gen_mf_gp_instances():
  """ Generates some MF GP instances. """
  # pylint: disable=star-args
  # The following list of lists maintains each problem instance in the following
  # order. (fidel_dim, domain_dim, num_data, kernel_scale)
  data = [[1, 1, 5, 1], [2, 4, 15, 2], [5, 10, 100, 2], [3, 20, 200, 4]]
  return [get_init_and_post_gp(*d) for d in data]


# Functions to build GPs -----------------------------------------------------------------
def build_mfgp_with_dataset(dataset):
  """ Builds a MF GP by using some reasonable values for the parameters. """
  mean_func = lambda x: np.array([np.median(dataset.Y_tr)] * len(x))
  noise_var = (dataset.Y_tr.std()**2)/20
  return mf_gp.get_mfgp_from_fidel_domain(dataset.Z_tr, dataset.X_tr, dataset.Y_tr,
           dataset.kernel_scale, dataset.fidel_kernel, dataset.domain_kernel,
           mean_func, noise_var)

def fit_simple_gp_with_dataset(dataset):
  """ Builds a simple GP with the dataset. """
  return fit_gp_with_dataset([dataset.ZX_tr, dataset.Y_tr])

def fit_mfgp_with_dataset(dataset):
  """ Fits a GP with the dataset. """
  options = _get_mf_gp_options(dataset.tune_noise)
  options.mean_func_type = 'median'
  fitted_gp, _ = (mf_gp.MFGPFitter(dataset.Z_tr, dataset.X_tr, dataset.Y_tr,
                   options=options)).fit_gp()
  return fitted_gp


# Test cases -----------------------------------------------------------------------------
class MFGPTestCase(BaseTestClass):
  """ Unit tests for the MF GP. """
  # pylint: disable=too-many-locals

  def setUp(self):
    """ Set up for the tests. """
    self.datasets = gen_mf_gp_test_data()

  def test_eval_at_fidel(self):
    """ Tests eval at fidel. """
    self.report('MFGP.eval_at_fidel vs GP.eval.')
    ds = self.datasets[0]
    curr_gp = build_mfgp_with_dataset(ds)
    curr_pred, curr_std = curr_gp.eval_at_fidel(ds.Z_te, ds.X_te, uncert_form='std')
    alt_pred, alt_std = curr_gp.eval(np.concatenate((ds.Z_te, ds.X_te), axis=1),
                                     uncert_form='std')
    assert np.linalg.norm(curr_pred - alt_pred) < 1e-5
    assert np.linalg.norm(curr_std - alt_std) < 1e-5

  def test_eval(self):
    """ Tests the evaluation. """
    self.report('MFGP.eval_at_fidel: Probabilistic test, might fail sometimes.')
    num_successes = 0
    for ds in self.datasets:
      curr_gp = build_mfgp_with_dataset(ds)
      curr_pred, _ = curr_gp.eval_at_fidel(ds.Z_te, ds.X_te)
      curr_err = compute_average_sq_prediction_error(ds.Y_te, curr_pred)
      const_err = compute_average_sq_prediction_error(ds.Y_te, ds.Y_tr.mean())
      success = curr_err < const_err
      self.report(('(N,DZ,DX)=' + str(ds.Z_tr.shape + (ds.X_tr.shape[1],)) +
                   ':: MFGP-err= ' + str(curr_err) + ',   Const-err= ' + str(const_err) +
                   ',  success=' + str(success)), 'test_result')
      num_successes += int(success)
    assert num_successes > 0.6 *len(self.datasets)

  def test_compute_log_marginal_likelihood(self):
    """ Tests compute_log_marginal_likelihood. Does not test for accurate implementation.
        Only tests if the function runs without runtime errors. """
    self.report('MFGP.compute_log_marginal_likelihood: ** Runtime test errors only **')
    for ds in self.datasets:
      curr_gp = build_mfgp_with_dataset(ds)
      lml = curr_gp.compute_log_marginal_likelihood()
      self.report(('(N,DZ,DX)=' + str(ds.Z_tr.shape + (ds.X_tr.shape[1],)) +
                   ':: MFGP-lml= ' + str(lml)), 'test_result')


class MFGPFitterTestCase(BaseTestClass):
  """ Unit tests for the MFGPFitter class. """
  # pylint: disable=too-many-locals

  def setUp(self):
    """ Set up for the tests. """
    self.datasets = gen_mf_gp_test_data()

  def test_set_up(self):
    """ Tests if everything has been set up properly. """
    for ds in self.datasets:
      options = _get_mf_gp_options(ds.tune_noise)
      fitter = mf_gp.MFGPFitter(ds.Z_tr, ds.X_tr, ds.Y_tr, options)
      # The number of hyperparameters should be 1 for the kernel, the total dimensionality
      # of the fidelity space and domain plus one more if we are tuning options.
      num_hps = 1 + ds.Z_tr.shape[1] + ds.X_tr.shape[1] + ds.tune_noise
      constructed_hp_bounds = ([fitter.scale_log_bounds] +
                               fitter.fidel_bandwidth_log_bounds +
                               fitter.domain_bandwidth_log_bounds)
      if ds.tune_noise:
        constructed_hp_bounds = [fitter.noise_var_log_bounds] + constructed_hp_bounds
      constructed_hp_bounds = np.array(constructed_hp_bounds)
      assert fitter.hp_bounds.shape == (num_hps, 2)
      assert np.linalg.norm(constructed_hp_bounds - fitter.hp_bounds) < 1e-5

  def test_marginal_likelihood(self):
    """ Test for marginal likelihood. """
    self.report(('Marginal Likelihood for fitted MFGP. Probabilistic test, might fail.' +
                 ' The domain bandwidth should be smaller than the fidelity bandwidth ' +
                 'for the second dataset.'))
    num_successes = 0
    for ds in self.datasets:
      naive_gp = build_mfgp_with_dataset(ds)
      fitted_gp = fit_mfgp_with_dataset(ds)
      naive_lml = naive_gp.compute_log_marginal_likelihood()
      fitted_lml = fitted_gp.compute_log_marginal_likelihood()
      success = naive_lml <= fitted_lml
      self.report('(N,DZ,DX)= %s, naive-lml=%0.4f, fitted-lml=%0.4f, succ=%d'%(
        str(ds.Z_tr.shape + (ds.X_tr.shape[1],)), naive_lml, fitted_lml, success),
        'test_result')
      self.report('  Naive GP :: %s'%(str(naive_gp)), 'test_result')
      self.report('  Fitted GP:: %s'%(str(fitted_gp)), 'test_result')
      num_successes += success
    assert num_successes > 0.6 * len(self.datasets)

  def test_eval(self):
    """ Test for prediction. """
    self.report('Prediction for fitted Simple vs MF GPs. Probabilistic test, might fail.')
    num_successes = 0
    for ds in self.datasets:
      simple_gp = fit_simple_gp_with_dataset(ds)
      simple_preds, _ = simple_gp.eval(ds.ZX_te)
      simple_err = compute_average_sq_prediction_error(ds.Y_te, simple_preds)
      fitted_gp = fit_mfgp_with_dataset(ds)
      fitted_preds, _ = fitted_gp.eval_at_fidel(ds.Z_te, ds.X_te)
      fitted_err = compute_average_sq_prediction_error(ds.Y_te, fitted_preds)
      success = abs(fitted_err - simple_err) < 1e-2
      self.report('(N,DZ,DX)= %s, simple-err=%0.4f, mfgp-err=%0.4f, succ=%d'%(
        str(ds.Z_tr.shape + (ds.X_tr.shape[1],)), simple_err, fitted_err, success),
        'test_result')
      num_successes += success
    assert num_successes > 0.6 * len(self.datasets)

  def test_draw_samples(self):
    """ Test for drawing samples. """
    self.report('Test for drawing samples. Probabilistic test, might fail.')
    total_coverage = 0
    num_test_pts = 100
    num_samples = 5 # Draw 5 samples at each point - just for testing.
    mfgp_instances = gen_mf_gp_instances()
    for inst in mfgp_instances:
      Z_test = np.random.random((num_test_pts, inst.fidel_dim))
      X_test = np.random.random((num_test_pts, inst.domain_dim))
      F_test = inst.post_gp.draw_mf_samples(num_samples, Z_test, X_test)
      post_mean, post_std = inst.post_gp.eval_at_fidel(Z_test, X_test, uncert_form='std')
      conf_band_width = 1.96
      ucb = post_mean + conf_band_width * post_std
      lcb = post_mean - conf_band_width * post_std
      below_ucb = F_test <= ucb
      above_lcb = F_test >= lcb
      coverage = (below_ucb * above_lcb).mean()
      total_coverage += coverage
      self.report(('(n, DZ, DX) = (%d, %d, %d)::: Coverage for 0.95 credible interval: ' +
                   '%0.4f')%(inst.num_data, inst.fidel_dim, inst.domain_dim, coverage),
                  'test_result')
    avg_coverage = total_coverage / len(mfgp_instances)
    avg_coverage_is_good = avg_coverage > 0.9
    self.report('Avg coverage (%0.3f) is larger than 0.9? %d'%(avg_coverage,
                avg_coverage_is_good), 'test_result')
    assert avg_coverage_is_good


if __name__ == '__main__':
  execute_tests()

