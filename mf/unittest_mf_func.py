"""
  Unit tests for mf_func.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=superfluous-parens
# pylint: disable=maybe-no-member

from argparse import Namespace
import numpy as np
# Local
import mf_func
from utils.base_test_class import BaseTestClass, execute_tests
from utils.general_utils import map_to_bounds


# Functions for preparing data -----------------------------------------------------------
def get_mf_func_data():
  """ Prepares data. """
  # pylint: disable=too-many-locals

  # Function 1 - vectorised
  g_1 = lambda z, x: (x**2).sum(axis=1) + ((2*z)**2).sum(axis=1)
  cost = lambda z: z[:, 0] * (z[:, 1]**2)
  vectorised = True
  fidel_bounds = np.array([[1, 4], [5, 6]])
  domain_bounds = np.array([[-1, 2], [0, 1], [1, 11]])
  dz = len(fidel_bounds)
  dx = len(domain_bounds)
  opt_fidel = np.array([3.2, 5.8])
  opt_fidel_cost = float(cost(opt_fidel.reshape((1, dz))))
  mff = mf_func.MFFunction(g_1, cost, fidel_bounds, domain_bounds, vectorised)
  mfof = mf_func.MFOptFunction(g_1, cost, fidel_bounds, domain_bounds, opt_fidel,
                                vectorised)
  func_1 = Namespace(g=g_1, cost=cost, dz=dz, dx=dx, vectorised=vectorised,
                     fidel_bounds=fidel_bounds, domain_bounds=domain_bounds, mfof=mfof,
                     mff=mff, opt_fidel=opt_fidel, opt_fidel_cost=opt_fidel_cost)
  # Function 2 - Same as Function 1 but we ravel
  g_2 = lambda z, x: (g_1(z, x)).ravel()
  func_2 = Namespace(g=g_2, cost=cost, dz=dz, dx=dx, vectorised=vectorised,
                     fidel_bounds=fidel_bounds, domain_bounds=domain_bounds, mfof=mfof,
                     mff=mff, opt_fidel=opt_fidel, opt_fidel_cost=opt_fidel_cost)

  # Function 3 - not vectorised
  g_3 = lambda z, x: np.cos(z**2) * (np.sin(x)).sum()
  cost = lambda z: z[0]**3
  dz = 1
  dx = 3
  vectorised = False
  fidel_bounds = np.array([[3, 6]])
  domain_bounds = np.array([[-4, 2], [-1, 4], [21, 41]])
  dz = len(fidel_bounds)
  dx = len(domain_bounds)
  opt_fidel = np.array([5.7])
  opt_fidel_cost = float(cost(opt_fidel))
  mff = mf_func.MFFunction(g_3, cost, fidel_bounds, domain_bounds, vectorised)
  mfof = mf_func.MFOptFunction(g_3, cost, fidel_bounds, domain_bounds, opt_fidel,
                               vectorised)
  func_3 = Namespace(g=g_3, cost=cost, dz=dz, dx=dx, vectorised=vectorised,
                     fidel_bounds=fidel_bounds, domain_bounds=domain_bounds, mfof=mfof,
                     mff=mff, opt_fidel=opt_fidel, opt_fidel_cost=opt_fidel_cost)
  # Function 4 - not vectorised
  g_4 = lambda z, x: float(g_3(z, x))
  func_4 = Namespace(g=g_4, cost=cost, dz=dz, dx=dx, vectorised=vectorised,
                     fidel_bounds=fidel_bounds, domain_bounds=domain_bounds, mfof=mfof,
                     mff=mff, opt_fidel=opt_fidel, opt_fidel_cost=opt_fidel_cost)

  # Return all functions
  return [func_1, func_2, func_3, func_4]


# Some functions we will need for testing ------------------------------------------------
def _get_test_points(dz, dx, z_bounds, x_bounds, n=5):
  """ Gets test points. """
  single_nz = np.random.random(dz)
  single_nx = np.random.random(dx)
  mult_nz = np.random.random((n, dz))
  mult_nx = np.random.random((n, dx))
  single_z = map_to_bounds(single_nz, z_bounds)
  single_x = map_to_bounds(single_nx, x_bounds)
  mult_z = map_to_bounds(mult_nz, z_bounds)
  mult_x = map_to_bounds(mult_nx, x_bounds)
  return (single_nz, single_nx, single_z, single_x,
          mult_nz, mult_nx, mult_z, mult_x)

def _get_gvals(single_z, single_x, mult_z, mult_x, func):
  """ Evaluates the function at the test points and returns the values. """
  if func.vectorised:
    single_gvals = float(func.g(single_z.reshape((1, func.dz)),
                                single_x.reshape((1, func.dx))))
    mult_gvals = func.g(mult_z, mult_x).ravel()
  else:
    single_gvals = float(func.g(single_z, single_x))
    mult_gvals = []
    for i in range(len(mult_z)):
      mult_gvals.append(float(func.g(mult_z[i, :], mult_x[i, :])))
    mult_gvals = np.array(mult_gvals)
  return single_gvals, mult_gvals

def _get_mff_vals_unnorm(single_z, single_x, mult_z, mult_x, func):
  """ Evaluates mff at the test points with unnormalised coordiantes. """
  single_mff_vals = func.mff.eval_at_fidel_single_point(single_z, single_x)
  mult_mff_vals = func.mff.eval_at_fidel_multiple_points(mult_z, mult_x)
  return single_mff_vals, mult_mff_vals

def _get_mff_vals_norm(single_nz, single_nx, mult_nz, mult_nx, func):
  """ Evaluates mff at the test points with unnormalised coordiantes. """
  single_mff_vals = func.mff.eval_at_fidel_single_point_normalised(single_nz, single_nx)
  mult_mff_vals = func.mff.eval_at_fidel_multiple_points_normalised(mult_nz, mult_nx)
  return single_mff_vals, mult_mff_vals

def _get_cost_vals(single_z, mult_z, func):
  """ Evaluates the function at the test points and returns the values. """
  if func.vectorised:
    single_gvals = float(func.cost(single_z.reshape((1, func.dz))))
    mult_gvals = func.cost(mult_z).ravel()
  else:
    single_gvals = float(func.cost(single_z))
    mult_gvals = []
    for i in range(len(mult_z)):
      mult_gvals.append(func.cost(mult_z[i, :]))
    mult_gvals = np.array(mult_gvals)
  return single_gvals, mult_gvals

def _get_mff_cost_vals_unnorm(single_z, mult_z, func):
  """ Evaluates mff for cost at the test points with unnormalised coordiantes. """
  single_mff_cost_vals = func.mff.eval_fidel_cost_single_point(single_z)
  mult_mff_cost_vals = func.mff.eval_fidel_cost_multiple_points(mult_z)
  return single_mff_cost_vals, mult_mff_cost_vals

def _get_mff_cost_vals_norm(single_nz, mult_nz, func):
  """ Evaluates mff for cost at the test points with unnormalised coordiantes. """
  single_mff_cost_vals = func.mff.eval_fidel_cost_single_point_normalised(single_nz)
  mult_mff_cost_vals = func.mff.eval_fidel_cost_multiple_points_normalised(mult_nz)
  return single_mff_cost_vals, mult_mff_cost_vals


# Test Cases -----------------------------------------------------------------------------
class MFFunctionTestCase(BaseTestClass):
  """ Unit tests for MFFunction. """
  # pylint: disable=too-many-locals

  def setUp(self):
    """ Set up for the tests. """
    self.functions = get_mf_func_data()

  def test_eval(self):
    """ Tests evaluation at single and multiple points using normalised and unnormalised
        coordinates """
    self.report(('Test eval at single/multiple points using normalised/unnormalised ' +
                 'coordinates.'))
    for func in self.functions:
      single_nz, single_nx, single_z, single_x, mult_nz, mult_nx, mult_z, mult_x = \
        _get_test_points(func.dz, func.dx, func.fidel_bounds, func.domain_bounds)
      single_gvals, mult_gvals = _get_gvals(single_z, single_x, mult_z, mult_x, func)
      single_n_mffvals, mult_n_mffvals = _get_mff_vals_norm(single_nz, single_nx,
                                                            mult_nz, mult_nx, func)
      single_mffvals, mult_mffvals = _get_mff_vals_unnorm(single_z, single_x,
                                                          mult_z, mult_x, func)
      assert abs(single_n_mffvals - single_gvals) < 1e-5
      assert abs(single_mffvals - single_gvals) < 1e-5
      assert np.linalg.norm(mult_n_mffvals - mult_gvals) < 1e-5
      assert np.linalg.norm(mult_mffvals - mult_gvals) < 1e-5

  def test_cost_eval(self):
    """ Tests evaluation of the cost function at single and multiple points using
        normalised and unnormalised coordinates """
    self.report(('Test evaluation of cost function at single/multiple points using' +
                 ' normalised/unnormalised coordinates.'))
    for func in self.functions:
      single_nz, _, single_z, _, mult_nz, _, mult_z, _ = \
        _get_test_points(func.dz, func.dx, func.fidel_bounds, func.domain_bounds)
      single_cost_vals, mult_cost_vals = _get_cost_vals(single_z, mult_z, func)
      single_n_mff_cost_vals, mult_n_mff_cost_vals = _get_mff_cost_vals_norm(single_nz,
                                                       mult_nz, func)
      single_mff_cost_vals, mult_mff_cost_vals = _get_mff_cost_vals_unnorm(single_z,
                                                       mult_z, func)
      assert abs(single_n_mff_cost_vals - single_cost_vals) < 1e-5
      assert abs(single_mff_cost_vals - single_cost_vals) < 1e-5
      assert np.linalg.norm(mult_n_mff_cost_vals - mult_cost_vals) < 1e-5
      assert np.linalg.norm(mult_mff_cost_vals - mult_cost_vals) < 1e-5


class MFOptFunctionTestCase(BaseTestClass):
  """ Unit tests for MFOptFunction. """
  # pylint: disable=too-many-locals

  def setUp(self):
    """ Set up for the tests. """
    self.functions = get_mf_func_data()

  def test_cost_ratio(self):
    """ Tests evaluation of cost ratio. """
    self.report('Testing cost ratio.')
    for func in self.functions:
      single_nz, _, single_z, _, mult_nz, _, mult_z, _ = \
        _get_test_points(func.dz, func.dx, func.fidel_bounds, func.domain_bounds)
      single_cost_vals, mult_cost_vals = _get_cost_vals(single_z, mult_z, func)
      single_cost_ratios = single_cost_vals / func.opt_fidel_cost
      mult_cost_ratios = mult_cost_vals / func.opt_fidel_cost
      single_mff_crs = func.mfof.get_cost_ratio(single_nz)
      mult_mff_crs = func.mfof.get_cost_ratio(mult_nz)
      assert abs(single_mff_crs - single_cost_ratios) < 1e-5
      assert np.linalg.norm(mult_mff_crs - mult_cost_ratios) < 1e-5

  def test_eval(self):
    """ Tests evaluation. """
    self.report('Testing evaluation.')
    for func in self.functions:
      single_nz, single_nx, single_z, single_x, mult_nz, mult_nx, mult_z, mult_x = \
        _get_test_points(func.dz, func.dx, func.fidel_bounds, func.domain_bounds)
      single_gvals, mult_gvals = _get_gvals(single_z, single_x, mult_z, mult_x, func)
      single_n_mffvals = func.mfof.eval(single_nz, single_nx)
      mult_n_mffvals = func.mfof.eval(mult_nz, mult_nx)
      assert abs(single_n_mffvals - single_gvals) < 1e-5
      assert np.linalg.norm(mult_n_mffvals - mult_gvals) < 1e-5

  def test_get_candidate_fidels(self):
    """ Tests obtaining candidate fidelities. """
    self.report('Testing obtaining of candidate fidelities.')
    mf_g = self.functions[0].g  # This has to be vectorised !!!
    domain_bounds = self.functions[0].domain_bounds
    dim_vals = [1, 2, 3, 5, 10]
    for dim in dim_vals:
      fidel_bounds = [[0, 1]] * dim
      opt_fidel = np.random.random(dim) * 0.3 + 0.5
      mf_cost = lambda z: (z**1.5 * (np.array(range(dim)) + 0.1)).sum(axis=1)
      mfof = mf_func.MFOptFunction(mf_g, mf_cost, fidel_bounds, domain_bounds, opt_fidel,
                                    vectorised=True)
      filt_candidates = mfof.get_candidate_fidelities()
      raw_candidates = mfof.get_candidate_fidelities(filter_by_cost=False)
      num_filt_cands = len(filt_candidates)
      num_raw_cands = len(raw_candidates)
      filt_cost_ratios = mfof.get_cost_ratio(filt_candidates)
      filt_equal = (filt_cost_ratios == 1.0).sum()
      filt_less = (filt_cost_ratios < 1.0).sum()
      # Tests
      assert len(filt_candidates.shape) == 2
      assert len(raw_candidates.shape) == 2
      assert filt_candidates.shape[1] == mfof.fidel_dim
      assert raw_candidates.shape[1] == mfof.fidel_dim
      assert num_filt_cands <= num_raw_cands
      assert filt_equal == 1
      assert filt_less == num_filt_cands - 1


class NoisyMFOptFunctionTestCase(BaseTestClass):
  """ Unit tests for the NoisyMFOptFunction class. """

  def setUp(self):
    """ Set up for the tests. """
    self.functions = get_mf_func_data()

  def test_noisy_eval(self):
    """ Tests evaluation. """
    self.report('Testing Noisy evaluation. Probabilisitic test, might fail.')
    for func in self.functions:
      curr_noise_var = 0.2 + 0.3 * np.random.random()
      curr_noise_std = np.sqrt(curr_noise_var)
      single_nz, single_nx, single_z, single_x, mult_nz, mult_nx, mult_z, mult_x = \
        _get_test_points(func.dz, func.dx, func.fidel_bounds, func.domain_bounds, n=10000)
      single_gvals, mult_gvals = _get_gvals(single_z, single_x, mult_z, mult_x, func)
      # Now get noisy values
      noisy_mfof = mf_func.get_noisy_mfof_from_mfof(func.mfof, curr_noise_var, 'gauss')
      noisy_single_n_mffvals = noisy_mfof.eval(single_nz, single_nx)
      noisy_mult_n_mffvals = noisy_mfof.eval(mult_nz, mult_nx)
      mult_diff_std = (noisy_mult_n_mffvals - mult_gvals).std()
      self.report('Noisy test single: true: %0.4f, noisy: %0.4f'%(single_gvals,
                  noisy_single_n_mffvals), 'test_result')
      self.report('Noisy test multiple: true-std: %0.4f, est-std: %0.4f'%(
                  curr_noise_std, mult_diff_std), 'test_result')
      assert abs(noisy_single_n_mffvals - single_gvals) < 5 * curr_noise_std
      assert abs(mult_diff_std - curr_noise_std) < 0.05


if __name__ == '__main__':
  execute_tests()

