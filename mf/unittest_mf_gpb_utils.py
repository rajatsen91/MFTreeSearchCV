"""
  Unit tests for MF-GP-Bandit Utilities.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=super-on-old-class

from copy import deepcopy
from argparse import Namespace
import numpy as np
# Local
from gen_mfgp_sample import gen_simple_mfgp_as_mfof
from mf_func import MFOptFunction
import mf_gpb_utils
from utils.base_test_class import BaseTestClass, execute_tests
from unittest_mf_gp import get_init_and_post_gp
from utils.ancillary_utils import is_non_decreasing_sequence

def _get_mfgp_instances(prob_params=None):
  """ Generates a bunch of GP-UCB instances. """
  # pylint: disable=star-args
  # The following list of lists maintains each problem instance in the following
  # order. (fidel_dim, domain_dim, num_data, kernel_scale, dim_bw_power)
  if prob_params is None:
    prob_params = [[1, 2, 40, 2, 0.5], [2, 4, 10, 1, 0], [4, 20, 40, 10, 0],
                   [3, 10, 10, 4, 0.5]]
  instances = [get_init_and_post_gp(*prob) for prob in prob_params]
  for inst in instances:
    inst.opt_fidel = 0.9 + 0.1 * np.random.random((inst.fidel_dim))
  instances[-1].opt_fidel = np.ones((instances[-1].fidel_dim))
  return instances


class MFGPBUtilsTestCase(BaseTestClass):
  """ Unit tests for mf_gpb_utils.py """

  def setUp(self):
    """ Sets up unit tests. """
    self.lhs_data = [(1, 10), (2, 5), (4, 10), (10, 100)]

  def test_latin_hc_indices(self):
    """ Tests latin hyper-cube index generation. """
    self.report('Test Latin hyper-cube indexing. Only a sufficient condition check.')
    for data in self.lhs_data:
      lhs_true_sum = data[1] * (data[1] - 1) / 2
      lhs_idxs = mf_gpb_utils.latin_hc_indices(data[0], data[1])
      lhs_idx_sums = np.array(lhs_idxs).sum(axis=0)
      assert np.all(lhs_true_sum == lhs_idx_sums)

  def test_latin_hc_sampling(self):
    """ Tests latin hyper-cube sampling. """
    self.report('Test Latin hyper-cube sampling. Only a sufficient condition check.')
    for data in self.lhs_data:
      lhs_max_sum = float(data[1] + 1)/2
      lhs_min_sum = float(data[1] - 1)/2
      lhs_samples = mf_gpb_utils.latin_hc_sampling(data[0], data[1])
      lhs_sample_sums = lhs_samples.sum(axis=0)
      assert lhs_sample_sums.max() <= lhs_max_sum
      assert lhs_sample_sums.min() >= lhs_min_sum


class AcquisitionTestCase(BaseTestClass):
  """ Test class for the Acquisitions. """

  def test_mf_gp_ucb_1(self):
    """ Tests the mf-gp-ucb acquisition using a sample from gen_mfgp_sample_as_mfof. """
    self.report('Tests mf-gp-ucb acquisition using sample from gen_simple_mfgp_as_mfof.')
    mfof = gen_simple_mfgp_as_mfof(random_seed=np.random.randint(1000))
    mfgp = deepcopy(mfof.mfgp)
    report_time_steps = set([int(x) for x in
                             np.logspace(np.log10(5), np.log10(1000), 20)])
    report_time_steps = sorted(list(report_time_steps))
    prev_time = 0
    num_test_pts = 200
    losses = []

    for t in report_time_steps:
      # Add new points to the GP
      num_new_points = t - prev_time
      Z_new = np.random.random((num_new_points, mfof.fidel_dim))
      X_new = np.random.random((num_new_points, mfof.domain_dim))
      Y_new = mfof.eval_multiple(Z_new, X_new)
      mfgp.add_mf_data(Z_new, X_new, Y_new)
      prev_time = t
      # Tests
      X_test = np.random.random((num_test_pts, mfof.domain_dim))
      opt_fidel_mat = np.repeat(mfof.opt_fidel.reshape(1, -1), num_test_pts, axis=0)
      F_test = mfof.eval_multiple(opt_fidel_mat, X_test)
      ucb_test, _ = mf_gpb_utils.acquisitions.mf_gp_ucb_multiple(X_test, mfgp,
                      mfof.opt_fidel, t)
      below_ucb = F_test < ucb_test
      coverage = below_ucb.mean()
      losses.append(1-coverage)
    total_loss = sum(losses)
    assert total_loss < 0.01
    result_str = '   (DZ, DX) = (%d, %d) Loss for this instance: %0.4f'%(
      mfof.fidel_dim, mfof.domain_dim, total_loss)
    self.report(result_str, 'test_result')


  def test_mf_gp_ucb_2(self):
    """ Tests the mf-gp-ucb acquisition. """
    # pylint: disable=too-many-locals
    self.report('Testing Coverage of mf-gp-ucb Acq. Probabilistic test, might fail.')
    report_time_steps = set([int(x) for x in
                             np.logspace(np.log10(5), np.log10(1000), 20)])
    report_time_steps = sorted(list(report_time_steps))
    instances = _get_mfgp_instances()
    all_losses = []

    # Now run test
    for inst in instances:
      num_test_points = 100 * inst.domain_dim
      post_gp = inst.post_gp
      inst_losses = []
      beta_th_vals = []
      prev_time = 0
      for t in report_time_steps:
        # First add new points to the GP.
        num_new_points = t - prev_time
        X_new = np.random.random((num_new_points, inst.domain_dim))
        Z_new = np.random.random((num_new_points, inst.fidel_dim))
        Y_new = (inst.post_gp.draw_mf_samples(1, Z_new, X_new).ravel() +
                 np.random.normal(0, np.sqrt(inst.noise_var), (num_new_points,)))
        inst.post_gp.add_mf_data(Z_new, X_new, Y_new, rebuild=True)
        prev_time = t
        # Now  do the tests.
        assert post_gp.num_tr_data == t + inst.num_data
        X_test = np.random.random((num_test_points, inst.domain_dim))
        opt_fidel_mat = np.repeat(inst.opt_fidel.reshape(1, -1), num_test_points, axis=0)
        F_test = post_gp.draw_mf_samples(1, opt_fidel_mat, X_test).ravel()
        ucb_test, beta_th = mf_gpb_utils.acquisitions.mf_gp_ucb_multiple(X_test,
                             inst.post_gp, inst.opt_fidel, t)
        below_ucb = F_test < ucb_test
        coverage = below_ucb.mean()
        inst_losses.append(1 - coverage)
        beta_th_vals.append(beta_th)
        # manually compute coverage
        mu, sigma = post_gp.eval_at_fidel(opt_fidel_mat, X_test, uncert_form='std')
        manual_conf = mu + beta_th * sigma
        manual_coverage = (F_test < manual_conf).mean()
        assert manual_coverage == coverage
      # Report results for this instance
      inst_result_str = ', '.join('%d: %0.3f (%0.2f)'%(report_time_steps[i],
        inst_losses[i], beta_th_vals[i]) for i in range(len(report_time_steps)))
      inst_result_str = '(DZ, DX) = (%d, %d):: coverage %s'%(inst.fidel_dim,
                          inst.domain_dim, inst_result_str)
      self.report(inst_result_str, 'test_result')
      total_inst_loss = sum(inst_losses)
      all_losses.append(total_inst_loss)
      inst_avg_result_str = '   (DZ, DX) = (%d, %d) Loss for this instance: %0.4f'%(
        inst.fidel_dim, inst.domain_dim, total_inst_loss)
      self.report(inst_avg_result_str, 'test_result')
      assert is_non_decreasing_sequence(beta_th_vals)
      assert np.all(beta_th_vals <=
                    2 * inst.domain_dim * np.sqrt(np.log(report_time_steps)))

    # Final accumulation
    avg_inst_loss = np.array(all_losses).mean()
    loss_thresh = 0.02
    avg_loss_is_good = avg_inst_loss < loss_thresh
    self.report('Avg loss (%0.3f) is smaller than %f? %d'%(avg_inst_loss,
                loss_thresh, avg_loss_is_good), 'test_result')
    assert avg_loss_is_good


class FidelityChoosersTestCase(BaseTestClass):
  """ Test class for the Acquisitions. """

  @classmethod
  def _get_mfof_obj(cls, fidel_dim, domain_dim, opt_fidel):
    """ Returns an MFOPTFunction object. """
    g = lambda z, x: (x**2).sum(axis=1) + ((2*z)**2).sum(axis=1)
    cost = lambda z: 1 + (z**1.5).sum(axis=1)
    vectorised = True
    fidel_bounds = np.array([[0, 1]] * fidel_dim)
    domain_bounds = np.array([[0, 1]] * domain_dim)
    return MFOptFunction(g, cost, fidel_bounds, domain_bounds, opt_fidel, vectorised)

  def test_mf_gp_ucb(self):
    """" Tests the mf-gp-ucb acquisition chooser. """
    # pylint: disable=too-many-locals
    self.report('Testing mf-gp-ucb Fidelity chooser.')
    prob_params = [[1, 2, 40, 2, 0.5], [3, 10, 10, 4, 0.5], [4, 20, 100, 1.5, 0.1]]
    num_next_pts = 5
    instances = _get_mfgp_instances(prob_params)

    for inst in instances:
      next_pts = np.random.random((num_next_pts, inst.domain_dim))
      curr_time = inst.post_gp.num_tr_data
      _, beta_th = mf_gpb_utils.acquisitions.mf_gp_ucb_multiple(next_pts,
                    inst.post_gp, inst.opt_fidel, curr_time)
      thresh_coeff = 0.5 + 0.5 * np.random.random()
      acq_params = Namespace(beta_th=beta_th, thresh_coeff=thresh_coeff)
      for next_pt in next_pts:
        mfof = self._get_mfof_obj(inst.post_gp.fidel_dim, inst.post_gp.domain_dim,
                                  inst.opt_fidel)
        # Determine the next fidelity
        next_fidel = mf_gpb_utils.fidelity_choosers.mf_gp_ucb(next_pt,
                       inst.post_gp, mfof, acq_params)
        # Compute the threshold independently
        cost_ratio_power = 1/float(inst.post_gp.domain_dim + inst.post_gp.fidel_dim + 2)
        std_thresh = thresh_coeff * (mfof.get_cost_ratio(next_fidel) ** cost_ratio_power *
          np.sqrt(inst.post_gp.kernel.scale) *
          inst.post_gp.fidel_kernel.compute_std_slack(next_fidel.reshape(1, -1),
                                                      inst.opt_fidel.reshape(1, -1)))
        # Compute the std
        _, next_fidel_std = inst.post_gp.eval_at_fidel(next_fidel.reshape(1, -1),
                              next_pt.reshape(1, -1), uncert_form='std')
        next_fidel_std = float(next_fidel_std)
        # Test
        is_opt_fidel = mf_gpb_utils.is_an_opt_fidel_query(next_fidel, inst.opt_fidel)
        is_larger_than_thresh = next_fidel_std >= std_thresh
        self.report(('(DZ, DX, n) = (%d, %d, %d):: threshold: %0.4f, std: %0.4f ' +
                     'is_larger_than_thresh: %d, is_opt_fidel:  %d')%(inst.fidel_dim,
                      inst.domain_dim, inst.post_gp.num_tr_data, std_thresh,
                      next_fidel_std, is_larger_than_thresh, is_opt_fidel),
                    'test_result')
        assert  is_opt_fidel or is_larger_than_thresh


if __name__ == '__main__':
  execute_tests()

