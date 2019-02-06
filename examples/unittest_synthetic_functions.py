"""
  Test cases for the functions in synthetic_functions.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=super-on-old-class

import numpy as np
from synthetic_functions import get_mf_hartmann_as_mfof
from utils.base_test_class import BaseTestClass, execute_tests


class SyntheticExamplesTestCase(BaseTestClass):
  """Unit test class for general utilities. """

  def __init__(self, *args, **kwargs):
    super(SyntheticExamplesTestCase, self).__init__(*args, **kwargs)

  def test_hartmann(self):
    """ Unit tests for the hartmann function. """
    self.report('Testing Hartmann function in 3 and 6 dimensions.')
    # The data are in the following order:
    test_data = [(3, 1), (6, 1), (3, 2), (6, 4)]

    for data in test_data:
      fidel_dim = data[1]
      domain_dim = data[0]
      num_test_pts = 1000 * domain_dim
      mfof = get_mf_hartmann_as_mfof(fidel_dim, domain_dim)
      # True max value
      if domain_dim == 3:
        true_opt_val = 3.86278
      elif domain_dim == 6:
        true_opt_val = 3.322368
      else:
        del true_opt_val
      computed_opt_val = mfof.eval_single(mfof.opt_fidel, mfof.opt_pt)
      # Evaluate at random points at the highest fidelity.
      X_test = np.random.random((num_test_pts, domain_dim))
      opt_fidel_mat = np.repeat(mfof.opt_fidel.reshape(1, -1), num_test_pts, axis=0)
      F_high_test = mfof.eval_multiple(opt_fidel_mat, X_test)
      max_F_high_test = max(F_high_test)
      # Tests across multiple fidelities.
      fidel_mid_mat = 0.5 * np.ones((num_test_pts, fidel_dim))
      fidel_low_mat = np.zeros((num_test_pts, fidel_dim))
      F_mid_test = mfof.eval_multiple(fidel_mid_mat, X_test)
      F_low_test = mfof.eval_multiple(fidel_low_mat, X_test)
      diff_mid = np.linalg.norm(F_high_test - F_mid_test) / num_test_pts
      diff_low = np.linalg.norm(F_high_test - F_low_test) / num_test_pts
      # Tests
      self.report(('(DZ, DX)=(%d, %d):: max(true, computed, test): (%f, %f, %f), ' +
                   'diff_mid: %0.4f, diff_low: %0.4f')%(fidel_dim, domain_dim,
                  true_opt_val, computed_opt_val, max_F_high_test, diff_mid, diff_low),
                  'test_result')
      assert np.abs(true_opt_val - computed_opt_val) < 1e-5
      assert computed_opt_val > max_F_high_test
      assert diff_mid < diff_low


if __name__ == '__main__':
  execute_tests()

