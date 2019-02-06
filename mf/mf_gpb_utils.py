"""
  A collection of utilities for MF-GP Bandits.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=super-on-old-class
# pylint: disable=no-name-in-module

from argparse import Namespace
from copy import deepcopy
import numpy as np
from scipy.stats import norm
from scratch.get_finite_fidel_mfof import mf_sko_fidel_chooser_single


def latin_hc_indices(dim, num_samples):
  """ Obtains indices for Latin Hyper-cube sampling. """
  index_set = [list(range(num_samples))] * dim
  lhs_indices = []
  for i in range(num_samples):
    curr_idx_idx = np.random.randint(num_samples-i, size=dim)
    curr_idx = [index_set[j][curr_idx_idx[j]] for j in range(dim)]
    index_set = [index_set[j][:curr_idx_idx[j]] + index_set[j][curr_idx_idx[j]+1:]
                 for j in range(dim)]
    lhs_indices.append(curr_idx)
  return lhs_indices

def latin_hc_sampling(dim, num_samples):
  """ Latin Hyper-cube sampling in the unit hyper-cube. """
  if num_samples == 0:
    return np.zeros((0, dim))
  elif num_samples == 1:
    return 0.5 * np.ones((1, dim))
  lhs_lower_boundaries = (np.linspace(0, 1, num_samples+1)[:num_samples]).reshape(1, -1)
  width = lhs_lower_boundaries[0][1] - lhs_lower_boundaries[0][0]
  lhs_lower_boundaries = np.repeat(lhs_lower_boundaries, dim, axis=0).T
  lhs_indices = latin_hc_indices(dim, num_samples)
  lhs_sample_boundaries = []
  for i in range(num_samples):
    curr_idx = lhs_indices[i]
    curr_sample_boundaries = [lhs_lower_boundaries[curr_idx[j]][j] for j in range(dim)]
    lhs_sample_boundaries.append(curr_sample_boundaries)
  lhs_sample_boundaries = np.array(lhs_sample_boundaries)
  uni_random_width = width * np.random.random((num_samples, dim))
  lhs_samples = lhs_sample_boundaries + uni_random_width
  return lhs_samples

def is_an_opt_fidel_query(query_fidel, opt_fidel):
  """ Returns true if query_fidels are at opt_fidel. """
  return np.linalg.norm(query_fidel - opt_fidel) < 1e-5

def are_opt_fidel_queries(query_fidels, opt_fidel):
  """ Returns a boolean list which is True if at opt_fidel. """
  return np.array([is_an_opt_fidel_query(qf, opt_fidel) for qf in query_fidels])


# Functions for acqusitions. ------------------------------------------------------------
def _mf_gp_ucb_single(dom_pt, mfgp, opt_fidel, time_step):
  """ MF-GP-UCB acquisition function for evaluation at a single point.
      dom_pt: The point at which we want to evaluate the acquisition
      mfgp: An MFGP object.
      opt_fidel: The fidelity at which the optimisation needs to occur.
      time_step: The current time step of the acquisition.
  """
  ucb, beta_th = _mf_gp_ucb_multiple(dom_pt.reshape(1, -1), mfgp, opt_fidel, time_step)
  return float(ucb), beta_th

def _mf_gp_ucb_multiple(dom_pts, mfgp, opt_fidel, time_step):
  """ MF-GP-UCB acquisition function for evaluation at multiple points. """
  eff_l1_boundary = max(10,
    mfgp.domain_kernel.get_effective_norm(np.ones(mfgp.domain_dim), order=1))
  beta_t = 0.5 * mfgp.domain_dim * np.log(2 * time_step * eff_l1_boundary + 1)
  beta_th = np.clip(np.sqrt(beta_t), 3, 20)
  opt_fidel_m = np.repeat(opt_fidel.reshape(1, -1), len(dom_pts), axis=0)
  mu, sigma = mfgp.eval_at_fidel(opt_fidel_m, dom_pts, uncert_form='std')
  ucb = mu + beta_th * sigma
  return ucb, beta_th

def _mf_gp_ucb(acq_query_type, dom_pts, mfgp, opt_fidel, time_step):
  """ Wrapper for either _mf_gp_ucb_single or _mf_gp_ucb multiple. """
  if acq_query_type == 'single':
    return _mf_gp_ucb_single(dom_pts, mfgp, opt_fidel, time_step)
  elif acq_query_type == 'multiple':
    return _mf_gp_ucb_multiple(dom_pts, mfgp, opt_fidel, time_step)
  else:
    raise ValueError('acq_query_type should be \'single\' or \'multiple\'. Given, ' +
                     '\'%s\' unrecognised.'%(acq_query_type))

def _gp_ei_single(dom_pt, mfgp, opt_fidel, curr_best):
  """ GP-EI Acquisition evaluated at a single point. """
  gpei_val = _gp_ei_multiple(dom_pt.reshape(1, -1), mfgp, opt_fidel, curr_best)
  return float(gpei_val)

def _gp_ei_multiple(dom_pts, mfgp, opt_fidel, curr_best):
  """ GP-EI Acquisition evaluated at multiple points. """
  # pylint: disable=unused-argument
  opt_fidel_m = np.repeat(opt_fidel.reshape(1, -1), len(dom_pts), axis=0)
  mu, sigma = mfgp.eval_at_fidel(opt_fidel_m, dom_pts, uncert_form='std')
  Z = (mu - curr_best) / sigma
  ei = (mu - curr_best)*norm.cdf(Z) + sigma*norm.pdf(Z)
  return ei

def _gp_ei(acq_query_type, dom_pts, mfgp, opt_fidel, curr_best):
  """ Wrapper for either _mf_gp_ei_single or _mf_gp_ei multiple. """
  if acq_query_type == 'single':
    return _gp_ei_single(dom_pts, mfgp, opt_fidel, curr_best)
  elif acq_query_type == 'multiple':
    return _gp_ei_multiple(dom_pts, mfgp, opt_fidel, curr_best)
  else:
    raise ValueError('acq_query_type should be \'single\' or \'multiple\'. Given, ' +
                     '\'%s\' unrecognised.'%(acq_query_type))


acquisitions = Namespace(
  # MF-GP-UCB
  mf_gp_ucb=_mf_gp_ucb,
  mf_gp_ucb_single=_mf_gp_ucb_single,
  mf_gp_ucb_multiple=_mf_gp_ucb_multiple,
  # GP-UCB
  gp_ucb=_mf_gp_ucb,               # The acquisitions for gp-ucb are the same
  gp_ucb_single=_mf_gp_ucb_single,
  gp_ucb_multiple=_mf_gp_ucb_multiple,
  # GP-EI
  gp_ei=_gp_ei,
  gp_ei_single=_gp_ei_single,
  gp_ei_multiple=_gp_ei_multiple,
  )


# Functions for determining next fidel. --------------------------------------------------
def _mf_gp_ucb_fidel_chooser_single(next_pt, mfgp, mfof, acq_params):
  """ Function to determine the next fidelity for MF-GP-UCB.
      next_pt: The next point in the domain at which we will evaluate the function.
      mfgp: An MFGP object.
      mfof: An MFOptFunction object.
      time_step = current time step
  """
  # pylint: disable=too-many-locals
  cand_fidels = mfof.get_candidate_fidelities(filter_by_cost=True)
  num_cand_fidels = len(cand_fidels)
  cand_fidel_cost_ratios = mfof.get_cost_ratio(cand_fidels)
  opt_fidel_mat = np.repeat(mfof.opt_fidel.reshape(1, -1), num_cand_fidels, axis=0)
  cand_fidel_slacks = mfgp.fidel_kernel.compute_std_slack(opt_fidel_mat, cand_fidels)
  cand_fidel_diffs = np.linalg.norm(opt_fidel_mat - cand_fidels, axis=1)

  # Only select points with high standard deviation
  next_pt_mat = np.repeat(next_pt.reshape(1, -1), num_cand_fidels, axis=0)
  _, cand_fidel_stds = mfgp.eval_at_fidel(cand_fidels, next_pt_mat, uncert_form='std')
  cost_ratio_power = 1/float(mfgp.fidel_dim + mfgp.domain_dim + 2)
  std_thresholds = acq_params.thresh_coeff * ((cand_fidel_cost_ratios ** cost_ratio_power)
                                              * cand_fidel_slacks)
  high_std_idxs = cand_fidel_stds > std_thresholds

  # Only slect points that are far enough from opt_fidel
  eps_t = np.clip(1/acq_params.beta_th, 0.001, 0.2)
  diam_slack = mfgp.fidel_kernel.compute_std_slack(np.zeros((1, mfgp.fidel_dim)),
                                                   np.ones((1, mfgp.fidel_dim)))
  far_away_idxs = cand_fidel_slacks > eps_t * diam_slack
#   print(far_away_idxs.mean())
#   far_away_idxs = cand_fidel_diffs > eps_t * np.sqrt(mfgp.fidel_dim)

  # Now filter
  sel_idxs = high_std_idxs * far_away_idxs
  if sel_idxs.sum() == 0:
    return deepcopy(mfof.opt_fidel)
  else:
    sel_fidels = cand_fidels[sel_idxs]
    sel_cost_ratios = cand_fidel_cost_ratios[sel_idxs]
    min_cost_idx = sel_cost_ratios.argmin()
    next_fidel = sel_fidels[min_cost_idx]
    return next_fidel

def _opt_fidel_chooser_single(next_pt, mfgp, mfof, acq_params):
  """ Always returns the optimum fidelity.
  """
  # pylint: disable=unused-argument
  return deepcopy(mfof.opt_fidel)

fidelity_choosers = Namespace(
  # MF-GP-UCB
  mf_gp_ucb=_mf_gp_ucb_fidel_chooser_single,
  mf_gp_ucb_single=_mf_gp_ucb_fidel_chooser_single,
  # GP-UCB
  gp_ucb=_opt_fidel_chooser_single,
  gp_ucb_single=_opt_fidel_chooser_single,
  # GP-EI
  gp_ei=_opt_fidel_chooser_single,
  gp_ei_single=_opt_fidel_chooser_single,
  # MF-SKO
  mf_sko=mf_sko_fidel_chooser_single,
  mf_sko_single=mf_sko_fidel_chooser_single,
  )

