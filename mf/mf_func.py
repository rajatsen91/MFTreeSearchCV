"""
  A collection of functions for managing multi-fidelity functions.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=super-on-old-class

import numpy as np
# Local imports
from utils.general_utils import map_to_cube, map_to_bounds


class MFFunction(object):
  """ This just creates a wrapper to call the function by appropriately creating bounds
      and querying appropriately. """

  def __init__(self, mf_func, fidel_cost_func, fidel_bounds, domain_bounds,
               vectorised=True):
    """ Constructor.
          mf_func: takes two arguments mf_func(z, x) where z is the fidelity and x is
            the point in the domain.
          fidel_cost_func: fidel_cost_func(z) gives the cost of evaluating at z.
          fidel_bounds, domain_bounds: are the bounds of the fidelity spaces, domains
            resp.
          vectorised: If True it means mf_func and fidel_cost_func can take
            multiple inputs and produce multiple outputs. If False, the functions
            can take only single inputs in 'column' form.
    """
    self.mf_func = mf_func
    self.fidel_cost_func = fidel_cost_func
    self.fidel_bounds = np.array(fidel_bounds)
    self.domain_bounds = np.array(domain_bounds)
    self.fidel_dim = len(fidel_bounds)
    self.domain_dim = len(domain_bounds)
    self.vectorised = vectorised

  # Wrappers for evaluating the function -------------------------------------------------
  def eval_at_fidel_single_point(self, Z, X):
    """ Evaluates X at the given Z at a single point. """
    if not self.vectorised:
      return float(self.mf_func(Z, X))
    else:
      Z = np.array(Z).reshape((1, self.fidel_dim))
      X = np.array(X).reshape((1, self.domain_dim))
      return float(self.mf_func(Z, X))

  def eval_at_fidel_multiple_points(self, Z, X):
    """ Evaluates X at the given Z at multiple points. """
    if self.vectorised:
      return self.mf_func(Z, X).ravel()
    else:
      ret = []
      for i in range(len(Z)):
        ret.append(self.eval_at_fidel_single_point(Z[i, :], X[i, :]))
      return np.array(ret)

  # Wrappers for evaluating the cost function --------------------------------------------
  def eval_fidel_cost_single_point(self, Z):
    """ Evaluates the cost function at a single point. """
    if not self.vectorised:
      return float(self.fidel_cost_func(Z))
    else:
      Z = np.array(Z).reshape((1, self.fidel_dim))
      return float(self.fidel_cost_func(Z))

  def eval_fidel_cost_multiple_points(self, Z):
    """ Evaluates the cost function at multiple points. """
    if self.vectorised:
      return self.fidel_cost_func(Z).ravel()
    else:
      ret = []
      for i in range(len(Z)):
        ret.append(self.eval_fidel_cost_single_point(Z[i, :]))
      return np.array(ret)

  # Wrappers for evaluating at normalised points -----------------------------------------
  def eval_at_fidel_single_point_normalised(self, Z, X):
    """ Evaluates X at the given Z at a single point using normalised coordinates. """
    Z, X = self.get_unnormalised_coords(Z, X)
    return self.eval_at_fidel_single_point(Z, X)

  def eval_at_fidel_multiple_points_normalised(self, Z, X):
    """ Evaluates X at the given Z at multiple points using normalised coordinates. """
    Z, X = self.get_unnormalised_coords(Z, X)
    return self.eval_at_fidel_multiple_points(Z, X)

  def eval_fidel_cost_single_point_normalised(self, Z):
    """ Evaluates the cost function at a single point using normalised coordinates. """
    Z, _ = self.get_unnormalised_coords(Z, None)
    return self.eval_fidel_cost_single_point(Z)

  def eval_fidel_cost_multiple_points_normalised(self, Z):
    """ Evaluates the cost function at multiple points using normalised coordinates. """
    Z, _ = self.get_unnormalised_coords(Z, None)
    return self.eval_fidel_cost_multiple_points(Z)

  # Maps to normalised coordinates and vice versa ----------------------------------------
  def get_normalised_coords(self, Z, X):
    """ Maps points in the original space to the cube. """
    ret_Z = None if Z is None else map_to_cube(Z, self.fidel_bounds)
    ret_X = None if X is None else map_to_cube(X, self.domain_bounds)
    return ret_Z, ret_X

  def get_unnormalised_coords(self, Z, X):
    """ Maps points in the cube to the original space. """
    ret_Z = None if Z is None else map_to_bounds(Z, self.fidel_bounds)
    ret_X = None if X is None else map_to_bounds(X, self.domain_bounds)
    return ret_Z, ret_X
# MFFunction ends here ===================================================================


class MFOptFunction(MFFunction):
  """ A class which we will use for MF Optimisation. """

  def __init__(self, mf_func, fidel_cost_func, fidel_bounds, domain_bounds,
               opt_fidel_unnormalised, vectorised=True, opt_pt=None, opt_val=None):
    """ Constructor.
          mf_func: takes two arguments mf_func(z, x) where z is the fidelity and x is
            the point in the domain.
          fidel_cost_func: fidel_cost_func(z) gives the cost of evaluating at z.
          fidel_bounds, domain_bounds: are the bounds of the fidelity spaces, domains
            resp.
          opt_fidel: The point in the fidelity space at which we want to optimise.
          vectorised: If True it means mf_func and fidel_cost_func can take
            multiple inputs and produce multiple outputs. If False, the functions
            can take only single inputs in 'column' form.
          opt_pt, opt_val: The optimum point and value in the domain.
    """
    super(MFOptFunction, self).__init__(mf_func, fidel_cost_func, fidel_bounds,
                                        domain_bounds, vectorised)
    self.opt_fidel_unnormalised = np.array(opt_fidel_unnormalised).ravel()
    self.opt_fidel, _ = self.get_normalised_coords(opt_fidel_unnormalised, None)
    if len(self.opt_fidel) != self.fidel_dim:
      raise ValueError('opt_fidel should be a %d-vector.'%(self.fidel_dim))
    self.opt_fidel_cost = self.cost_single(self.opt_fidel)
    # Set the optimisation point.
    self.opt_pt = opt_pt
    self.opt_val = opt_val
    self.mfgp = None # we will need this later on.
    self.finite_fidels = None
    self.is_finite = False

  # Evaluation ---------------------------------------------------------------------------
  def eval_single(self, Z, X):
    """ Evaluate at a single point. """
    return self.eval_at_fidel_single_point_normalised(Z, X)

  def eval_multiple(self, Z, X):
    """ Evaluate at multiple points. """
    return self.eval_at_fidel_multiple_points_normalised(Z, X)

  def eval(self, Z, X):
    """ Executes either eval_single or eval_multiple. """
    if len(Z.shape) == 1:
      return self.eval_single(Z, X)
    elif len(Z.shape) == 2:
      return self.eval_multiple(Z, X)
    else:
      raise ValueError('Z should be either a vector or matrix.')

  # Cost ---------------------------------------------------------------------------------
  def cost_single(self, Z):
    """ Evaluates cost at a single point. """
    return self.eval_fidel_cost_single_point_normalised(Z)

  def cost_multiple(self, Z):
    """ Evaluates cost at multiple points. """
    return self.eval_fidel_cost_multiple_points_normalised(Z)

  def cost(self, Z):
    """ Executes either cost_single or cost_multiple. """
    if len(Z.shape) == 1:
      return self.cost_single(Z)
    elif len(Z.shape) == 2:
      return self.cost_multiple(Z)
    else:
      raise ValueError('Z should be either a vector or matrix.')

  # Other --------------------------------------------------------------------------------
  def get_cost_ratio(self, Z1, Z2=None):
    """ Obtains the ration between the costs. """
    if Z2 is None:
      cost_Z2 = self.opt_fidel_cost
    else:
      cost_Z2 = self.cost(Z2)
    return self.cost(Z1)/cost_Z2

  def get_candidate_fidelities(self, filter_by_cost=True):
    """ Gets candidate fidelities. If filter_by_cost is True then it doesn't return those
        whose cost is larger than opt_cost_fidel. """
    # Determine the candidates randomly
    if self.is_finite:
      return self.get_candidate_fidelities_finite()
    if self.fidel_dim == 1:
      candidates = np.linspace(0, 1, 200).reshape((-1, 1))
    elif self.fidel_dim == 2:
      num_per_dim = 25
      candidates = (np.indices((num_per_dim, num_per_dim)).reshape(2, -1).T + 0.5) / \
                     float(num_per_dim)
    elif self.fidel_dim == 3:
      num_per_dim = 10
      cand_1 = (np.indices((num_per_dim, num_per_dim, num_per_dim)).reshape(3, -1).T
                + 0.5) / float(num_per_dim)
      cand_2 = np.random.random((1000, self.fidel_dim))
      candidates = np.vstack((cand_1, cand_2))
    else:
      candidates = np.random.random((4000, self.fidel_dim))
    # To filter by cost?
    if filter_by_cost:
      fidel_costs = self.cost_multiple(candidates)
      filtered_idxs = fidel_costs < self.opt_fidel_cost
      candidates = candidates[filtered_idxs, :]
    # Finally add the highest fidelity.
    candidates = np.vstack((self.opt_fidel.reshape((1, self.fidel_dim)), candidates))
    return candidates

  def set_finite_fidels(self, finite_fidels_raw, is_normalised):
    """ Sets the finite fidels. """
    self.is_finite = True
    if is_normalised:
      self.finite_fidels = finite_fidels_raw
    else:
      self.finite_fidels_unnormalised = finite_fidels_raw
      self.finite_fidels, _ = self.get_normalised_coords(finite_fidels_raw, None)

  def get_candidate_fidelities_finite(self):
    """ Gets the finite candidate fidelities. """
    candidates = np.repeat(self.finite_fidels, 100, axis=0)
    np.random.shuffle(candidates)
    candidates = candidates[1:500, :]
    candidates = np.vstack((self.opt_fidel.reshape((1, self.fidel_dim)), candidates))
    return candidates

# MFOptFunction ends here ================================================================


class NoisyMFOptFunction(MFOptFunction):
  """ Child class of MFOptFunction which also adds noise to the evaluations. """

  def __init__(self, mf_func, fidel_cost_func, fidel_bounds, domain_bounds,
               opt_fidel_unnormalised, noise_var, noise_type='gauss',
               *args, **kwargs):
    """ Constructor. See MFOptFunction and MFFunction for args. """
    super(NoisyMFOptFunction, self).__init__(mf_func, fidel_cost_func, fidel_bounds,
      domain_bounds, opt_fidel_unnormalised, *args, **kwargs)
    self.noise_var = noise_var
    self.noise_type = noise_type

  # Noise functions ----------------------------------------------------------------------
  def noise_multiple(self, num_samples):
    """ Returns noise. """
    if self.noise_type == 'gauss':
      return np.random.normal(scale=np.sqrt(self.noise_var), size=(num_samples))
    else:
      raise NotImplementedError('Only implemented gauss noise so far. ')

  def noise_single(self):
    """ Single noise value. """
    return float(self.noise_multiple(1))

  # Override evaluation functions to add noise. ------------------------------------------
  def eval_single_noiseless(self, Z, X):
    """ Evaluate at a single point. """
    return super(NoisyMFOptFunction, self).eval_single(Z, X)

  def eval_multiple_noiseless(self, Z, X):
    """ Evaluate at multiple points. """
    return super(NoisyMFOptFunction, self).eval_multiple(Z, X)

  def eval_single(self, Z, X):
    """ Evaluate at a single point. """
    return self.eval_single_noiseless(Z, X) + self.noise_single()

  def eval_multiple(self, Z, X):
    """ Evaluate at multiple points. """
    return self.eval_multiple_noiseless(Z, X) + self.noise_multiple(len(Z))


def get_noisy_mfof_from_mfof(mfof, noise_var, noise_type='gauss', additional_attrs=None):
  """ Returns a noisy mfof object from an mfof object. """
  nmfof = NoisyMFOptFunction(mfof.mf_func, mfof.fidel_cost_func, mfof.fidel_bounds,
                             mfof.domain_bounds, mfof.opt_fidel_unnormalised, noise_var,
                             noise_type=noise_type,
                             vectorised=mfof.vectorised,
                             opt_pt=mfof.opt_pt,
                             opt_val=mfof.opt_val,
                            )
  if additional_attrs is None:
    additional_attrs = ['init_mfgp', 'mfgp']
  for attr in additional_attrs:
    if hasattr(mfof, attr):
      setattr(nmfof, attr, getattr(mfof, attr))
  return nmfof
# NOisyMFOptFunction ends here ===========================================================
