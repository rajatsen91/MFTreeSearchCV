"""
  A collection of utilities for MF-GP Bandits.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=super-on-old-class

import numpy as np
# Local imports
from mf.mf_func import MFOptFunction
from utils.general_utils import map_to_cube


# Hartmann Functions ---------------------------------------------------------------------
def hartmann(x, alpha, A, P, max_val=np.inf):
  """ Computes the hartmann function for any given A and P. """
  log_sum_terms = (A * (P - x)**2).sum(axis=1)
  return min(max_val, alpha.dot(np.exp(-log_sum_terms)))

def _get_hartmann_data(domain_dim):
  """ Returns A and P for the 3D hartmann function. """
  # pylint: disable=bad-whitespace
  if domain_dim == 3:
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]], dtype=np.float64)
    P = 1e-4 * np.array([[3689, 1170, 2673],
                         [4699, 4387, 7470],
                         [1091, 8732, 5547],
                         [ 381, 5743, 8828]], dtype=np.float64)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    domain = [[0, 1]] * 3
    opt_pt = np.array([0.114614, 0.555649, 0.852547])
    max_val = 3.86278

  elif domain_dim == 6:
    A = np.array([[  10,   3,   17, 3.5, 1.7,  8],
                  [0.05,  10,   17, 0.1,   8, 14],
                  [   3, 3.5,  1.7,  10,  17,  8],
                  [  17,   8, 0.05,  10, 0.1, 14]], dtype=np.float64)
    P = 1e-4 * np.array([[1312, 1696, 5569,  124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091,  381]], dtype=np.float64)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    domain = [[0, 1]] * 6
    opt_pt = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    max_val = 3.322368

  else:
    raise NotImplementedError('Only implemented in 3 and 6 dimensions.')
  return A, P, alpha, opt_pt, domain, max_val


def get_mf_hartmann_function(fidel_dim, domain_dim):
  """ Returns a function f(z, x). z refers to the fidelity and x is the point in the
      domain. """
  A, P, alpha, opt_pt, domain_bounds, max_val = _get_hartmann_data(domain_dim)
  # This is how much we will perturb the alphas
  delta = np.array([0.1] * fidel_dim + [0] * (4-fidel_dim))
  # Define a wrapper for the objective
  def mf_hart_obj(z, x):
    """ Wrapper for the hartmann objective. z is fidelity and x is domain. """
    assert len(z) == fidel_dim
    z_extended = np.append(z, [0] * (4-fidel_dim))
    alpha_z = alpha - (1 - z_extended) * delta
    return hartmann(x, alpha_z, A, P, max_val)
  # Define the optimum fidelity and the fidelity bounds
  opt_fidel = np.ones(fidel_dim)
  fidel_bounds = [[0, 1]] * fidel_dim
  return mf_hart_obj, opt_pt, opt_fidel, fidel_bounds, domain_bounds

def get_mf_hartmann_as_mfof(fidel_dim, domain_dim):
  """ Wrapper for get_mf_hartmann_function which returns the function as a
      mf.mf_func.MFOptFunction object. """
  mf_hart, opt_pt, opt_fidel, fidel_bounds, domain_bounds = get_mf_hartmann_function(
                                                              fidel_dim, domain_dim)
  fidel_cost_function = _get_mf_cost_function(fidel_bounds, True)
  opt_val = mf_hart(opt_fidel, opt_pt)
  return MFOptFunction(mf_hart, fidel_cost_function, fidel_bounds, domain_bounds,
                       opt_fidel, vectorised=False, opt_pt=opt_pt, opt_val=opt_val)
# Hartmann Functions end here ------------------------------------------------------------


# Currin Exponential Function ------------------------------------------------------------
def currin_exp(x, alpha):
  """ Computes the currin exponential function. """
  x1 = x[0]
  x2 = x[1]
  val_1 = 1 - alpha * np.exp(-1/(2 * x2))
  val_2 = (2300*x1**3 + 1900*x1**2 + 2092*x1 + 60) / (100*x1**3 + 500*x1**2 + 4*x1 + 20)
  return val_1 * val_2

def get_mf_currin_exp_function():
  """ Returns the multi-fidelity currin exponential function with d=6 and p=2. """
  opt_val = 13.7986850
  def mf_currin_exp_obj(z, x):
    """ Wrapper for the MF currin objective. """
    alpha_z = 1 - 0.1 * z
    return min(opt_val, currin_exp(x, alpha_z))
  opt_fidel = np.array([1])
  opt_pt = None
  fidel_bounds = np.array([[0, 1]])
  domain_bounds = np.array([[0, 1], [0, 1]])
  return mf_currin_exp_obj, opt_pt, opt_val, opt_fidel, fidel_bounds, domain_bounds

def get_mf_currin_exp_as_mfof():
  """ Wrapper for get_mf_currin_exp_function which returns the function as a
      mf.mf_func.MFOptFunction object. """
  mf_currin_exp_obj, opt_pt, opt_val, opt_fidel, fidel_bounds, domain_bounds = \
    get_mf_currin_exp_function()
  fidel_cost_function = lambda z: 0.1 + z**2
  return MFOptFunction(mf_currin_exp_obj, fidel_cost_function, fidel_bounds,
                       domain_bounds, opt_fidel, vectorised=False, opt_pt=opt_pt,
                       opt_val=opt_val)
# Currin Exponential Function ends here --------------------------------------------------

# Branin Function ------------------------------------------------------------------------
def branin_function(x, a, b, c, r, s, t):
  """ Computes the Branin function. """
  x1 = x[0]
  x2 = x[1]
  neg_ret = a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s
  return -neg_ret

def branin_function_alpha(x, alpha, a, r, s):
  """ Alternative form for the branin function. """
  return branin_function(x, a, alpha[0], alpha[1], r, s, alpha[2])

def get_mf_branin_function(fidel_dim):
  """ Returns the Branin function as a multifidelity function. """
  a0 = 1
  b0 = 5.1/(4*np.pi**2)
  c0 = 5/np.pi
  r0 = 6
  s0 = 10
  t0 = 1/(8*np.pi)
  alpha = np.array([b0, c0, t0])
  # Define delta
  delta = [0.01, 0.1, -0.005]
  delta = np.array(delta[0:fidel_dim] + [0] * (3 - fidel_dim))

  def mf_branin_obj(z, x):
    """ Wrapper for the MF Branin objective. """
    assert len(z) == fidel_dim
    z_extended = np.append(z, [0] * (3-fidel_dim))
    alpha_z = alpha - (1 - z_extended) * delta
    return branin_function_alpha(x, alpha_z, a0, r0, s0)
  # Other data
  opt_fidel = np.ones((fidel_dim))
  fidel_bounds = [[0, 1]] * fidel_dim
  opt_pt = np.array([np.pi, 2.275])
  domain_bounds = [[-5, 10], [0, 15]]
  return mf_branin_obj, opt_pt, opt_fidel, fidel_bounds, domain_bounds

def get_mf_branin_as_mfof(fidel_dim):
  """ Wrapper for get_mf_branin_function which returns as a mfof. """
  mf_branin_obj, opt_pt, opt_fidel, fidel_bounds, domain_bounds = \
    get_mf_branin_function(fidel_dim)
  fidel_cost_function = _get_mf_cost_function(fidel_bounds, True)
  opt_val = mf_branin_obj(opt_fidel, opt_pt)
  return MFOptFunction(mf_branin_obj, fidel_cost_function, fidel_bounds, domain_bounds,
                       opt_fidel, vectorised=False, opt_pt=opt_pt, opt_val=opt_val)
# Branin Function ends here --------------------------------------------------------------


# Borehole Function ----------------------------------------------------------------------
def borehole_function(x, z, max_val):
  """ Computes the Bore Hole function. """
  # pylint: disable=bad-whitespace
  rw = x[0]
  r  = x[1]
  Tu = x[2]
  Hu = x[3]
  Tl = x[4]
  Hl = x[5]
  L  = x[6]
  Kw = x[7]
  # Compute high fidelity function
  frac2 = 2*L*Tu/(np.log(r/rw) * rw**2 * Kw)
  f2 = min(max_val, 2 * np.pi * Tu * (Hu - Hl)/(np.log(r/rw) * (1 + frac2 + Tu/Tl)))
  # Compute low fidelity function
  f1 = 5 * Tu * (Hu - Hl)/(np.log(r/rw) * (1.5 + frac2 + Tu/Tl))
  # Compute final output
  return f2*z + f1*(1-z)

def get_mf_borehole_function():
  """ Gets the MF BoreHole function. """
  opt_val = 309.523221
  opt_pt = None
  mf_borehole_function = lambda z, x: borehole_function(x, z, opt_val)
  domain_bounds = [[0.05, 0.15],
                   [100, 50000],
                   [63070, 115600],
                   [990, 1110],
                   [63.1, 116],
                   [700, 820],
                   [1120, 1680],
                   [9855, 12045]]
  fidel_bounds = [[0, 1]]
  opt_fidel = np.array([1])
  return mf_borehole_function, opt_pt, opt_val, opt_fidel, fidel_bounds, domain_bounds

def get_mf_borehole_as_mfof():
  """ Gets the MF BoreHold as an mfof. """
  mf_borehole_function, opt_pt, opt_val, opt_fidel, fidel_bounds, domain_bounds = \
    get_mf_borehole_function()
  fidel_cost_function = lambda z: 0.1 + z**1.5
  return MFOptFunction(mf_borehole_function, fidel_cost_function, fidel_bounds,
                       domain_bounds, opt_fidel, vectorised=False, opt_pt=opt_pt,
                       opt_val=opt_val)
# Borehole Function ends here ------------------------------------------------------------


def _get_mf_cost_function(fidel_bounds, is_0_1):
  """ Returns the cost function. fidel_bounds are the bounds for the fidelity space
      and is_0_1 should be true if fidel_bounds is [0,1]^p. """
  fidel_dim = len(fidel_bounds)
  if fidel_dim == 1:
    fidel_powers = [2]
  elif fidel_dim == 2:
    fidel_powers = [3, 2]
  elif fidel_dim == 3:
    fidel_powers = [3, 2, 1.5]
  else:
    fidel_powers = [3] + list(np.linspace(2, 1.2, fidel_dim-1))
  # Define the normalised
  def _norm_cost_function(norm_z):
    """ The cost function with normalised coordinates. """
    min_cost = 0.05
    return min_cost + (1-min_cost) * np.power(norm_z, fidel_powers).sum()
  # Now return based on whether or not is_0_1
  ret = (_norm_cost_function if is_0_1 else
           lambda z: _norm_cost_function(map_to_cube(z, fidel_bounds)))
  return ret

