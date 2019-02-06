"""
  Used to generate a sample from an MFGP sample.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=too-many-locals
# pylint: disable=no-name-in-module
# pylint: disable=superfluous-parens

import numpy as np
from scipy.interpolate import RectBivariateSpline
# Local imports
from gp.kernel import SEKernel
import mf_func
import mf_gp
from utils.ancillary_utils import plot_2d_function


num_per_dim = 50
spline_degree = 3


def gen_mfgp_sample_as_mfof(mfgp, fidel_cost_func, random_seed):
  """ Generates an mfgp sample as a mfof. """
  if mfgp.fidel_dim != 1 or mfgp.domain_dim != 1:
    raise NotImplementedError('Only implemented 1 dimensional fidel/domain so far!')
  # Get/set the random state.
  st0 = np.random.get_state()
  np.random.seed(random_seed)

  # Set some attributes up
  fidel_bounds = [[0, 1]] * mfgp.fidel_dim
  domain_bounds = [[0, 1]] * mfgp.domain_dim
  opt_fidel = np.array([1])

  # This part of the code relies on dim_z = dim_x = 1
  # Create a grid for interpolation
  dim_grid = np.linspace(0, 1, num_per_dim)
  ZZ, XX = np.meshgrid(dim_grid, dim_grid)
  grid_pts = np.concatenate((ZZ.reshape(-1, 1), XX.reshape(-1, 1)), axis=1)
  grid_samples = mfgp.draw_samples(1, grid_pts).ravel()
  grid_samples_as_grid = grid_samples.reshape((num_per_dim, num_per_dim))

  rbs = RectBivariateSpline(dim_grid, dim_grid, grid_samples_as_grid,
                            kx=spline_degree, ky=spline_degree)
  g = lambda z, x: rbs.ev(x, z)

  # compute optimum point
  opt_search_grid_size = 1000
  opt_search_dom_grid = np.linspace(0, 1, opt_search_grid_size).reshape(-1, 1)
  opt_search_fidel_m = np.repeat(opt_fidel.reshape(-1, 1), opt_search_grid_size, axis=0)
  opt_fidel_grid_vals = g(opt_search_fidel_m, opt_search_dom_grid)
  opt_idx = opt_fidel_grid_vals.argmax()
  opt_pt = np.array(opt_search_dom_grid[opt_idx])
  opt_val = opt_fidel_grid_vals[opt_idx]

  mfof = mf_func.MFOptFunction(g, fidel_cost_func, fidel_bounds, domain_bounds,
                              opt_fidel, vectorised=True, opt_pt=opt_pt, opt_val=opt_val)
  mfof.mfgp = mfgp

  # before returning restate the np random state
  np.random.set_state(st0)
  return mfof


def gen_mfgp_sample_as_noisy_mfof(mfgp, fidel_cost_func, random_seed, noise_var):
  """ Generates an mfgp sample as a noisy mfof. """
  mfof = gen_mfgp_sample_as_mfof(mfgp, fidel_cost_func, random_seed)
  return mf_func.get_noisy_mfof_from_mfof(mfof, noise_var)


def gen_simple_mfgp_as_mfof(fidel_bw=0.8, random_seed=512):
  """ Gets a simple mfgp wrapped into an mfof. """
  # Create a GP
  kernel_scale = 2
  fidel_kernel = SEKernel(1, 1, [fidel_bw])
  domain_kernel = SEKernel(1, 1, [0.08])
  noise_var = 0.1
  dummy_ZZ = np.zeros((0, 1))
  dummy_XX = np.zeros((0, 1))
  dummy_YY = np.zeros((0))
  mean_func = lambda x: np.zeros((len(x)))
  mfgp = mf_gp.get_mfgp_from_fidel_domain(dummy_ZZ, dummy_XX, dummy_YY, kernel_scale,
           fidel_kernel, domain_kernel, mean_func, noise_var, build_posterior=True)
  # Get an mfof object
  fidel_cost_func = lambda z: 0.2 + 6 * z ** 2
  return gen_mfgp_sample_as_mfof(mfgp, fidel_cost_func, random_seed)


def visualise_mfof(mfof):
  """ Visualises the mfof object. """
  plot_func = mfof.eval_multiple
  _, ax, plt = plot_2d_function(plot_func,
                               np.array([mfof.fidel_bounds[0], mfof.domain_bounds[0]]),
                               x_label='fidel', y_label='domain')
  ax.scatter(mfof.opt_fidel, mfof.opt_pt, mfof.opt_val, c='r', s=100)
  plt.show()


def main():
  """ Main function. """
  print(np.random.random())
  mfof = gen_simple_mfgp_as_mfof()
  visualise_mfof(mfof)
  print(np.random.random())


if __name__ == '__main__':
  main()

