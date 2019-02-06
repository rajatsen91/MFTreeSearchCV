"""
  Implements the kernel, GP and fitter for multi-fidelity GPs.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=super-on-old-class

import numpy as np

# Local imports
from gp.kernel import CoordinateProductKernel, PolyKernel, SEKernel
from gp.gp_core import GP, GPFitter, mandatory_gp_args
from utils.option_handler import get_option_specs, load_options
from utils.reporters import get_reporter
from utils.ancillary_utils import get_list_of_floats_as_str


# Define hyper-parameters for Multi-fidelity GPs.
mf_gp_args = [
  # Fidelity kernel
  get_option_specs('fidel_kernel_type', False, 'se',
    'Type of kernel for the fidelity space. Should be se or poly'),
  get_option_specs('fidel_use_same_bandwidth', False, False,
    ('If true, will use same bandwidth on all fidelity dimensions. Applies only when '
     'fidel_kernel_type is se. Default=False.')),
  get_option_specs('fidel_use_same_scalings', False, False,
    ('If true, will use same scaling on all fidelity dimensions. Applies only when '
     'fidel_kernel_type is poly. Default=False.')),
  get_option_specs('fidel_poly_order', False, 1,
    ('Order of the polynomial for the fidelity kernel. Default = 1 (linear kernel)')),
  # Domain kernel
  get_option_specs('domain_kernel_type', False, 'se',
    'Type of kernel for the domain. Should be se or poly'),
  get_option_specs('domain_use_same_bandwidth', False, False,
    ('If true, will use same bandwidth on all domain dimensions. Applies only when '
     'domain_kernel_type is se. Default=False.')),
  get_option_specs('domain_use_same_scalings', False, False,
    ('If true, will use same scaling on all domain dimensions. Applies only when '
     'domain_kernel_type is poly. Default=False.')),
  get_option_specs('domain_poly_order', False, 1,
    ('Order of the polynomial for the domain kernel. Default = 1 (linear kernel)')),
  # Mean function
  get_option_specs('mean_func_type', False, 'median',
    ('Specify the type of mean function. Should be upper_bound, mean, median, const ',
     'or zero. If const, specifcy value in mean-func-const.')),
  get_option_specs('mean_func_const', False, 0.0,
    'The constant value to use if mean_func_type is const.'),
  # Kernel scale
  get_option_specs('kernel_scale_type', False, 'tune',
    ('Specify how to obtain the kernel scale. Should be tune, label or value. Specify '
     'appropriate value in kernel_scale_label or kernel_scale_value')),
  get_option_specs('kernel_scale_label', False, 2,
    'The fraction of label variance to use as noise variance.'),
  get_option_specs('kernel_scale_value', False, 1,
    'The (absolute) value to use as noise variance.'),
  # Noise variance
  get_option_specs('noise_var_type', False, 'tune',
    ('Specify how to obtain the noise variance. Should be tune, label or value. Specify '
     'appropriate value in noise_var_label or noise_var_value')),
  get_option_specs('noise_var_label', False, 0.05,
    'The fraction of label variance to use as noise variance.'),
  get_option_specs('noise_var_value', False, 0.1,
    'The (absolute) value to use as noise variance.'),
  ]
# Define this which includes the mandatory GP args
all_mf_gp_args = mandatory_gp_args + mf_gp_args


class MFGP(GP):
  """ A GP to be used for multi-fidelity optimisation. """

  def __init__(self, ZX, YY, fidel_coords, domain_coords,
               kernel_scale, fidel_kernel, domain_kernel,
               mean_func, noise_var, *args, **kwargs):
    """ Constructor. ZZ, XX, YY are the fidelity points, domain points and labels
        respectively.
    """
    self.fidel_coords = fidel_coords
    self.domain_coords = domain_coords
    self.fidel_dim = len(fidel_coords)
    self.domain_dim = len(domain_coords)
    self.fidel_kernel = fidel_kernel
    self.domain_kernel = domain_kernel
    # Construct coordinate product kernel
    mf_kernel = CoordinateProductKernel(self.fidel_dim + self.domain_dim, kernel_scale,
                                        [self.fidel_kernel, self.domain_kernel],
                                        [self.fidel_coords, self.domain_coords],
                                       )
    # Call super constructor
    super(MFGP, self).__init__(ZX, YY, mf_kernel, mean_func, noise_var, *args, **kwargs)

  def get_domain_pts(self, data_idxs=None):
    """ Returns only the domain points. """
    data_idxs = data_idxs if not data_idxs is None else range(self.num_tr_data)
    return self.ZX[data_idxs, self.domain_coords]

  def get_fidel_pts(self, data_idxs=None):
    """ Returns only the fidelity points. """
    data_idxs = data_idxs if not data_idxs is None else range(self.num_tr_data)
    return self.ZX[data_idxs, self.fidel_coords]

  def _get_ZX_from_ZZ_XX(self, ZZ, XX):
    """ Gets the coordinates in the joint space from the individual fidelity and
        domain spaces. """
    if ZZ.shape[1] != self.fidel_dim or XX.shape[1] != self.domain_dim:
      raise ValueError('ZZ, XX dimensions should be (%d, %d). Given (%d, %d)'%(
                       self.fidel_dim, self.domain_dim, ZZ.shape[1], XX.shape[1]))
    ZX_unordered = np.concatenate((ZZ, XX), axis=1)
    ordering = np.argsort(self.fidel_coords + self.domain_coords)
    return ZX_unordered[:, ordering]

  def eval_at_fidel(self, ZZ_test, XX_test, *args, **kwargs):
    """ Evaluates the GP at [ZZ, XX]. Read eval in gp_core.GP for more details. """
    ZX_test = self._get_ZX_from_ZZ_XX(ZZ_test, XX_test)
    return self.eval(ZX_test, *args, **kwargs)

  def add_mf_data(self, ZZ_new, XX_new, YY_new, *args, **kwargs):
    """ Adds new data to the multi-fidelity GP. """
    ZX_new = self._get_ZX_from_ZZ_XX(ZZ_new, XX_new)
    self.add_data(ZX_new, YY_new, *args, **kwargs)

  def draw_mf_samples(self, num_samples, ZZ_test=None, XX_test=None, *args, **kwargs):
    """ Draws samples from a multi-fidelity GP. """
    ZX_test = None if ZZ_test is None else self._get_ZX_from_ZZ_XX(ZZ_test, XX_test)
    return self.draw_samples(num_samples, ZX_test, *args, **kwargs)

  def __str__(self):
    """ Returns a string representation of the MF-GP. """
    fidel_ke_str = self._get_kernel_str(self.fidel_kernel)
    domain_ke_str = self._get_kernel_str(self.domain_kernel)
    ret = 'noise: %0.4f, scale: %0.3f, fid: %s, dom: %s'%(self.noise_var,
      self.kernel.scale, fidel_ke_str, domain_ke_str)
    return ret

  @classmethod
  def _get_kernel_str(cls, kern):
    """ Gets a string format of the kernel depending on whether it is SE/Poly."""
    if isinstance(kern, SEKernel):
      hp_name = 'dim_bandwidths'
      kern_name = 'se'
    elif isinstance(kern, PolyKernel):
      hp_name = 'dim_scalings'
      kern_name = 'poly'
    if kern.dim > 4:
      ret = '%0.4f(avg)'%(kern.hyperparams[hp_name].mean())
    else:
      ret = get_list_of_floats_as_str(kern.hyperparams[hp_name])
    ret = kern_name + '-' + ret
    return ret


def get_mfgp_from_fidel_domain(ZZ, XX, YY, kernel_scale, fidel_kernel,
    domain_kernel, mean_func, noise_var, *args, **kwargs):
  # pylint: disable=too-many-locals
  """ A wrapper which concatenates the ZZ and XX and returns an MFGP object. """
  fidel_dim = ZZ.shape[1]
  domain_dim = XX.shape[1]
  fidel_coords = range(fidel_dim)
  domain_coords = range(fidel_dim, fidel_dim + domain_dim)
  ZX = np.concatenate((ZZ, XX), axis=1)
  return MFGP(ZX, YY, fidel_coords, domain_coords,
              kernel_scale, fidel_kernel, domain_kernel,
              mean_func, noise_var, *args, **kwargs)


class MFGPFitter(GPFitter):
  """ A fitter for GPs in multi-fidelity optimisation. """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, ZZ, XX, YY, options=None, reporter=None):
    """ Constructor. options should either be a Namespace, a list or None"""
    reporter = get_reporter(reporter)
    if options is None:
      options = load_options(all_mf_gp_args, 'MF-GP', reporter)
    self.ZZ = ZZ
    self.XX = XX
    self.YY = YY
    self.ZX = np.concatenate((self.ZZ, self.XX), axis=1)
    self.fidel_dim = self.ZZ.shape[1]
    self.domain_dim = self.XX.shape[1]
    self.input_dim = self.fidel_dim + self.domain_dim
    super(MFGPFitter, self).__init__(options, reporter)

  # Child Set up Methods
  # ======================================================================================
  def _child_set_up(self):
    """ Sets parameters for GPFitter. """
    # Check args - so that we don't have to keep doing this all the time
    if not self.options.fidel_kernel_type in ['se', 'poly']:
      raise ValueError('Unknown fidel_kernel_type. Should be either se or poly.')
    if not self.options.domain_kernel_type in ['se', 'poly']:
      raise ValueError('Unknown domain_kernel_type. Should be either se or poly.')
    if not self.options.noise_var_type in ['tune', 'label', 'value']:
      raise ValueError('Unknown noise_var_type. Should be either tune, label or value.')
    if not self.options.mean_func_type in ['mean', 'median', 'const', 'zero',
                                           'upper_bound']:
      raise ValueError(('Unknown mean_func_type. Should be one of ',
                        'mean/median/const/zero.'))
    # Set some parameters we will be using often.
    self.Y_var = self.YY.std()**2
    self.ZX_std_norm = np.linalg.norm(self.ZX.std(axis=0))

    # Bounds for the hyper parameters
    # -------------------------------
    self.hp_bounds = []
    # Noise variance
    if self.options.noise_var_type == 'tune':
      self.noise_var_log_bounds = [np.log(0.005 * self.Y_var), np.log(0.2 * self.Y_var)]
      self.hp_bounds.append(self.noise_var_log_bounds)
    # Kernel scale
    self.scale_log_bounds = [np.log(0.1 * self.Y_var), np.log(10 * self.Y_var)]
    self.hp_bounds.append(self.scale_log_bounds)
    # Fidelity kernel
    if self.options.fidel_kernel_type == 'se':
      self._fidel_se_kernel_setup()
    elif self.options.fidel_kernel_type == 'poly':
      self._fidel_poly_kernel_setup()
    # Domain kernel
    if self.options.domain_kernel_type == 'se':
      self._domain_se_kernel_setup()
    elif self.options.domain_kernel_type == 'poly':
      self._domain_pol_kernel_setup()

  def _fidel_se_kernel_setup(self):
    """ Sets up the fidelity kernel as a SE kernel. """
    if (hasattr(self.options, 'fidel_bandwidth_log_bounds') and
        self.options.fidel_bandwidth_log_bounds is not None):
      self.fidel_bandwidth_log_bounds = self.options.fidel_bandwidth_log_bounds
    else:
      self.fidel_bandwidth_log_bounds = self._get_se_kernel_bounds(
        self.fidel_dim, self.ZX_std_norm, self.options.fidel_use_same_bandwidth)
    self.hp_bounds.extend(self.fidel_bandwidth_log_bounds)

  def _fidel_poly_kernel_setup(self):
    """ Sets up the fidelity kernel as a Poly kernel. """
    self.fidel_scaling_log_bounds = self._get_poly_kernel_bounds(self.ZX,
                                        self.options.fidel_use_same_scalings)
    self.hp_bounds.extend(self.fidel_scaling_log_bounds)

  def _domain_se_kernel_setup(self):
    """ Sets up the domainity kernel as a SE kernel. """
    if (hasattr(self.options, 'domain_bandwidth_log_bounds') and
        self.options.domain_bandwidth_log_bounds is not None):
      self.domain_bandwidth_log_bounds = self.options.domain_bandwidth_log_bounds
    else:
      self.domain_bandwidth_log_bounds = self._get_se_kernel_bounds(
         self.domain_dim, self.ZX_std_norm, self.options.domain_use_same_bandwidth)
    self.hp_bounds.extend(self.domain_bandwidth_log_bounds)

  def _domain_poly_kernel_setup(self):
    """ Sets up the domainity kernel as a Poly kernel. """
    self.domain_scaling_log_bounds = self._get_poly_kernel_bounds(self.ZX,
                                        self.options.domain_use_same_scalings)
    self.hp_bounds.extend(self.domain_scaling_log_bounds)

  @classmethod
  def _get_se_kernel_bounds(cls, dim, single_bw_bounds, use_same_bandwidth):
    """ Gets bandwidths for the SE kernel. """
    if isinstance(single_bw_bounds, float) or isinstance(single_bw_bounds, int):
      single_bw_bounds = [0.01*single_bw_bounds, 10*single_bw_bounds]
    single_bandwidth_log_bounds = [np.log(x) for x in single_bw_bounds]
    bandwidth_log_bounds = ([single_bandwidth_log_bounds] if use_same_bandwidth
                            else [single_bandwidth_log_bounds] * dim)
    return bandwidth_log_bounds

  def _get_poly_kernel_bounds(self, data, use_same_scalings):
    """ Gets bandwidths for the Polynomial kerne. """
    # TODO: implement poly kernel
    raise NotImplementedError('Yet to implement polynomial kernel.')
  # _child_set_up methods end here -------------------------------------------------------

  # build_gp Methods
  # ======================================================================================
  def _child_build_gp(self, gp_hyperparams):
    """ Builds a Multi-fidelity GP from the hyper-parameters. """
    # pylint: disable=too-many-branches
    # Noise variance ------------------------------------
    if self.options.noise_var_type == 'tune':
      noise_var = np.exp(gp_hyperparams[0])
      gp_hyperparams = gp_hyperparams[1:]
    elif self.options.noise_var_type == 'label':
      noise_var = self.options.noise_var_label * (self.Y.std()**2)
    else:
      noise_var = self.options.noise_var_value
    # Mean function -------------------------------------
    if hasattr(self.options, 'mean_func') and self.options.mean_func is not None:
      mean_func = self.options.mean_func
    else:
      if self.options.mean_func_type == 'mean':
        mean_func_const_value = self.YY.mean()
      elif self.options.mean_func_type == 'median':
        mean_func_const_value = np.median(self.YY)
      elif self.options.mean_func_type == 'upper_bound':
        mean_func_const_value = np.mean(self.YY) + 3 * np.std(self.YY)
      elif self.options.mean_func_type == 'const':
        mean_func_const_value = self.options.mean_func_const
      else:
        mean_func_const_value = 0
      mean_func = lambda x: np.array([mean_func_const_value] * len(x))
    # TODO: The noise and mean parts are reusing a lot of code from
    # gp_instances.SimpleGPFitter. Think of merging these two.
    # Kernel scale ---------------------------------------
    ke_scale = np.exp(gp_hyperparams[0])
    gp_hyperparams = gp_hyperparams[1:]
    # Fidelity kernel ------------------------------------
    if self.options.fidel_kernel_type == 'se':
      fidel_kernel, gp_hyperparams = self._get_se_kernel(self.fidel_dim,
        gp_hyperparams, self.options.fidel_use_same_bandwidth)
    elif self.options.fidel_kernel_type == 'poly':
      fidel_kernel, gp_hyperparams = self._get_poly_kernel(self.fidel_dim,
        self.options.fidel_poly_order, gp_hyperparams,
        self.options.fidel_use_same_scalings)
    # Domain kernel --------------------------------------
    if self.options.domain_kernel_type == 'se':
      domain_kernel, gp_hyperparams = self._get_se_kernel(self.domain_dim,
        gp_hyperparams, self.options.domain_use_same_bandwidth)
    elif self.options.domain_kernel_type == 'poly':
      domain_kernel, gp_hyperparams = self._get_poly_kernel(self.domain_dim,
        self.options.domain_poly_order, gp_hyperparams,
        self.options.domain_use_same_scalings)
    # Construct and return MF GP
    return MFGP(self.ZX, self.YY, range(self.fidel_dim),
                range(self.fidel_dim, self.domain_dim + self.fidel_dim),
                ke_scale, fidel_kernel, domain_kernel, mean_func, noise_var,
                reporter=self.reporter)

  @classmethod
  def _get_se_kernel(cls, dim, gp_hyperparams, use_same_bandwidth):
    """ Builds a squared exponential kernel. """
    if use_same_bandwidth:
      ke_dim_bandwidths = [np.exp(gp_hyperparams[0])] * dim
      gp_hyperparams = gp_hyperparams[1:]
    else:
      ke_dim_bandwidths = np.exp(gp_hyperparams[0:dim])
      gp_hyperparams = gp_hyperparams[dim:]
    kernel = SEKernel(dim=dim, scale=1, dim_bandwidths=ke_dim_bandwidths)
    return kernel, gp_hyperparams

  @classmethod
  def _get_poly_kernel(cls, dim, order, gp_hyperparams, use_same_scalings):
    """ Builds a polynomial kernel. """
    if use_same_scalings:
      ke_dim_scalings = [np.exp(gp_hyperparams[0])] * dim
      gp_hyperparams = gp_hyperparams[1:]
    else:
      ke_dim_scalings = np.exp(gp_hyperparams[0:dim])
      gp_hyperparams = gp_hyperparams[dim:]
    kernel = PolyKernel(dim=dim, order=order, scale=1, dim_scalings=ke_dim_scalings)
    return kernel, gp_hyperparams
  # _child_build_gp methods end here -----------------------------------------------------

