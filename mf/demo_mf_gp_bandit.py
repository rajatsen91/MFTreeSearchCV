"""
  A Demo for MF-GP-Bandit
"""
# pylint: disable=invalid-name

# Local
from synthetic_functions import get_mf_hartmann_as_mfof
from mf_gp_bandit import mfgpb_from_mfoptfunc

# methods = ['gp_ucb', 'gp_ei', 'mf_gp_ucb']
methods = ['mf_gp_ucb']

fidel_dim = 2
domain_dim = 6
num_max_hf_queries = 200

# fidel_dim = 1
# domain_dim = 3
# num_max_hf_queries = 100

def main():
  """ Main function. """
  mfof = get_mf_hartmann_as_mfof(fidel_dim, domain_dim)
  capital = num_max_hf_queries * mfof.opt_fidel_cost
  # Execute each method
  for meth in methods:
    print('Method:: %s, opt-val: %0.4f\n============================================='%(
          meth, mfof.opt_val))
    mfgpb_from_mfoptfunc(mfof, capital, meth)


if __name__ == '__main__':
  main()

