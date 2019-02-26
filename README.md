# MFTreeSearchCV

This is a package for fast hyper-parameter tuning using noisy multi-fidelity tree-search for scikit-learn estimators (classifiers/regressors). Given ranges and types (categorical, integer, reals) for several hyper-parameters, this package is desgined to search for a good configuartion by treating the k-fold cross-validation errors and different setting under different fidelities (levels of approximation based on amount od data used), as a multi-fidelity noisy black-box function. This work is based on the publications:

1. [A deterministic version of MF Tree Seach](http://proceedings.mlr.press/v80/sen18a/sen18a.pdf)
2. [A version that can hadle noise -- which is there in tuning](https://arxiv.org/pdf/1810.10482)

Please cite the above papers, if using this software in any publications/reports. 


### Prerequisites

1. You will need a C and a Fortran compiler such as gnu95. Please install them by following the correct steps for your machine and OS. 

2. You will need the following python packacges: sklearn, numpy, brewer2mpl, scipy, 

### Installing

1. Go to the location of your choice and clone the repo.

```
git clone https://github.com/rajatsen91/MFTreeSearchCV.git
```

2. Now the next steps are to setup the multi-fidelity enviroment. 
- We will assume that the location of the project is `/home/MFTreeSearchCV/`
- You first need to build the direct fortran library. For this `cd` into
  `/home/MFTreeSearchCV/utils/direct_fortran` and run `bash make_direct.sh`. You will need a fortran compiler
  such as gnu95. Once this is done, you can run `simple_direct_test.py` to make sure that
  it was installed correctly.
- Edit the `set_up_gittins` file to change the `GITTINS_PATH` to the local location of the repo. 
```
GITTINS_PATH=/home/MFTreeSearchCV
``` 
- Run `source set_up_gittins` to set up all environment variables.
- To test the installation, run `bash run_all_tests.sh`. Some of the tests are
  probabilistic and could fail at times. Some tests that require a scratch directory will fail,
  but nothing to worry about. 


### Usage 

The usage of this repository is designed to be similar to parameter tuning function in sklearn.model_selection. The main function is `MFTreeSearchCV.MFTreeSearchCV`. The arguments and methods of thsi function are as follows,



## Authors

* **Rajat Sen** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* We acknowledge the support from [@kirthevasank](https://github.com/kirthevasank) for providing the original multi-fidelity black-box function code base.
