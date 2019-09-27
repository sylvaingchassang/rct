# xdesign
[![Build Status](https://travis-ci.com/sylvaingchassang/xdesign.svg?branch=master)](https://travis-ci.com/sylvaingchassang/xdesign)

### What this package does

This package provides tools to generate robust and balanced random assignments
following [Banerjee, Chassang, Montero, and Snowberg (2019)](https://www.sylvainchassang.org/assets/papers/adversarial_experimentation.pdf).

The `RCT`, `KRerandomizedRCT`, and `QuantileTargetingRCT` classes of
 the `xdesign.design` module implement RCT, K-rerandomized
 RCT, and Quantile Targeting RCT designs described in [Banerjee, Chassang, Montero, and Snowberg (2019)](https://www.sylvainchassang.org/assets/papers/adversarial_experimentation.pdf).
  
For each design, `assignment_from_iid` draws designs selected from i.i.d. assignments;
  `assignment_from_shuffled` draws designs selected from exchangeable
  assignments guaranteed to exactly match desired sampling weights (up to
  integer issues).

The package allows for an arbitrary number of treatment arms, specified via
the `weights` argument in each design.

`xdesign` implements various balance objectives, including:   
 - minimizing the Mahalanobis distance between the mean of selected
    covariates  across treatment arms;   
 - maximizing the minimum p-value for the regression of covariates on
     treatment dummies;   
 - soft blocking on selected covariates;   
 - linear combinations of existing objectives.

Customizing balance objectives, besides linear combinations of existing balance functions, is straightforward. First, you can pass different 
aggregating functions to the `BalanceObjective` constructor. For instance, this would allow to maximize the mean p-value rather than the minimum p-value. Second, you can simply define a new class inheriting from `BalanceObjective`  and implementing the abstract method `_balance_func`.


### Citation

To cite `xdesign` in publications, use    
```
Banerjee, Abhijit, Sylvain Chassang, Sergio Montero, and Erik Snowberg.   
A theory of experimenters. No. w23867. National Bureau of Economic Research, 2017.
```
Corresponding `bibtex` entry:   
```
@techreport{banerjee2017theory,   
  title={A theory of experimenters},   
  author={Banerjee, Abhijit and Chassang, Sylvain and Montero, Sergio and Snowberg, Erik},   
  year={2017},   
  institution={National Bureau of Economic Research}   
}
```

### Installation

This package is tested for `python 3.6` and `python 3.7` under Ubuntu
Linux 16.04.

You may download the package via `pip`:

`$ pip install xdesign`

this will install all required dependencies.

Alternatively, you can clone (`git@github.com:sylvaingchassang/xdesign.git`) or [download a `.zip`](https://github.com/sylvaingchassang/xdesign/archive/master.zip) of the repo. If you
do so you must install requirements for the package manually. With `pip`, run   

`./xdesign$ pip install -r requirements.txt`

### Running tests

Before using the package, you may want to check that unit and
integration tests pass on your machine. To this end, run

`./xdesign$ pytest --cov=. --cov-report=term-missing`

### Examples

Example notebooks illustrate the use of `xdesign` modules:
 - [`xdesign/notebooks/examples_rct.ipynb`](https://github.com/sylvaingchassang/xdesign/blob/master/notebooks/examples_rct.ipynb) shows how to generate
 traditional i.i.d. and shuffled RCT assignments for binary and ternary
 treatments. The `pvalue_report` function provides a useful summary of
 assignment balance by reporting the `(#treatments -1, #covariates)`
 matrix of p-values obtained from regressing covariates on different
 treatment dummies.

 - [`xdesign/notebooks/examples_k_rerandomized_rct.ipynb`](https://github.com/sylvaingchassang/xdesign/blob/master/notebooks/examples_k_rerandomized_rct.ipynb) shows how to
 obtain k-rerandomized i.i.d. and shuffled assignments under various balance
  objectives.

  - [`xdesign/notebooks/examples_quantile_targeting_rct.ipynb`](https://github.com/sylvaingchassang/xdesign/blob/master/notebooks/examples_quantile_targeting_rct.ipynb) performs a
   similar exercise for quantile targeting experiment designs.

Integration tests located at `xdesign/tests/test_integration.py` replicate
the content of these notebooks.

### Contribute

If you want to improve `xdesign` please reach out! 

Whether you are a programmer who wants to improve our code, or an experiment designer with a
practical comment, or a new design idea, we want to talk to you!

On our current todo list (2019/09/20):   
 - adding type hints to improve readability;   
 - profiling & speed improvement;   
 - implementing sequential designs.   
