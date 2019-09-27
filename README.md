# xdesign
[![Build Status](https://travis-ci.com/sylvaingchassang/xdesign.svg?branch=master)](https://travis-ci.com/sylvaingchassang/xdesign)

### What this package does

This package provides tools to generate robust and balanced random assignments
following [Banerjee, Chassang, Montero S, and Snowberg (2019)](https://www.sylvainchassang.org/assets/papers/adversarial_experimentation.pdf).

The package allows for an arbitrary number of treatment arms (specified via
the `weights` argument).

`xdesign` implements various balance objectives, including:   
    - minimizing the Mahalanobis distance between the mean of selected
    covariates  across treatment arms;   
    - maximizing the minimum p-value for the regression of covariates on
     treatment dummies;   
    - soft blocking on selected covariates;   
    - linear combinations of existing objectives.

Customizing balance functions is easy, either by passing appropriate
aggregating functions to the `BalanceObjective` constructor, or by
defining a new class, inheriting from `BalanceObjective`  and implementing
the abstract method `_balance_func`.

Finally the `RCT`, `KRerandomizedRCT`, and `QuantileTargetingRCT` classes of
 the `xdesign.design` module implement the standard RCT, K-rerandomized
 experiment, and quantile targeting experiment designs described in [Banerjee, Chassang, Montero S, and Snowberg (2019)](https://www.sylvainchassang.org/assets/papers/adversarial_experimentation.pdf).
  
For each design, `assignment_from_iid` draws designs selected from i.i.d. assignments;
  `assignment_from_shuffled` draws designs selected from exchangeable
  assignments guaranteed to exactly match desired sampling weights (up to
  integer issues).

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

Alternatively, you can [clone]() or [download a `.zip`]() the repo. If you
do so you must install requirements for the package manually. With `pip`, run   

`./xdesign$ pip install -r requirements.txt`

### Running tests

Before using the package, you may want to check that unit and
integration tests pass on your machine. To this end, run

`./xdesign$ pytest --cov=. --cov-report=term-missing`

### Examples

Example notebooks illustrate the use of `xdesign` modules:
 - [`xdesign/notebooks/examples_rct.ipynb`]() shows how to generate
 traditional i.i.d. and shuffled RCT assignments for binary and ternary
 treatments. The `pvalue_report` function provides a useful summary of
 assignment balance by reporting the `(#treatments -1 x #covariates)`
 matrix of p-values obtained from regressing covariates on different
 treatment dummies.

 - [`xdesign/notebooks/examples_k_rerandomized_rct.ipynb`]() shows how to
 obtain k-rerandomized i.i.d. and shuffled assignments under various balance
  objectives.

  - [`xdesign/notebooks/examples_quantile_targeting_rct.ipynb`]() performs a
   similar exercise for quantile targeting experiment designs.

Integration tests located at `xdesign/tests/test_integration.py` replicate
and verify the content of these notebooks.

### Contribute

If you want to improve `xdesign` please reach out, whether you are a
programmer who wants to improve our code, or an experiment designer with a
practical comment, or a new design idea you'd like to see included.

On our todo list:
    - adding type hints to improve readability;
    - profiling & speed improvement;
    - implementing sequential designs.


### Other randomization resources

Uschner D, Schindler D, Hilgers R, Heussen N (2018). “randomizeR: An R Package for the Assessment and Implementation of Randomization in Clinical Trials.” Journal of Statistical Software, 85(8), 1–22. doi: 10.18637/jss.v085.i08.    
[[CRAN Package]](https://cran.r-project.org/web/packages/randomizeR/index.html)
