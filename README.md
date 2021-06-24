# rct
[![Build Status](https://github.com/sylvaingchassang/rct/actions/workflows/test_rct.yml/badge.svg)](https://github.com/sylvaingchassang/rct/actions/workflows/test_rct.yml/badge.svg)

### What this package does

This package provides tools to generate robust and balanced random assignments
following [Banerjee, Chassang, Montero, and Snowberg (2019)](https://www.sylvainchassang.org/assets/papers/adversarial_experimentation.pdf).

The `RCT`, `KRerandomizedRCT`, and `QuantileTargetingRCT` classes of
 the `rct.design` module implement RCT, K-rerandomized
 RCT, and Quantile Targeting RCT designs described in [Banerjee, Chassang, Montero, and Snowberg (2019)](https://www.sylvainchassang.org/assets/papers/adversarial_experimentation.pdf).
  
For each design, `assignment_from_iid` draws designs selected from i.i.d. assignments;
  `assignment_from_shuffled` draws designs selected from exchangeable
  assignments guaranteed to exactly match desired sampling weights (up to
  integer issues).

The package allows for an arbitrary number of treatment arms, specified via
the `weights` argument in each design.

`rct` implements various balance objectives, including:   
 - minimizing the Mahalanobis distance between the mean of selected
    covariates  across treatment arms;   
 - maximizing the minimum p-value for the regression of covariates on
     treatment dummies;   
 - soft blocking on selected covariates;   
 - linear combinations of existing objectives.

Customizing balance objectives, besides linear combinations of existing balance functions, is straightforward. First, you can pass different 
aggregating functions to the `BalanceObjective` constructor. For instance, this would allow to maximize the mean p-value rather than the minimum p-value. Second, you can simply define a new class inheriting from `BalanceObjective`  and implementing the abstract method `_balance_func`.


### Citation

To cite `rct` in publications, use    
```
Banerjee, Abhijit, Sylvain Chassang, Sergio Montero, and Erik Snowberg.   
A theory of experimenters. NBER Working Paper No. w23867. National Bureau of Economic Research, 2017.
```
The corresponding [`bibtex` entry](w23867.bib) is:   
```
@techreport{NBERw23867,
 title = "A Theory of Experimenters",
 author = "Banerjee, Abhijit and Chassang, Sylvain and Montero, Sergio and Snowberg, Erik",
 institution = "National Bureau of Economic Research",
 type = "Working Paper",
 series = "Working Paper Series",
 number = "23867",
 year = "2017",
 month = "September",
 doi = {10.3386/w23867},
 URL = "http://www.nber.org/papers/w23867",
 abstract = {This paper proposes a decision-theoretic framework for experiment design. We model experimenters as ambiguity-averse decision-makers, who make trade-offs between subjective expected performance and robustness. This framework accounts for experimenters' preference for randomization, and clarifies the circumstances in which randomization is optimal: when the available sample size is large enough or robustness is an important concern. We illustrate the practical value of such a framework by studying the issue of rerandomization. Rerandomization creates a trade-off between subjective performance and robustness. However,  robustness loss grows very slowly with the number of times one randomizes. This argues for rerandomizing in most environments.},
}
```

### Installation

This package is tested for `python 3.6` and `python 3.7` under Ubuntu
Linux 16.04.

You may download the package via `pip`:

`$ pip install rct`

this will install all required dependencies.

Alternatively, if you want to use recent updates, you can clone (`git@github.com:sylvaingchassang/rct.git`) or [download a `.zip`](https://github.com/sylvaingchassang/rct/archive/master.zip) of the repo. If you
do so you must install requirements for the package manually. With `pip`, run   

`./rct$ pip install -r requirements.txt`

### Running tests

Before using the package, you may want to check that unit and
integration tests pass on your machine. To this end, run

`./rct$ pytest --cov=. --cov-report=term-missing`

### Examples

Example notebooks illustrate the use of `rct` modules:
 - [`rct/notebooks/examples_rct.ipynb`](https://github.com/sylvaingchassang/rct/blob/master/notebooks/examples_rct.ipynb) shows how to generate
 traditional i.i.d. and shuffled RCT assignments for binary and ternary
 treatments. The `pvalue_report` function provides a useful summary of
 assignment balance by reporting the `(#treatments -1, #covariates)`
 matrix of p-values obtained from regressing covariates on different
 treatment dummies.

 - [`rct/notebooks/examples_k_rerandomized_rct.ipynb`](https://github.com/sylvaingchassang/rct/blob/master/notebooks/examples_k_rerandomized_rct.ipynb) shows how to
 obtain k-rerandomized i.i.d. and shuffled assignments under various balance
  objectives.

  - [`rct/notebooks/examples_quantile_targeting_rct.ipynb`](https://github.com/sylvaingchassang/rct/blob/master/notebooks/examples_quantile_targeting_rct.ipynb) performs a
   similar exercise for quantile targeting experiment designs.

Integration tests located at `rct/tests/test_integration.py` replicate
the content of these notebooks.

### Contribute

If you want to improve `rct` please reach out! 

Whether you are a programmer who wants to improve our code, or an experiment designer with a
practical comment, or a new design idea, we want to talk to you!

On our current todo list (2019/09/20):   
 - adding type hints to improve readability;   
 - profiling & speed improvement;   
 - implementing sequential designs.   
