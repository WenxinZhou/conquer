# conquer (Convolution Smoothed Quantile Regression)
This package (in python) consists of two parts. Part I applies a convolution smoothing approach to fit linear quantile regression models, referred to as *conquer*. Normal-based and (multiplier) bootstrap confidence intervals for all slope coefficients are constructed. The Barzilai-Borwein gradient descent algorithm, initialized at a Huberized expectile regression estimate, is used to compute conquer estimators. This algorithm is scalable to very large-scale datasets. For R implementation, see the ``conquer`` package on [``CRAN``](https://cran.r-project.org/package=conquer) (also embedded in [``quantreg``](https://cran.r-project.org/package=quantreg) as an alternative approach to `fn` and `pfn`).

Part II fits sparse quantile regression models in high dimensions via *L<sub>1</sub>*-penalized and iteratively reweighted *L<sub>1</sub>*-penalized (IRW-*L<sub>1</sub>*) conquer methods. The IRW method is motivated by the local linear approximation (LLA) algorithm proposed by [Zou & Li (2008)](https://doi.org/10.1214/009053607000000802) for folded concave penalized estimation, typified by the SCAD penalty ([Fan & Li, 2001](https://fan.princeton.edu/papers/01/penlike.pdf)) and the minimax concave penalty (MCP) ([Zhang, 2010](https://doi.org/10.1214/09-AOS729)). Computationally, the local adaptive majorize-minimization algorithm ([LAMM](https://doi.org/10.1214/17-AOS1568)) is used to solve each weighted *l<sub>1</sub>*-penalized conquer estimator. For the purposes of comparison, the proximal ADMM algorithm ([pADMM](https://doi.org/10.1080/00401706.2017.1345703)) is also implemented.


## Dependencies

```
python >=3, numpy, scipy
optional: pandas, matplotlib
```


## Installation

Download the folder ``conquer`` (containing `linear_model.py`) in your working directory, or clone the git repo. and install:
```
git clone https://github.com/WenxinZhou/conquer.git
python setup.py install
```

## Examples

```
import numpy as np
import numpy.random as rgt
from scipy.stats import t
import time
from conquer.linear_model import low_dim, high_dim, pADMM
```
Generate data from a linear model with random covariates. The dimension of the feature/covariate space is `p`, and the sample size is `n`. The itercept is 4, and all the `p` regression coefficients are set as 1 in magnitude. The errors are generated from the *t<sub>2</sub>*-distribution (*t*-distribution with 2 degrees of freedom), centered by subtracting the population *&tau;*-quantile of *t<sub>2</sub>*. 

When `p < n`, the module `low_dim` constains functions for fitting linear quantile regression models with uncertainty quantification. If the bandwidth `h` is unspecified, the default value *max\{0.01, \{&tau;(1- &tau;)\}^0.5 \{(p+log(n))/n\}^0.4\}* is used. The default kernel function is ``Laplacian``. Other choices are ``Gaussian``, ``Logistic``, ``Uniform`` and ``Epanechnikov``.

```
n, p = 8000, 400
itcp, beta = 4, np.ones(p)
tau, t_df = 0.75, 2

X = rgt.normal(0, 1.5, size=(n,p))
Y = itcp + X.dot(beta) + rgt.standard_t(t_df, n) - t.ppf(tau, t_df)

qr = low_dim(X, Y, intercept=True)
model = qr.fit(tau=tau)

# model['beta'] : conquer estimate (intercept & slope coefficients).
# model['res'] : n-vector of fitted residuals.
# model['niter'] : number of iterations.
# model['bw'] : bandwidth.
```

At each quantile level *&tau;*, the `norm_ci` and `boot_ci` methods provide four 100* (1-alpha)% confidence intervals (CIs) for regression coefficients: (i) normal distribution calibrated CI using estimated covariance matrix, (ii) percentile bootstrap CI, (iii) pivotal bootstrap CI, and (iv) normal-based CI using bootstrap variance estimates. For multiplier/weighted bootstrap implementation with `boot_ci`, the default weight distribution is ``Exponential``. Other choices are ``Rademacher``, ``Multinomial`` (Efron's nonparametric bootstrap), ``Gaussian``, ``Uniform`` and ``Folded-normal``. The latter two require a variance adjustment; see Remark 4.7 in [Paper](https://doi.org/10.1016/j.jeconom.2021.07.010).

```
n, p = 500, 20
itcp, beta = 4, np.ones(p)
tau, t_df = 0.75, 2

X = rgt.normal(0, 1.5, size=(n,p))
Y = itcp + X.dot(beta) + rgt.standard_t(t_df, n) - t.ppf(tau, t_df)

qr = low_dim(X, Y, intercept=True)
model1 = qr.norm_ci(tau=tau)
model2 = qr.mb_ci(tau=tau)

# model1['normal_ci'] : p+1 by 2 numpy array of normal CIs based on estimated asymptotic covariance matrix.
# model2['percentile_ci'] : p+1 by 2 numpy array of bootstrap percentile CIs.
# model2['pivotal_ci'] : p+1 by 2 numpy array of bootstrap pivotal CIs.
# model2['normal_ci'] : p+1 by 2 numpy array of normal CIs based on bootstrap variance estimates.
```

The module `high_dim` contains functions that fit high-dimensional sparse quantile regression models through the LAMM algorithm. The default bandwidth value is *max\{0.05, \{&tau;(1- &tau;)\}^0.5 \{ log(p)/n\}^0.25\}*. To choose the penalty level, the `self_tuning` function implements the simulation-based approach proposed by [Belloni & Chernozhukov (2011)](https://doi.org/10.1214/10-AOS827). 
The `l1` and `irw` functions compute *L<sub>1</sub>*- and IRW-*L<sub>1</sub>*-penalized conquer estimators, respectively. For the latter, the default concave penality is `SCAD` with constant `a=3.7` ([Fan & Li, 2001](https://fan.princeton.edu/papers/01/penlike.pdf)). Given a sequence of penalty levels, the solution paths can be computed by `l1_path` and `irw_path`. 

```
p, n = 1028, 256
tau = 0.8
itcp, beta = 4, np.zeros(p)
beta[:15] = [1.8, 0, 1.6, 0, 1.4, 0, 1.2, 0, 1, 0, -1, 0, -1.2, 0, -1.6]

X = rgt.normal(0, 1, size=(n,p))
Y = itcp + X.dot(beta) + rgt.standard_t(2,size=n) - t.ppf(tau,df=2)

sqr = high_dim(X, Y, intercept=True)
lambda_max = np.max(sqr.self_tuning(tau))
lambda_seq = np.linspace(0.25*lambda_max, lambda_max, num=20)

## l1-penalized conquer
l1_model = sqr.l1(tau=tau, Lambda=0.75*lambda_max)

## iteratively reweighted l1-penalized conquer (default is SCAD penality)
irw_model = sqr.irw(tau=tau, Lambda=0.75*lambda_max)

## solution path of l1-penalized conquer
l1_models = sqr.l1_path(tau=tau, lambda_seq=lambda_seq)

## solution path of irw-l1-penalized conquer
irw_models = sqr.irw_path(tau=tau, lambda_seq=lambda_seq)

## bootstrap model selection
boot_model = sqr.boot_select(tau=tau, Lambda=sim_lambda, weight="Multinomial")
print('selected model via bootstrap:', boot_model['majority_vote'])
```

The module `pADMM` has a similar functionality to `high_dim`. It employs the proximal ADMM algorithm to solve weighted *L<sub>1</sub>*-penalized quantile regression (without smoothing).
```
lambda_max = np.max(high_dim(X, Y, intercept=True).self_tuning(tau))
lambda_seq = np.linspace(0.25*lambda_max, lambda_max, num=20)
admm = pADMM(X, Y, intercept=True)

## l1-penalized QR
l1_model = admm.l1(tau=tau, Lambda=0.5*lambda_max)

## iteratively reweighted l1-penalized QR (default is SCAD penality)
irw_model = admm.irw(tau=tau, Lambda=0.75*lambda_max)

## solution path of l1-penalized QR
l1_models = admm.l1_path(tau=tau, lambda_seq=lambda_seq)

## solution path of irw-l1-penalized QR
irw_models = admm.irw_path(tau=tau, lambda_seq=lambda_seq)
```


## References

Fernandes, M., Guerre, E. and Horta, E. (2021). Smoothing quantile regressions. *J. Bus. Econ. Statist.* **39**(1) 338–357. [Paper](https://www.tandfonline.com/doi/abs/10.1080/07350015.2019.1660177?journalCode=ubes20)

He, X., Pan, X., Tan, K. M. and Zhou, W.-X. (2021). Smoothed quantile regression with large-scale inference. *J. Econom.* [Paper](https://doi.org/10.1016/j.jeconom.2021.07.010)

Koenker, R. (2005). *Quantile Regression*. Cambridge University Press, Cambridge. [Book](https://www.cambridge.org/core/books/quantile-regression/C18AE7BCF3EC43C16937390D44A328B1)

Pan, X., Sun, Q. and Zhou, W.-X. (2021). Iteratively reweighted *l<sub>1</sub>*-penalized robust regression. *Electron. J. Stat.* **15**(1) 3287-3348. [Paper](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-15/issue-1/Iteratively-reweighted-%E2%84%931-penalized-robust-regression/10.1214/21-EJS1862.full)

Tan, K. M., Wang, L. and Zhou, W.-X. (2022). High-dimensional quantile regression: convolution smoothing and concave regularization. *J. R. Stat. Soc. B.*  **84**(1) 205-233. [Paper](https://rss.onlinelibrary.wiley.com/doi/10.1111/rssb.12485)

## License 

This package is released under the GPL-3.0 license.
