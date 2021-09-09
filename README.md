# conquer (Convolution Smoothed Quantile Regression)
This package (in python) consists of two parts. Part I applies a convolution smoothing approach to fit linear quantile regression models, referred to as *conquer*. Normal-based and (multiplier) bootstrap confidence intervals for all slope coefficients are constructed. The Barzilai-Borwein gradient descent algorithm, initialized at a Huberized expectile regression estimate, is used to compute conquer estimators. This algorithm is scalable to very large-scale datasets. For R implementation, see the ``conquer`` package on [``CRAN``](https://cran.r-project.org/web/packages/conquer/index.html) (also embedded in [``quantreg``](https://cran.r-project.org/web/packages/quantreg/index.html) as an alternative approach to `fn` and `pfn`).

Part II fits sparse quantile regression models in high dimensions via *L<sub>1</sub>*-penalized and iteratively reweighted *L<sub>1</sub>*-penalized (IRW-*L<sub>1</sub>*) conquer methods. The IRW method is motivated by the local linear approximation (LLA) algorithm proposed by [Zou & Li (2008)](https://projecteuclid.org/journals/annals-of-statistics/volume-36/issue-4/One-step-sparse-estimates-in-nonconcave-penalized-likelihood-models/10.1214/009053607000000802.full) for folded concave penalized estimation, typified by the SCAD penalty ([Fan & Li, 2001](https://fan.princeton.edu/papers/01/penlike.pdf)) and the minimax concave penalty (MCP) ([Zhang, 2010](https://projecteuclid.org/journals/annals-of-statistics/volume-38/issue-2/Nearly-unbiased-variable-selection-under-minimax-concave-penalty/10.1214/09-AOS729.full)). Computationally, the local adaptive majorize-minimization ([LAMM](https://github.com/XiaoouPan/ILAMM)) algorithm is used to solve each weighted *l<sub>1</sub>*-penalized conquer estimator.


## Dependencies

```
python >=3, numpy, 
optional: scipy, matplotlib
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
from conquer.linear_model import low_dim
```
Generate data from a linear model with random covariates. The dimension of the feature/covariate space is `p`, and the sample size is `n`. The itercept is 4, and all the `p` regression coefficients are set as 1 in magnitude. The errors are generated from the *t<sub>2</sub>*-distribution (*t*-distribution with 2 degrees of freedom), centered by subtracting the population *&tau;*-quantile of *t<sub>2</sub>*. 

When `p < n`, the module `qr` constains functions for fitting linear quantile regression models with uncertainty quantification. If the bandwidth `h` is unspecified, the default value *max\{0.01, \{&tau;(1- &tau;)\}^0.5 \{(p+log(n))/n\}^0.4\}* is used. The default kernel function is ``Laplacian``. Other choices are ``Gaussian``, ``Logistic``, ``Uniform`` and ``Epanechnikov``.

```
n, p = 8000, 400
itcp, beta = 4, np.ones(p)
tau, t_df = 0.75, 2

X = rgt.normal(0, 1.5, size=(n,p))
Y = itcp + X.dot(beta) + rgt.standard_t(t_df, n) - t.ppf(tau, t_df)

sqr = low_dim(X, Y, intercept=True)
model = sqr.fit(tau=tau)

# model['beta'] : conquer estimate (intercept & slope coefficients).
# model['res'] : n-vector of fitted residuals.
# model['niter'] : number of iterations.
# model['bw'] : bandwidth.
```

At each quantile level *&tau;*, the `norm_ci` and `boot_ci` methods provide four 100* (1-alpha)% confidence intervals (CIs) for regression coefficients: (i) normal distribution calibrated CI using estimated covariance matrix, (ii) percentile bootstrap CI, (iii) pivotal bootstrap CI, and (iv) normal-based CI using bootstrap variance estimates. For multiplier/weighted bootstrap implementation with `boot_ci`, the default weight distribution is ``Exponential``. Other choices are ``Rademacher``, ``Multinomial`` (Efron's nonparametric bootstrap), ``Gaussian``, ``Uniform`` and ``Folded-normal``. The latter two require a variance adjustment; see Remark 4.7 in [Paper](https://github.com/WenxinZhou/conquer/blob/main/papers/SmoothQR.pdf).

```
n, p = 500, 20
itcp, beta = 4, np.ones(p)
tau, t_df = 0.75, 2

X = rgt.normal(0, 1.5, size=(n,p))
Y = itcp + X.dot(beta) + rgt.standard_t(t_df, n) - t.ppf(tau, t_df)

sqr = low_dim(X, Y, intercept=True)
model1 = sqr.norm_ci(tau)
model2 = sqr.mb_ci(tau)

# model1['normal_ci'] : p+1 by 2 numpy array of normal CIs based on estimated asymptotic covariance matrix.
# model2['percentile_ci'] : p+1 by 2 numpy array of bootstrap percentile CIs.
# model2['pivotal_ci'] : p+1 by 2 numpy array of bootstrap pivotal CIs.
# model2['normal_ci'] : p+1 by 2 numpy array of normal CIs based on bootstrap variance estimates.
```

The second module `high_dim` contains functions that fit high-dimensional sparse quantile regression models. The default bandwidth value is *max\{0.05, \{&tau;(1- &tau;)\}^0.5 \{ log(p)/n\}^0.25\}*. To choose the penalty level, the `self_tuning` function implements the simulation-based approach proposed by [Belloni & Chernozhukov (2011)](https://projecteuclid.org/journals/annals-of-statistics/volume-39/issue-1/%e2%84%931-penalized-quantile-regression-in-high-dimensional-sparse-models/10.1214/10-AOS827.full). 
The `l1` and `irw` functions compute *L<sub>1</sub>*- and IRW-*L<sub>1</sub>*-penalized conquer estimators, respectively. For the latter, the default concave penality is `SCAD` with constant `a=3.7` ([Fan & Li, 2001](https://fan.princeton.edu/papers/01/penlike.pdf)). Given a sequence of penalty levels, the solution paths can be computed by `l1_path` and `irw_path`.

```
import numpy as np
import numpy.random as rgt
from scipy.stats import t
from conquer.linear_model import high_dim

s, p, n = 8, 1028, 256
tau = 0.8
itcp, beta = 4, np.zeros(p)
beta[:15] = [1.8, 0, 1.6, 0, 1.4, 0, 1.2, 0, 1, 0, -1, 0, -1.2, 0, -1.6]

X = rgt.normal(0, 1, size=(n,p))
Y = itcp + X.dot(beta) + rgt.standard_t(2,size=n) - t.ppf(tau,df=2)

hd_sqr = high_dim(X, Y, intercept=True)
sim_lambda = 0.75*np.quantile(hd_sqr.self_tuning(tau), 0.9)
lambda_seq = np.linspace(0.5*sim_lambda, sim_lambda, L=20)

## l1-penalized conquer
l1_model = hd_sqr.l1(Lambda=sim_lambda, tau=tau)

## iteratively reweighted l1-penalized conquer (default is SCAD penality)
irw_model = hd_sqr.irw(Lambda=sim_lambda, tau=tau)

## solution path of l1-penalized conquer
l1_models = hd_sqr.l1_path(lambda_seq=lambda_seq, tau=tau)

## solution path of irw-l1-penalized conquer
irw_models = hd_sqr.irw_path(lambda_seq=lambda_seq, tau=tau)

## bootstrap model selection
boot_model = hd_sqr.boot_select(sim_lambda, tau, weight="Multinomial")
print('selected model via bootstrap:', boot_model['majority_vote'])
```


## References

Fernandes, M., Guerre, E. and Horta, E. (2021). Smoothing quantile regressions. *J. Bus. Econ. Statist.* **39**(1) 338–357. [Paper](https://www.tandfonline.com/doi/abs/10.1080/07350015.2019.1660177?journalCode=ubes20)

He, X., Pan, X., Tan, K. M. and Zhou, W.-X. (2020). Smoothed quantile regression with large-scale inference. *J. Econom.* [Paper](https://github.com/WenxinZhou/conquer/blob/main/papers/SmoothQR.pdf)

Koenker, R. (2005). *Quantile Regression*. Cambridge University Press, Cambridge. [Book](https://www.cambridge.org/core/books/quantile-regression/C18AE7BCF3EC43C16937390D44A328B1)

Pan, X., Sun, Q. and Zhou, W.-X. (2021). Iteratively reweighted *l<sub>1</sub>*-penalized robust regression. *Electron. J. Stat.* **15**(1) 3287-3348. [Paper](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-15/issue-1/Iteratively-reweighted-%E2%84%931-penalized-robust-regression/10.1214/21-EJS1862.full)

Tan, K. M., Wang, L. and Zhou, W.-X. (2021). High-dimensional quantile regression: convolution smoothing and concave regularization. *Preprint*. [Paper](https://github.com/WenxinZhou/conquer/blob/main/papers/NcvxQR.pdf)

## License 

This package is released under the GPL-3.0 license.
