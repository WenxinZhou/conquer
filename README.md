# Conquer
This package (in python) consists of two parts. Part I applies a convolution smoothing approach to fit linear quantile regression models, referred to as *conquer*. Normal-based and (multiplier) bootstrap confidence intervals for all slope coefficients are constructed. The Barzilai-Borwein gradient descent algorithm, initialized at a Huberized expectile regression estimate, is used to compute conquer estimators. This algorithm is scalable to very large-scale datasets. For R implementation, see the ``conquer`` package on [``CRAN``](https://cran.r-project.org/web/packages/conquer/index.html) (also embedded in [``quantreg``](https://cran.r-project.org/web/packages/quantreg/index.html) as an alternative approach to `fn` and `pfn`).

Part II fits sparse quantile regression models in high dimensions via *L<sub>1</sub>*-penalized and iteratively reweighted *L<sub>1</sub>*-penalized (IRW-*L<sub>1</sub>*) conquer methods. The latter is motivated by the local linear approximation (LLA) algorithm proposed by [Zou & Li (2008)](https://projecteuclid.org/journals/annals-of-statistics/volume-36/issue-4/One-step-sparse-estimates-in-nonconcave-penalized-likelihood-models/10.1214/009053607000000802.full) for folded concave penalized estimation, typified by the SCAD penalty ([Fan & Li, 2001)](https://fan.princeton.edu/papers/01/penlike.pdf)) and minimax concave penalty (MCP) ([Zhang, 2010](https://projecteuclid.org/journals/annals-of-statistics/volume-38/issue-2/Nearly-unbiased-variable-selection-under-minimax-concave-penalty/10.1214/09-AOS729.full)). Computationally, the local adaptive majorize-minimization ([LAMM](https://github.com/XiaoouPan/ILAMM)) algorithm is used to solve each weighted *l<sub>1</sub>*-penalized conquer estimator.


## Dependencies

```
python >=3, numpy, 
optional: scipy, matplotlib
```

## Examples

```
import numpy as np
import numpy.random as rgt
from conquer import conquer
from scipy.stats import t
import time
```
Generate data from a linear model with random covariates. The dimension of the feature/covariate space is `p`, and the sample size is `n`. The itercept is 4, and all the `p` regression coefficients are set as 1 in magnitude. The errors are generated from the *t<sub>2</sub>*-distribution (*t*-distribution with 2 degrees of freedom), centered by subtracting the population *&tau;*-quantile of *t<sub>2</sub>*. 

If the bandwidth `h` is not specified, the default value *max\{0.01, \{&tau;(1- &tau;)\}^0.5 \{(p+log(n))/n\}^0.4\}* is used. The default kernel function is ``Laplacian``. Other choices are ``Gaussian``, ``Logistic``, ``Uniform`` and ``Epanechnikov``.

```
n, p = 8000, 400
mask = 2*rgt.binomial(1, 1/2, p) - 1
itcp, beta = 4, 1*np.ones(p)*mask
tau, t_df = 0.75, 2

X = rgt.normal(0, 1.5, size=(n,p))
Y = itcp + X.dot(beta) + rgt.standard_t(t_df, n) - t.ppf(tau, t_df)

sqr = conquer(X,Y)
sqr_beta, sqr_fit = sqr.fit(tau=tau)

# sqr_beta is the conquer estimator.
# sqr_fit is a list of residual vector, number of iterations, and bandwidth.

```

At each quantile level *&tau;*, our method provides four 100* (1-alpha)% confidence intervals (CIs) for regression coefficients: (i) normal distribution calibrated CI using estimated covariance matrix, (ii) percentile bootstrap CI, (iii) pivotal bootstrap CI, and (iv) normal-based CI using bootstrap variance estimates. In the multiplier/weighted bootstrap implementation, the default weight distribution is ``Exponential``. Other choices are ``Rademacher``, ``Multinomial`` (Efron's nonparametric bootstrap), ``Gaussian``, ``Uniform`` and ``Folded-normal``. The latter two require a variance adjustment; see Remark 4.6 in [Paper](https://arxiv.org/pdf/2012.05187.pdf).

```
n, p = 500, 20
mask = 2*rgt.binomial(1, 1/2, p) - 1
itcp, beta = 4, 1*np.ones(p)*mask
tau, t_df = 0.75, 2

X = rgt.normal(0, 1.5, size=(n,p))
Y = itcp + X.dot(beta) + rgt.standard_t(t_df, n) - t.ppf(tau, t_df)

sqr = conquer(X, Y)
mb_beta, boot_ci = sqr.mb_ci(tau)
sqr_beta, norm_ci = sqr.norm_ci(tau)

# mb_beta is a numpy array. 1st column: conquer estimator; 2nd to last: bootstrap estimates.
# boot_ci is a 3 by p+1 by 2 (or 3 by p by 2) numpy array. 
# boot_ci[1,:,:]: percentile CI; 
# boot_ci[2,:,:]: pivotal CI; 
# boot_ci[3,:,:]: normal-based CI using bootstrap variance estimate.

# norm_ci is a p+1 by 2 (or p by 2) numpy array. Normal CI based on estimated asymptotic covariance matrix

```

## Reference
Fernandes, M., Guerre, E. and Horta, E. (2021). Smoothing quantile regressions. *J. Bus. Econ. Statist.* **39**(1) 338–357. [Paper](https://www.tandfonline.com/doi/abs/10.1080/07350015.2019.1660177?journalCode=ubes20)

He, X., Pan, X., Tan, K. M. and Zhou, W.-X. (2020). Smoothed quantile regression with large-scale inference. *Preprint*. [Paper](https://arxiv.org/pdf/2012.05187.pdf)

Koenker, R. (2005). *Quantile Regression*. Cambridge University Press, Cambridge. [Book](https://www.cambridge.org/core/books/quantile-regression/C18AE7BCF3EC43C16937390D44A328B1)

Tan, K. M., Wang, L. and Zhou, W.-X. (2020). High-dimensional quantile regression: convolution smoothing and concave regularization. *Preprint*.

## License 

This package is released under the GPL-3.0 license.
