# Conquer
This package (in python) applies a convolution smoothing approach to fit linear quantile regression models. We refer this method as conquer. Normal-based and (multiplier) bootstrap confidence intervals for all slope coefficients are constructed. The Barzilai-Borwein gradient descent algorithm, initialized at a Huberized expectile regression estimate, is used to compute conquer estimators. This algorithm is scalable to very large-scale datasets.

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
Generate data from a linear model with random covariates. The dimension of the feature/covariate space is `p`, and the sample size is `n`. The itercept is 4, and all the `p` regression coefficients are set as 1 in magnitude. The errors are generated from a t-distribution of 2 degrees of freedom, centered by subtracting the population `tau`-quantile.

```
n, p = 8000, 400
mask = 2*rgt.binomial(1, 1/2, p) - 1
itcp, beta = 4, 1*np.ones(p)*mask
tau, t_df = 0.75, 2
runtime = 0

B = 200
itcp_se = np.empty(B)
coef_se = np.empty(B)
for b in range(B):
    X = rgt.normal(0, 1.5, size=(n,p))
    err = rgt.standard_t(t_df, n) - t.ppf(tau, t_df)
    Y = itcp + X.dot(beta) + err

    tic = time.time()
    sqr = conquer(X,Y)
    sqr_beta, sqr_fit = sqr.conquer(tau=tau)
    runtime += time.time() - tic

    itcp_se[b] = (sqr_beta[0] - itcp)**2
    coef_se[b] = (sqr_beta[1:] - beta).dot(sqr_beta[1:] - beta)
    
    
print('\nItcp_se:', np.mean(itcp_se),
      '\nCoef_se:', np.mean(coef_se),
      '\nRuntime:', runtime/B)
```

For statistical inference, our method provides 4 confidence intervals (CIs), which are normal-based CI using estimated covariance matrix, percentile bootstrap CI, pivotal bootstrap CI and normal-based CI using bootstrap variance estimates.

```
n, p = 500, 20
mask = 2*rgt.binomial(1, 1/2, p) - 1
itcp, beta = 4, 1*np.ones(p)*mask
tau, t_df = 0.75, 2

B = 200
ci_cover = np.zeros([4, p])
ci_width = np.empty([B, 4, p])
for b in range(B):
    X = rgt.normal(0, 1.5, size=(n,p))
    err = rgt.standard_t(t_df, n) - t.ppf(tau, t_df)
    Y = itcp + X.dot(beta) + err

    sqr = conquer(X, Y)
    mb_beta, boot_ci = sqr.mb_ci(tau)
    sqr_beta, norm_ci = sqr.norm_ci(tau)

    ci = np.concatenate([norm_ci[None,:,:], boot_ci], axis=0)
    
    for i in range(4):
        ci_cover[i,:] += 1*(beta >= ci[i,1:,0])*(beta<= ci[i,1:,1])
    ci_width[b,:,:] = ci[:,1:,1] - ci[:,1:,0]
    print(b)

print('All Coverage:',ci_cover/B,
      '\nAver. Cover:', np.mean(ci_cover/B, axis=1),
      '\nAver. Width:',np.mean(ci_width, axis=(0,2)))
```


## Reference

Fernandes, M., Guerre, E. and Horta, E. (2021). Smoothing quantile regressions. J. Bus. Econ. Statist. 39(1) 338–357. [Paper](https://www.tandfonline.com/doi/abs/10.1080/07350015.2019.1660177?journalCode=ubes20)

He, X., Pan, X., Tan, K. M. and Zhou, W.-X. (2020). Smoothed quantile regression with large-scale inference. Preprint. [Paper](https://arxiv.org/pdf/2012.05187.pdf)
 
