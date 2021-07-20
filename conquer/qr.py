import numpy as np
import numpy.random as rgt
from scipy.stats import norm

class conquer():
    '''
        Convolution Smoothed Quantile Regression

    Reference
    ---------
    Smoothed Quantile Regression with Large-Scale Inference (2020)
    by Xuming He, Xiaoou Pan, Kean Ming Tan & Wenxin Zhou
    https://arxiv.org/abs/2012.05187
    '''
    kernels = ["Laplacian", "Gaussian", "Logistic", "Uniform", "Epanechnikov"]
    weights = ["Exponential", "Multinomial", "Rademacher", "Gaussian", "Uniform", "Folded-normal"]

    def __init__(self, X, Y, intercept=True, max_iter=500, tol=1e-10):
        '''
        Arguments
        ---------
            X : n by p numpy array of covariates; each row is an observation vector.
            
            Y : n by 1 numpy array of response variables. 
            
            intercept : logical flag for adding an intercept to the model.

        Internal Optimization Parameters
        --------------------------------
        max_iter : maximum numder of iterations in the GD-BB algorithm; default is 500.
        
        tol : minimum change in (squared) Euclidean distance for stopping GD iterations; default is 1e-10.
        '''
        n = len(Y)
        self.Y = Y
        self.mX, self.sdX = np.mean(X, axis=0), np.std(X, axis=0)
        self.itcp = intercept
        if intercept:
            self.X = np.concatenate([np.ones((n,1)), X], axis=1)
            self.X1 = np.concatenate([np.ones((n,1)), (X - self.mX)/self.sdX], axis=1)
        else:
            self.X, self.X1 = X, X/self.sdX

        self.opt_para = [max_iter, tol]


    def mad(self, x):
        return np.median(abs(x - np.median(x)))*1.4826

    def bandwidth(self, tau):
        n, p = len(self.Y), len(self.mX)
        h0 = min((p + np.log(n))/n, 0.5)**0.4
        return max(0.01, h0*(tau-tau**2)**0.5)

    def boot_weight(self, weight):            
        n = len(self.Y)
        if weight == 'Exponential':
            return rgt.exponential(size=n)
        if weight == 'Rademacher':
            return 2*rgt.binomial(1,1/2,n)
        if weight == 'Multinomial':
            return rgt.multinomial(n, pvals=np.ones(n)/n)
        if weight == 'Gaussian':
            return rgt.normal(1,1,n)
        if weight == 'Uniform':
            return rgt.uniform(0,2,n)
        if weight == 'Folded-normal':
            return abs(rgt.normal(size=n))*np.sqrt(np.pi/2)

    def retire_weight(self, x, tau, c):
        tmp1 = tau*c*(x>c) - (1-tau)*c*(x<-c)
        tmp2 = tau*x*(x>=0)*(x<=c) + (1-tau)*x*(x<0)*(x>=-c)   
        return -(tmp1 + tmp2)/len(x)

    def conquer_weight(self, x, tau, kernel="Laplacian", w=np.array([])):
        if kernel == "Laplacian":
            out = 0.5 + 0.5 * np.sign(x) * (1 - np.exp(-abs(x)))
        if kernel == "Gaussian":
            out = norm.cdf(x)
        if kernel == "Logistic":
            out = 1/(1 + np.exp(-x))
        if kernel == "Uniform":
            out = np.where(x>1,1,0) + np.where(abs(x)<=1, 0.5*(1+x),0)
        if kernel == "Epanechnikov":
            c = np.sqrt(5)
            out = 0.25*(2 + 3*x/c - (x/c)**3 )*(abs(x)<=c) + 1*(x>c)
                        
        if not w.any():
            return (out - tau)/len(x)
        else:
            return w*(out - tau)/len(x)

    def qr_weight(self, x, tau):
        return ((x<=0) - tau)/len(x)

    def retire(self, tau=0.5, tune=2, standardize=True, adjust=False):
        '''
            Robustified/Huberized Expectile Regression
        '''
        if standardize: X = self.X1
        else: X = self.X

        beta0 = np.zeros(X.shape[1])
        if self.itcp: beta0[0] = np.quantile(self.Y, tau)
        res = self.Y - beta0[0]
        c = tune*self.mad(res)
        grad0 = X.T.dot(self.retire_weight(res, tau, c))
        diff_beta = -grad0
        beta1 = beta0 + diff_beta
        res = self.Y - X.dot(beta1)
        
        max_iter, tol = self.opt_para[0], self.opt_para[1]
        r0, count = 1, 0
        while r0 > tol and count <= max_iter:
            c = tune*self.mad(res)
            grad1 = X.T.dot(self.retire_weight(res, tau, c))
            diff_grad = grad1 - grad0
            r0, r1 = diff_beta.dot(diff_beta), diff_grad.dot(diff_grad)
            r01 = diff_grad.dot(diff_beta)
            lr1, lr2 = r01/r1, r0/r01
            grad0, beta0 = grad1, beta1
            diff_beta = -min(lr1, lr2, 10)*grad1
            beta1 += diff_beta
            res = self.Y - X.dot(beta1)
            count += 1

        if standardize and adjust:
            beta1[1*self.itcp:] = beta1[self.itcp:]/self.sdX
            if self.itcp: beta1[0] -= self.mX.dot(beta1[1:])

        return beta1, res, count


    def fit(self, tau=0.5, h=None, kernel="Laplacian", beta0=np.array([]), res=np.array([]), 
            weight=np.array([]), standardize=True, adjust=True):
        '''
            Convolution Smoothed Quantile Regression

        Arguments
        ---------
        tau : quantile level between 0 and 1; default is 0.5.
        
        h : bandwidth/smoothing parameter. The default is computed by self.bandwidth(tau).
        
        kernel : a character string representing one of the built-in smoothing kernels; default is "Laplacian".
                
        beta0 : p+1 dimensional initial estimator; default is np.array([]).
        
        res : n-vector of fitted residuals; default is np.array([]).
        
        weight : n-vector of observation weights; default is np.array([]).
        
        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale.

        Returns
        -------
        beta1 : conquer estimator.
        
        list : a list of residual vector, number of iterations, and bandwidth.
        '''
        if h==None: h=self.bandwidth(tau)
        
        if kernel not in self.kernels:
            raise ValueError("kernel must be either Laplacian, Gaussian, Logistic, Uniform or Epanechnikov")
           
        if not beta0.any():
            beta0, res, count = self.retire(tau=tau, standardize=standardize)
        else: count = 0
        
        if standardize: X = self.X1
        else: X = self.X

        grad0 = X.T.dot(self.conquer_weight(-res/h, tau, kernel, weight))
        diff_beta = -grad0
        beta1 = beta0 + diff_beta
        res = self.Y - X.dot(beta1)
        r0, max_iter, tol = 1, self.opt_para[0], self.opt_para[1]
        while count <= max_iter and r0 > tol:
            grad1 = X.T.dot(self.conquer_weight(-res/h, tau, kernel, weight))
            diff_grad = grad1 - grad0
            r0, r1 = diff_beta.dot(diff_beta), diff_grad.dot(diff_grad)
            if r1 == 0: lr = 1
            else:
                r01 = diff_grad.dot(diff_beta)
                lr1, lr2 = r01/r1, r0/r01
                lr = min(lr1, lr2, 10)

            grad0, beta0 = grad1, beta1
            diff_beta = -lr*grad1
            beta1 += diff_beta
            res = self.Y - X.dot(beta1)
            count += 1
        
        if standardize and adjust:
            beta1[1*self.itcp:] = beta1[self.itcp:]/self.sdX
            if self.itcp: beta1[0] -= self.mX.dot(beta1[1:])

        return beta1, [res, count, h]


    def path(self, tau=0.5, h_seq=np.array([]), L=20, kernel="Laplacian", standardize=True, adjust=True):
        '''
            Solution Path of Conquer
        
        Arguments
        ---------
        h_seq: a sequence of bandwidths.

        L: number of bandwdiths; default is 20.

        '''
        n, p = self.X.shape
        if not np.array(h_seq).any():
            h_seq = np.linspace(0.01, min((p + np.log(n))/n, 0.5)**0.4, num=L)

        if standardize: X = self.X1
        else: X = self.X

        h_seq, L = np.sort(h_seq)[::-1], len(h_seq)
        beta_seq = np.empty(shape=(X.shape[1], L))
        res_seq = np.empty(shape=(n, L))
        beta_seq[:,0], fit = self.fit(tau, h_seq[0], kernel, standardize=standardize, adjust=False)
        res_seq[:,0] = fit[0]
        
        for l in range(1,L):      
            beta_seq[:,l], fit = self.fit(tau, h_seq[l], kernel, beta_seq[:,l-1], fit[0], standardize=standardize, adjust=False)
            res_seq[:,l] = fit[0]
   
        if standardize and adjust:
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp:
                beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])

        return beta_seq, [res_seq, h_seq]
        


    def norm_ci(self, tau=0.5, h=None, kernel="Laplacian", alpha=0.05, standardize=True):
        '''
            Normal Calibrated Confidence Intervals

        Parameters
        ----------
        tau : quantile level; default is 0.5.
        
        h : bandwidth. The default is computed by self.bandwidth(tau).
        
        alpha : 100*(1-alpha)% CI; default is 0.05.

        Returns
        -------
        hat_beta : conquer estimate.
        
        ci : p+1 by 2 (or p by 2) numpy array of normal CIs based on estimated asymptotic covariance matrix.
        '''
        if h == None: h = self.bandwidth(tau)
        n, X = len(self.Y), self.X
        hat_beta, fit = self.fit(tau, h, kernel, standardize=standardize)
        hess_weight = norm.pdf(fit[0]/h)
        grad_weight = ( norm.cdf(-fit[0]/h) - tau)**2
        hat_V = (X.T * grad_weight).dot(X)/n
        inv_J = np.linalg.inv((X.T * hess_weight).dot(X)/(n*h))
        ACov = inv_J.dot(hat_V).dot(inv_J)
        rad = norm.ppf(1-0.5*alpha)*np.sqrt( np.diag(ACov)/n )        
        ci = np.concatenate(((hat_beta - rad)[:,None], (hat_beta + rad)[:,None]), axis=1)
        return hat_beta, ci


    def mb(self, tau=0.5, h=None, kernel="Laplacian", weight="Exponential", standardize=True, B=500):
        '''
            Multiplier Bootstrap Estimates
   
        Parameters
        ----------
        tau : quantile level; default is 0.5.
        
        h : bandwidth. The default is computed by self.bandwidth(tau).

        kernel : a character string representing one of the built-in smoothing kernels; default is "Laplacian".

        weight : a character string representing one of the built-in bootstrap weight distributions; default is "Exponential".

        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.

        B : number of bootstrap replications; default is 500.

        Returns
        -------
        mb_beta : p+1 by B+1 (or p by B+1) numpy array. 1st column: conquer estimator; 2nd to last: bootstrap estimates.
        '''
        if h==None: h = self.bandwidth(tau)
        
        if weight not in self.weights:
            raise ValueError("weight distribution must be either Exponential, Rademacher, Multinomial, Gaussian, Uniform or Folded-normal")
           
        beta0, fit0 = self.fit(tau, h, kernel, standardize=standardize, adjust=False)     
        mb_beta = np.zeros([len(beta0), B+1])
        mb_beta[:,0] = np.copy(beta0)
        if standardize:
            mb_beta[1*self.itcp:,0] = mb_beta[1*self.itcp:,0]/self.sdX
            if self.itcp: mb_beta[0,0] -= self.mX.dot(mb_beta[1:,0])
        
        for b in range(B):
            boot_fit = self.fit(tau, h, kernel, beta0=beta0, res=fit0[0], 
                                weight=self.boot_weight(weight), standardize=standardize)
            mb_beta[:,b+1] = boot_fit[0]

        ## delete NaN bootstrap estimates (when using Gaussian weights)
        mb_beta = mb_beta[:,~np.isnan(mb_beta).any(axis=0)]
        return mb_beta

    
    def mb_ci(self, tau=0.5, h=None, kernel="Laplacian", weight="Exponential", standardize=True, B=500, alpha=0.05):
        '''
            Multiplier Bootstrap Confidence Intervals

        Arguments
        ---------
        tau : quantile level; default is 0.5.

        h : bandwidth. The default is computed by self.bandwidth(tau).

        kernel : a character string representing one of the built-in smoothing kernels; default is "Laplacian".

        weight : a character string representing one of the built-in bootstrap weight distributions; default is "Exponential".

        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.

        B : number of bootstrap replications; default is 500.

        alpha : 100*(1-alpha)% CI; default is 0.05.

        Returns
        -------
        mb_beta : p+1 by B+1 (or p by B+1) numpy array. 1st column: conquer estimator; 2nd to last: bootstrap estimates.
        
        ci : 3 by p+1 by 2 (or 3 by p by 2) numpy array. 1st row: percentile CI; 2nd row: pivotal CI; 3rd row: normal-based CI using bootstrap variance estimate.
        '''
        if h==None: h = self.bandwidth(tau)
        
        mb_beta = self.mb(tau, h, kernel, weight, standardize, B)
        if weight in self.weights[:4]:
            adj = 1
        elif weight == 'Uniform': 
            adj = np.sqrt(1/3)
        elif weight == 'Folded-normal':
            adj = np.sqrt(0.5*np.pi - 1)

        ci = np.empty([3, mb_beta.shape[0], 2])
        ci[0,:,1] = np.quantile(mb_beta[:,1:], 1-0.5*alpha, axis=1)
        ci[0,:,0] = np.quantile(mb_beta[:,1:], 0.5*alpha, axis=1)
        ci[1,:,1] = (1+1/adj)*mb_beta[:,0] - ci[0,:,0]/adj
        ci[1,:,0] = (1+1/adj)*mb_beta[:,0] - ci[0,:,1]/adj
        radi = norm.ppf(1-0.5*alpha)*np.std(mb_beta[:,1:], axis=1)/adj
        ci[2,:,0], ci[2,:,1] = mb_beta[:,0] - radi, mb_beta[:,0] + radi
        return mb_beta, ci


    def rq(self, tau=0.5, lr=1, beta0=np.array([]), res=np.array([]), standardize=True, adjust=True, max_iter=5e3):
        '''
            Quantile Regression via Subgradient Descent and Conquer Initialization
        
        Arguments
        ---------
        lr : learning rate (step size); default is 1.

        '''
        if not beta0.any():
            beta0, fit0 = self.fit(tau=tau, standardize=standardize, adjust=False)

        if standardize: X = self.X1
        else: X = self.X

        beta1, res = np.copy(beta0), fit0[0]
        dev, count = 1, 0
        while count <= max_iter and dev > self.opt_para[1]:
            diff = lr*X.T.dot(self.qr_weight(res, tau))
            beta1 -= diff
            dev = diff.dot(diff)
            res = self.Y - X.dot(beta1)
            count += 1

        if standardize and adjust:
            beta1[self.itcp:] = beta1[self.itcp:]/self.sdX
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            if self.itcp: 
                beta1[0] -= self.mX.dot(beta1[1:])
                beta0[0] -= self.mX.dot(beta0[1:])

        return beta0, beta1, [fit0[0], res, count]
        
