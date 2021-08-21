import numpy as np
import numpy.random as rgt
from scipy.stats import norm


class low_dim():
    '''
        Convolution Smoothed Quantile Regression
    '''
    kernels = ["Laplacian", "Gaussian", "Logistic", "Uniform", "Epanechnikov"]
    weights = ["Exponential", "Multinomial", "Rademacher", "Gaussian", "Uniform", "Folded-normal"]

    def __init__(self, X, Y, intercept=True, max_iter=500, max_lr=10, tol=1e-5):
        '''
        Arguments
        ---------
        X : n by p matrix of covariates; each row is an observation vector.
           
        Y : n-dimensional vector of response variables.
            
        intercept : logical flag for adding an intercept to the model.

        Internal Optimization Parameters
        --------------------------------
        max_iter : maximum numder of iterations in the GD-BB algorithm; default is 500.

        max_lr : maximum step size/learning rate.
        
        tol : minimum change in (squared) Euclidean distance for stopping GD iterations; default is 1e-5.
        '''
        self.Y, self.n = Y, len(Y)
        if X.shape[1] >= self.n: raise ValueError("covariate dimension exceeds sample size")
        self.mX, self.sdX = np.mean(X, axis=0), np.std(X, axis=0)
        self.itcp = intercept
        if intercept:
            self.X = np.concatenate([np.ones((self.n, 1)), X], axis=1)
            self.X1 = np.concatenate([np.ones((self.n, 1)), (X - self.mX)/self.sdX], axis=1)
        else:
            self.X, self.X1 = X, X/self.sdX

        self.opt_para = [max_iter, max_lr, tol]


    def mad(self, x):
        return np.median(abs(x - np.median(x)))*1.4826

    def bandwidth(self, tau):
        h0 = min((len(self.mX) + np.log(self.n))/self.n, 0.5)**0.4
        return max(0.01, h0*(tau-tau**2)**0.5)

    def smooth_check(self, x, tau=0.5, h=None, kernel='Laplacian', w=np.array([])):
        if h == None: h = self.bandwidth(tau)       
        u = x/h     
        if kernel == "Gaussian":
            out = 0.5*h*np.sqrt(2/np.pi)*np.exp(-u**2/2) + x*(0.5-norm.cdf(-u))
        if kernel == "Logistic":
            out = 0.5*h*(u + 2*np.log(1 + np.exp(-u)))
        if kernel == "Uniform":
            out = 0.5*h*( (0.5* u**2 + 0.5)*(abs(u)<=1) + abs(u)*(abs(u)>1))
        if kernel == "Laplacian":
            out = 0.5*h*(abs(u) + np.exp(-abs(u)))
        if kernel == "Epanechnikov":
            out = 0.5*h*((0.75*u**2-u**4/8+3/8)*(abs(u)<=1)+abs(u)*(abs(u)>1))

        out += (tau-0.5)*x
        if not w.any(): 
            return np.mean(out)
        else:
            return np.mean(w*out)

    def boot_weight(self, weight):            
        if weight == 'Exponential':
            return rgt.exponential(size=self.n)
        if weight == 'Rademacher':
            return 2*rgt.binomial(1, 1/2, self.n)
        if weight == 'Multinomial':
            return rgt.multinomial(self.n, pvals=np.ones(self.n)/self.n)
        if weight == 'Gaussian':
            return rgt.normal(1, 1, self.n)
        if weight == 'Uniform':
            return rgt.uniform(0, 2, self.n)
        if weight == 'Folded-normal':
            return abs(rgt.normal(size=self.n))*np.sqrt(np.pi/2)

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
        
        max_iter, tol = self.opt_para[0], self.opt_para[2]
        r0, count = 1, 0
        while r0 > tol and count <= max_iter:
            c = tune*self.mad(res)
            grad1 = X.T.dot(self.retire_weight(res, tau, c))
            diff_grad = grad1 - grad0
            r0, r1 = diff_beta.dot(diff_beta), diff_grad.dot(diff_grad)
            r01 = diff_grad.dot(diff_beta)
            lr1, lr2 = r01/r1, r0/r01
            grad0, beta0 = grad1, beta1
            diff_beta = -min(lr1, lr2, self.opt_para[1])*grad1
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
        r0, max_iter, tol = 1, self.opt_para[0], self.opt_para[2]
        while count <= max_iter and r0 > tol:
            grad1 = X.T.dot(self.conquer_weight(-res/h, tau, kernel, weight))
            diff_grad = grad1 - grad0
            r0, r1 = diff_beta.dot(diff_beta), diff_grad.dot(diff_grad)
            if r1 == 0: lr = 1
            else:
                r01 = diff_grad.dot(diff_beta)
                lr1, lr2 = r01/r1, r0/r01
                lr = min(lr1, lr2, self.opt_para[1])

            grad0, beta0 = grad1, beta1
            diff_beta = -lr*grad1
            beta1 += diff_beta
            res = self.Y - X.dot(beta1)
            count += 1
        
        if standardize and adjust:
            beta1[1*self.itcp:] = beta1[self.itcp:]/self.sdX
            if self.itcp: beta1[0] -= self.mX.dot(beta1[1:])

        return beta1, [res, count, h]


    def bw_path(self, tau=0.5, h_seq=np.array([]), L=20, kernel="Laplacian", standardize=True, adjust=True):
        '''
            Solution Path of Conquer at a Sequence of Bandwidths
        
        Arguments
        ---------
        h_seq: a sequence of bandwidths.

        L: number of bandwdiths; default is 20.

        '''
        if not np.array(h_seq).any():
            h_seq = np.linspace(0.01, min((len(self.mX) + np.log(self.n))/self.n, 0.5)**0.4, num=L)

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
        X = self.X
        hat_beta, fit = self.fit(tau, h, kernel, standardize=standardize)
        hess_weight = norm.pdf(fit[0]/h)
        grad_weight = ( norm.cdf(-fit[0]/h) - tau)**2
        hat_V = (X.T * grad_weight).dot(X)/self.n
        inv_J = np.linalg.inv((X.T * hess_weight).dot(X)/(self.n * h))
        ACov = inv_J.dot(hat_V).dot(inv_J)
        rad = norm.ppf(1-0.5*alpha)*np.sqrt( np.diag(ACov) / self.n )        
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


    def qr(self, tau=0.5, lr=1, beta0=np.array([]), res=np.array([]), standardize=True, adjust=True, max_iter=1e3):
        '''
            Quantile Regression via Subgradient Descent and Conquer Initialization
        
        Arguments
        ---------
        tau : quantile level; default is 0.5.

        lr : learning rate (step size); default is 1.

        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale.

        max_iter : maximum number of (subgradient descent) iterations; default is 1000.

        '''
        if not beta0.any():
            beta0, fit0 = self.fit(tau=tau, standardize=standardize, adjust=False)
            res = fit0[0]

        if standardize: X = self.X1
        else: X = self.X

        dev, count = 1, 0
        while count <= max_iter and dev > self.opt_para[2]:
            diff = lr*X.T.dot(self.qr_weight(res, tau))
            beta0 -= diff
            dev = diff.dot(diff)
            res = self.Y - X.dot(beta0)
            count += 1

        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            if self.itcp: 
                beta0[0] -= self.mX.dot(beta0[1:])

        return beta0, [res, count]
        



class high_dim(low_dim):
    '''
        Regularized Convolution Smoothed Quantile Regression via ILAMM
                        (iterative local adaptive majorize-minimization)
    '''
    weights = ['Exponential', 'Rademacher', 'Multinomial']

    def __init__(self, X, Y, intercept=True, phi=0.1, gamma=1.25, max_iter=500, tol=1e-5):

        '''
        Arguments
        ---------
        X : n by p matrix of covariates; each row is an observation vector.
           
        Y : n-dimensional vector of response variables.
            
        intercept : logical flag for adding an intercept to the model.
        
        Internal Optimization Parameters
        --------------------------------
        phi : initial quadratic coefficient parameter in the ILAMM algorithm; default is 0.1.
        
        gamma : adaptive search parameter that is larger than 1; default is 1.25.
        
        max_iter : maximum numder of iterations in the ILAMM algorithm; default is 500.
        
        tol : minimum change in (squared) Euclidean distance for stopping LAMM iterations; default is 1e-5.

        '''
        self.n, self.p = X.shape
        self.Xinit, self.Y = X, Y
        self.mX, self.sdX = np.mean(X, axis=0), np.std(X, axis=0)
        self.itcp = intercept
        if intercept:
            self.X = np.concatenate([np.ones((self.n,1)), X], axis=1)
            self.X1 = np.concatenate([np.ones((self.n,1)), (X - self.mX)/self.sdX], axis=1)
        else:
            self.X, self.X1 = X, X/self.sdX

        self.opt_para = [phi, gamma, max_iter, tol]

    def bandwidth(self, tau):
        h0 = (np.log(self.p)/self.n)**0.25
        return max(0.05, h0*(tau-tau**2)**0.5)

    def soft_thresh(self, x, c):
        tmp = abs(x) - c
        return np.sign(x)*np.where(tmp<=0, 0, tmp)
    
    def self_tuning(self, tau=0.5, standardize=True, B=200):
        '''
            A Simulation-based Approach for Choosing the Penalty Level (Lambda)
        
        Reference
        ---------
        l1-Penalized Quantile Regression in High-dimensinoal Sparse Models (2011)
        by Alexandre Belloni and Victor Chernozhukov
        The Annals of Statistics 39(1): 82--130.

        Arguments
        ---------
        tau : quantile level; default is 0.5.
        
        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.
        
        B : number of simulations; default is 200.

        Returns
        -------
        lambda_sim : a numpy array (with length B) of simulated lambda values.
        '''    
        if standardize: X = self.X1 
        else: X = self.X
        lambda_sim = np.array([max(abs(X.T.dot(tau - (rgt.uniform(0,1,self.n) <= tau))/self.n)) for b in range(B)])
        return lambda_sim
    

    def concave_weight(self, x, penalty="SCAD", a=None):
        if penalty == "SCAD":
            if a==None: a = 3.7
            tmp = 1 - (abs(x)-1)/(a-1)
            tmp = np.where(tmp<=0,0,tmp)
            return np.where(tmp>1, 1, tmp)
        if penalty == "MCP":
            if a==None: a = 3
            tmp = 1 - abs(x)/a 
            return np.where(tmp<=0, 0, tmp)
        if penalty == "CapppedL1":
            if a==None: a = 3
            return 1*(abs(x) <= a/2)
    

    def retire_loss(self, x, tau, c):
        out = 0.5*(abs(x)<=c)* x**2 + (c*abs(x)-0.5*c**2)*(abs(x)>c)
        return np.mean( abs(tau - (x<0))*out )
    

    def l1_retire(self, Lambda=np.array([]), tau=0.5, tune=3, beta0=np.array([]), res=np.array([]), 
                    standardize=True, adjust=True): 
        '''
            L1-Penalized Robustified Expectile Regression (l1-retire)
        ''' 
        if not np.array(Lambda).any(): 
            Lambda = np.quantile(self.self_tuning(tau, standardize), 0.9)

        if standardize: X = self.X1
        else: X = self.X
        
        if not beta0.any():
            beta0 = np.zeros(X.shape[1])
            if self.itcp: beta0[0] = np.quantile(self.Y, tau)
            res = self.Y - beta0[0]

        phi, gamma, max_iter, tol = self.opt_para[0], self.opt_para[1], self.opt_para[2], self.opt_para[3]
        phi0, r0, count = phi, 1, 0
        while r0 > tol and count <= max_iter:
            c = tune*self.mad(res)
            grad0 = X.T.dot(self.retire_weight(res, tau, c))
            loss_eval0 = self.retire_loss(res, tau, c)
            beta1 = beta0 - grad0/phi0
            beta1[self.itcp:] = self.soft_thresh(beta1[self.itcp:], Lambda/phi0)
            diff_beta = beta1 - beta0
            r0 = diff_beta.dot(diff_beta)
            
            res = self.Y - X.dot(beta1)
            loss_proxy = loss_eval0 + diff_beta.dot(grad0) + 0.5*phi0*r0
            loss_eval1 = self.retire_loss(res, tau, c)
            
            while loss_proxy < loss_eval1:
                phi0 *= gamma
                beta1 = beta0 - grad0/phi0
                beta1[self.itcp:] = self.soft_thresh(beta1[self.itcp:], Lambda/phi0)
                diff_beta = beta1 - beta0
                r0 = diff_beta.dot(diff_beta)
                res = self.Y - X.dot(beta1)
                loss_proxy = loss_eval0 + diff_beta.dot(grad0) + 0.5*phi0*r0
                loss_eval1 = self.retire_loss(res, tau, c)
                
            beta0, phi0 = beta1, phi
            count += 1

        if standardize and adjust:
            beta1[self.itcp:] = beta1[self.itcp:]/self.sdX
            if self.itcp: beta1[0] -= self.mX.dot(beta1[1:])

        return beta1, [res, count, Lambda]
        

    def l1(self, Lambda=np.array([]), tau=0.5, h=None, kernel="Laplacian", beta0=np.array([]), res=np.array([]), 
            standardize=True, adjust=True, weight=np.array([])):
        '''
            L1-Penalized Convolution Smoothed Quantile Regression (l1-conquer)
        
        Arguments
        ---------
        Lambda : regularization parameter. This should be either a scalar, or 
                 a vector of length equal to the column dimension of X. If unspecified, 
                 it will be computed by self.self_tuning().

        tau : quantile level; default is 0.5.

        h : bandwidth/smoothing parameter. The default is computed by self.bandwidth().
        
        kernel : a character string representing one of the built-in smoothing kernels; default is "Laplacian".

        beta0 : initial estimator. If unspecified, it will be set as zero.

        res : residual vector of the initial estiamtor.
        
        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale.            
        
        weight : n-vector of observation weights; default is np.array([]) (empty).

        Returns
        -------
        beta1 : a numpy array of estimated coefficients.
        
        list : a list of residual vector, number of iterations and lambda value.

        '''
        if not np.array(Lambda).any():
            Lambda = 0.7*np.quantile(self.self_tuning(tau,standardize), 0.9)
        if h == None: h = self.bandwidth(tau)
        
        if not beta0.any():
            beta0, retire_fit = self.l1_retire(Lambda, tau, standardize=standardize, adjust=False)
            res = retire_fit[0]            
                    
        if standardize: X = self.X1
        else: X = self.X
        
        phi, gamma, max_iter, tol = self.opt_para[0], self.opt_para[1], self.opt_para[2], self.opt_para[3]
        phi0, r0, count = phi, 1, 0
        while r0 > tol and count <= max_iter:
            
            grad0 = X.T.dot(self.conquer_weight(-res/h, tau, kernel, weight))
            loss_eval0 = self.smooth_check(res, tau, h, kernel, weight)
            beta1 = beta0 - grad0/phi0
            beta1[self.itcp:] = self.soft_thresh(beta1[self.itcp:], Lambda/phi0)
            diff_beta = beta1 - beta0
            r0 = diff_beta.dot(diff_beta)
            
            res = self.Y - X.dot(beta1)
            loss_proxy = loss_eval0 + diff_beta.dot(grad0) + 0.5*phi0*r0
            loss_eval1 = self.smooth_check(res, tau, h, kernel, weight)
            
            while loss_proxy < loss_eval1:
                phi0 *= gamma
                beta1 = beta0 - grad0/phi0
                beta1[self.itcp:] = self.soft_thresh(beta1[self.itcp:], Lambda/phi0)
                diff_beta = beta1 - beta0
                r0 = diff_beta.dot(diff_beta)
                res = self.Y - X.dot(beta1)
                loss_proxy = loss_eval0 + diff_beta.dot(grad0) + 0.5*phi0*r0
                loss_eval1 = self.smooth_check(res, tau, h, kernel, weight)
                
            beta0, phi0 = beta1, phi
            count += 1

        if standardize and adjust:
            beta1[self.itcp:] = beta1[self.itcp:]/self.sdX
            if self.itcp: beta1[0] -= self.mX.dot(beta1[1:])
            
        return beta1, [res, count, Lambda]
 

    def irw(self, Lambda=None, tau=0.5, h=None, kernel="Laplacian", beta0=np.array([]), res=np.array([]),
            penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True, weight=np.array([]), tol=1e-5):
        '''
            Iteratively Reweighted L1-Penalized Conquer (irw-l1-conquer)
            
        Arguments
        ----------
        penalty : a character string representing one of the built-in concave penalties; default is "SCAD".
        
        a : the constant (>2) in the concave penality; default is 3.7.
        
        nstep : the number of iterations/steps of the IRW algorithm; default is 5.

        Returns
        -------
        beta1 : a numpy array of estimated coefficients.
        
        list : a list of residual vector, number of iterations and lambda value.
        '''
        if Lambda == None: 
            Lambda = 0.7*np.quantile(self.self_tuning(tau,standardize), 0.9)
        if h == None: h = self.bandwidth(tau)
        
        if not beta0.any():
            beta0, fit = self.l1(Lambda, tau, h, kernel, standardize=standardize, adjust=False, weight=weight)
        else:
            beta0, fit = self.l1(Lambda, tau, h, kernel, beta0, res, standardize, adjust=False, weight=weight)
        res = fit[0]

        err, count = 1, 1
        while err > tol and count <= nstep:
            rw_lambda = Lambda * self.concave_weight(beta0[self.itcp:]/Lambda, penalty, a)
            beta1, fit = self.l1(rw_lambda, tau, h, kernel, beta0, res, standardize, adjust=False, weight=weight)
            err = max(abs(beta1-beta0))
            beta0, res = beta1, fit[0]
            count += 1
        
        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            if self.itcp: beta0[0] -= self.mX.dot(beta0[1:])
            
        return beta0, [res, count, Lambda]
    
     
    def irw_retire(self, Lambda=None, tau=0.5, tune=3, penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True, tol=1e-5):
        '''
            Iteratively Reweighted L1-Penalized Retire (irw-l1-retire)
        '''
        if Lambda == None: 
            Lambda = np.quantile(self.self_tuning(tau,standardize), 0.9)
        
        beta0, fit0 = self.l1_retire(Lambda, tau, tune, standardize=standardize, adjust=False)
        res = fit0[0]
        err, count = 1, 1
        while err > tol and count <= nstep:
            rw_lambda = Lambda * self.concave_weight(beta0[self.itcp:]/Lambda, penalty, a)
            beta1, fit1 = self.l1_retire(rw_lambda, tau, tune, beta0, res, standardize, adjust=False)
            err = max(abs(beta1-beta0))
            beta0, res = beta1, fit1[0]
            count += 1
        
        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            if self.itcp: beta0[0] -= self.mX.dot(beta0[1:])
            
        return beta0, [res, count, Lambda]
    
    
    def l1_path(self, lambda_seq, tau=0.5, h=None, kernel="Laplacian", standardize=True, adjust=True):
        '''
            Solution Path of L1-Penalized Conquer

        Arguments
        ---------
        lambda_seq : a numpy array of lambda values.

        Returns
        -------
        beta_seq : a sequence of l1-conquer estimates. Each column corresponds to an estiamte for a lambda value.

        list : a list of redisual sequence, a sequence of model sizes, a sequence of lambda values in descending order, and bandwidth.
        '''
        if h == None: h = self.bandwidth(tau)
        lambda_seq = np.sort(lambda_seq)[::-1]
        beta_seq = np.empty(shape=(self.X.shape[1], len(lambda_seq)))
        res_seq = np.empty(shape=(self.n, len(lambda_seq)))
        beta_seq[:,0], fit = self.l1(lambda_seq[0], tau, h, kernel, standardize=standardize, adjust=False)
        res_seq[:,0] = fit[0]
        
        for l in range(1,len(lambda_seq)):
            beta_seq[:,l], fit = self.l1(lambda_seq[l], tau, h, kernel, beta_seq[:,l-1], fit[0], standardize, adjust=False)
            res_seq[:,l] = fit[0]

        if standardize and adjust:
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])

        return beta_seq, [res_seq, np.sum((abs(beta_seq[self.itcp:,:])>0), axis=0), lambda_seq, h]
    
    
    def irw_path(self, lambda_seq, tau=0.5, h=None, kernel="Laplacian", penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True):
        '''
            Solution Path of Iteratively Reweighted L1-Conquer

        Arguments
        ---------
        lambda_seq : a numpy array of lambda values.

        tau : quantile level; default is 0.5.

        h : smoothing parameter/bandwidth. The default is computed by self.bandwidth().
        
        kernel : a character string representing one of the built-in smoothing kernels; default is "Laplacian".
        
        penalty : a character string representing one of the built-in concave penalties; default is "SCAD".
        
        a : the constant (>2) in the concave penality; default is 3.7.
        
        nstep : the number of iterations/steps of the IRW algorithm; default is 5.
        
        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale; default is TRUE.

        Returns
        -------
        beta_seq : a sequence of irw-l1-conquer estimates. Each column corresponds to an estimate for a lambda value.

        list : a list of redisual sequence, a sequence of model sizes, a sequence of lambda values in descending order, and bandwidth.
        '''
        if h == None: h = self.bandwidth(tau)
        lambda_seq, nlambda = np.sort(lambda_seq)[::-1], len(lambda_seq)
        beta_seq = np.empty(shape=(self.X.shape[1], nlambda))
        res_seq = np.empty(shape=(self.n, nlambda))
        beta_seq[:,0], fit = self.irw(lambda_seq[0], tau, h, kernel, penalty=penalty, a=a, nstep=nstep, standardize=standardize, adjust=False)
        res_seq[:,0] = fit[0]
        for l in range(1, nlambda):
            beta_seq[:,l], fit = self.irw(lambda_seq[l], tau, h, kernel, beta_seq[:,l-1], fit[0], penalty, a, nstep, standardize, adjust=False)
            res_seq[:,l] = fit[0]
        
        if standardize and adjust:
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])
    
        return beta_seq, [res_seq, np.sum((abs(beta_seq[self.itcp:,:])>0), axis=0), lambda_seq, h]



    def boot_select(self, Lambda=None, tau=0.5, h=None, kernel="Laplacian", weight="Multinomial",
                    B=200, alpha=0.05, penalty="SCAD", a=3.7, nstep=5, standardize=True, parallel=False, ncore=None):
        '''
            Model Selection via Bootstrap 

        Arguments
        ---------
        Lambda : scalar regularization parameter. If unspecified, it will be computed by self.self_tuning().
        
        tau : quantile level; default is 0.5.

        h : smoothing parameter/bandwidth. The default is computed by self.bandwidth().
        
        kernel : a character string representing one of the built-in smoothing kernels; default is "Laplacian".

        weight : a character string representing one of the built-in bootstrap weight distributions; default is "Multinomial".
        
        B : number of bootstrap replications; default is 200.
        
        alpha : 100*(1-alpha)% CI; default is 0.05.
        
        penalty : a character string representing one of the built-in concave penalties; default is "SCAD".
        
        a : the constant (>2) in the concave penality; default is 3.7.
        
        nstep : the number of iterations/steps of the IRW algorithm; default is 5.
        
        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.

        parallel : logical flag to implement bootstrap using parallel computing; default is FALSE.

        ncore : the number of cores used for parallel computing.

        Returns
        -------
        mb_beta : p+1/p by B+1 numpy array. 1st column: penalized conquer estimator; 2nd to last: bootstrap estimates.
            
        list : a list of two selected models.
        '''
        if Lambda == None:
            Lambda = np.quantile(self.self_tuning(tau, standardize), 0.9)
        if h == None: h = self.bandwidth(tau) 
        if weight not in self.weights[:3]:
            raise ValueError("weight distribution must be either Exponential, Rademacher or Multinomial")

        fit0 = self.irw(Lambda, tau, h, kernel, penalty=penalty, a=a, nstep=nstep, standardize=standardize, adjust=False)
        mb_beta = np.zeros(shape=(self.p+self.itcp, B+1))
        mb_beta[:,0] = fit0[0]
        if standardize:
            mb_beta[self.itcp:,0] = mb_beta[self.itcp:,0]/self.sdX
            if self.itcp: mb_beta[0,0] -= self.mX.dot(mb_beta[1:,0])

        if parallel:
            import multiprocessing
            max_ncore = multiprocessing.cpu_count()
            if ncore == None: ncore = max_ncore
            if ncore > max_ncore: raise ValueError("number of cores exceeds the limit")

        def bootstrap(b):
            boot_fit = self.irw(Lambda, tau, h, kernel, beta0=fit0[0], res=fit0[1][0], penalty=penalty, a=a, nstep=nstep,
                                standardize=standardize, weight=self.boot_weight(weight))
            return boot_fit[0]

        if not parallel:
            for b in range(B): mb_beta[:,b+1] = bootstrap(b)
        else:
            from joblib import Parallel, delayed
            num_cores = multiprocessing.cpu_count()
            boot_results = Parallel(n_jobs=ncore)(delayed(bootstrap)(b) for b in range(B))
            mb_beta[:,1:] = np.array(boot_results).T
        
        ## delete NaN bootstrap estimates (when using Gaussian weights)
        mb_beta = mb_beta[:,~np.isnan(mb_beta).any(axis=0)]
        
        ## Method 1: Majority vote among B bootstrap models
        selection_rate = 1 - np.mean(mb_beta[self.itcp:,1:]==0, axis=1)
        model_1 = np.where(selection_rate>=0.5)[0]
        
        ## Method 2: Intersection of B bootstrap models
        model_2 = np.arange(self.p)
        for b in range(len(mb_beta[0,1:])):
            boot_model = np.where(abs(mb_beta[self.itcp:,b+1])>0)[0]
            model_2 = np.intersect1d(model_2, boot_model)

        return mb_beta, [model_1, model_2]



    def boot_inference(self, Lambda=None, tau=0.5, h=None, kernel="Laplacian", weight="Multinomial",
                        B=200, alpha=0.05, penalty="SCAD", a=3.7, nstep=5, standardize=True, parallel=False, ncore=1):
        '''
            Post-Selection-Inference via Bootstrap

        Arguments
        ---------
        see boot_select().

        Returns
        -------
        see boot_select().

        mb_ci : 3 by p/p+1 by 2 numpy array of confidence intervals.

        mb_ci[0,:,:] : percentile CI; 

        mb_ci[1,:,:]: pivotal CI; 

        mb_ci[2,:,:]: normal-based CI using bootstrap variance estimate.      
        '''
        mb_beta, mb_model = self.boot_select(Lambda, tau, h, kernel, weight, B, alpha, penalty, a, nstep, standardize, parallel)
        
        mb_ci = np.zeros([3, self.p + self.itcp, 2])
        X_select = self.X[:, mb_model[0]+self.itcp]
        post_sqr = low_dim(X_select, self.Y, self.itcp)
        post_boot = post_sqr.mb_ci(tau, kernel=kernel, weight=weight, alpha=alpha, standardize=standardize)
        mb_ci[:,mb_model[0]+self.itcp,:] = post_boot[1][:,self.itcp:,:]
        if self.itcp: mb_ci[:,0,:] = post_boot[1][:,0,:]

        return mb_beta, [mb_ci, mb_model[0], mb_model[1]]
    


class cv_lambda():
    '''
        Cross-Validated Penalized Conquer 
    '''
    penalties = ["L1", "SCAD", "MCP"]

    def __init__(self, X, Y, intercept=True, B=200, phi=0.1, gamma=1.5, max_iter=500, tol=1e-5):
        self.n, self.p = X.shape
        self.X, self.Y = X, Y
        self.itcp = intercept
        self.B = B
        self.phi, self.gamma, self.max_iter, self.tol = phi, gamma, max_iter, tol
        
    def check(self, x, tau):
        '''
            Empirical Quantile Loss (check function)
        '''
        return np.mean(0.5*abs(x)+(tau-0.5)*x)

    def divide_sample(self, nfolds=5):
        '''
            Divide the Sample into V=nfolds Folds
        '''
        idx, folds = np.arange(self.n), []
        for v in range(nfolds):
            folds.append(idx[v::nfolds])
        return idx, folds

    def fit(self, tau=0.5, h=None, lambda_seq=np.array([]), nlambda=40, nfolds=5,
            kernel="Laplacian", penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True):

        sqr_fit = high_dim(self.X, self.Y, self.itcp, self.phi, self.gamma, self.max_iter, self.tol)
        if h == None: h = sqr_fit.bandwidth(tau)

        if not lambda_seq.any():
            lambda_max = np.max(sqr_fit.self_tuning(tau, standardize, self.B))
            lambda_seq = np.linspace(0.25*lambda_max, 1.5*lambda_max, nlambda)
        else:
            nlambda = len(lambda_seq)

        if penalty not in self.penalties: raise ValueError("penalty must be either L1, SCAD or MCP")

        idx, folds = self.divide_sample(nfolds)
        val_error = np.zeros((nfolds, nlambda))
        for v in range(nfolds):
            X_train, Y_train = self.X[np.setdiff1d(idx,folds[v]),:], self.Y[np.setdiff1d(idx,folds[v])]
            X_val, Y_val = self.X[folds[v],:], self.Y[folds[v]]
            sqr_train = high_dim(X_train, Y_train, self.itcp, self.phi, self.gamma, self.max_iter, self.tol)

            if penalty == "L1":
                train_beta, train_fit = sqr_train.l1_path(lambda_seq, tau, h, kernel, standardize, adjust)
            else:
                train_beta, train_fit = sqr_train.irw_path(lambda_seq, tau, h, kernel, penalty, a, nstep, standardize, adjust)
                       
            for l in range(nlambda):
                val_error[v,l] = self.check(Y_val - train_beta[0,l]*self.itcp - X_val.dot(train_beta[self.itcp:,l]), tau)
        
        cv_error = np.mean(val_error, axis=0)
        cv_min = min(cv_error)
        l_min = np.where(cv_error == cv_min)[0][0]
        lambda_seq, lambda_min = train_fit[2], train_fit[2][l_min]
        if penalty == "L1":
            cv_beta, cv_fit = sqr_fit.l1(lambda_min, tau, h, kernel, standardize=standardize, adjust=adjust)
        else:
            cv_beta, cv_fit = sqr_fit.irw(lambda_min, tau, h, kernel, penalty=penalty, a=a, nstep=nstep, standardize=standardize, adjust=adjust)

        return cv_beta, [cv_fit[0], lambda_min, lambda_seq, cv_min, cv_error]



class validate_lambda(cv_lambda):
    '''
        Train Penalized Conquer on a Validation Set
    '''
    penalties = ["L1", "SCAD", "MCP"]
    
    def __init__(self, X_train, Y_train, X_val, Y_val, intercept=True, B=200):
        self.n, self.p = X_train.shape
        self.X_train, self.Y_train = X_train, Y_train
        self.X_val, self.Y_val = X_val, Y_val 
        self.itcp = intercept
        self.B = B 

   
    def train(self, tau=0.5, h=None, lambda_seq=np.array([]), nlambda=20, kernel="Laplacian", penalty="SCAD", a=3.7, nstep=5, standardize=True):
        '''
        Arguments
        ---------
        lambda_seq : a numpy array of lambda values. If unspecified, it will be determined by the self_tuning() function in high_dim.

        nlambda : number of lambda values if unspecified; default is 20.

        penalty : a character string representing one of the built-in penalties; default is "SCAD".

        Returns
        -------
        val_beta : a numpy array of regression estimates. 
        
        val_res : a numpy array of fitted residuals.

        model_size : a sequence of selected model sizes.

        lambda_min : the value of lambda that gives minimum validation error.

        lambda_seq : a sequence of lambdas in descending order. 

        val_min : minimum validation error.

        val_seq : a sequence of validation errors.
        '''    
        sqr_train = high_dim(self.X_train, self.Y_train, intercept=self.itcp)
        if not lambda_seq.any():
            lambda_max = np.max(sqr_train.self_tuning(tau, standardize, self.B))
            lambda_seq = np.linspace(0.25*lambda_max, lambda_max, nlambda)
        else:
            nlambda = len(lambda_seq)
            
        if h == None: h = sqr_train.bandwidth(tau)

        if penalty not in self.penalties:
            raise ValueError("penalty must be either L1, SCAD or MCP")
        elif penalty == "L1":
            train_beta, train_fit = sqr_train.l1_path(lambda_seq, tau, h, kernel, standardize)
        else:
            train_beta, train_fit = sqr_train.irw_path(lambda_seq, tau, h, kernel, penalty, a, nstep, standardize)
        val_error = np.zeros(nlambda)
        for l in range(nlambda):
            val_error[l] = self.check(self.Y_val - train_beta[0,l]*self.itcp - self.X_val.dot(train_beta[self.itcp:,l]), tau)

        val_min = min(val_error)
        l_min = np.where(val_error == val_min)[0][0]
        val_beta, val_res = train_beta[:,l_min], train_fit[0][:,l_min]
        model_size = sum((abs(val_beta[self.itcp:])>0))
        lambda_seq, lambda_min = train_fit[2], train_fit[2][l_min]

        return val_beta, [val_res, model_size, lambda_min, lambda_seq, val_min, val_error]
