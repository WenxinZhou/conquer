import numpy as np
import numpy.random as rgt
from scipy.stats import norm


class low_dim():
    '''
        Convolution Smoothed Quantile Regression
    '''
    kernels = ["Laplacian", "Gaussian", "Logistic", "Uniform", "Epanechnikov"]
    weights = ["Exponential", "Multinomial", "Rademacher", "Gaussian", "Uniform", "Folded-normal"]
    opt = {'max_iter': 1e3, 'max_lr': 10, 'tol': 1e-4, 'nboot': 200}

    def __init__(self, X, Y, intercept=True, options=dict()):
        '''
        Arguments
        ---------
        X : n by p matrix of covariates; each row is an observation vector.
           
        Y : n-dimensional vector of response variables.
            
        intercept : logical flag for adding an intercept to the model.

        options : a dictionary of internal statistical and optimization parameters.

            max_iter : maximum numder of iterations in the GD-BB algorithm; default is 500.

            max_lr : maximum step size/learning rate.
            
            tol : the iteration will stop when max{|g_j|: j = 1, ..., p} <= tol 
                  where g_j is the j-th component of the (smoothed) gradient; default is 1e-4.

            nboot : number of bootstrap samples for inference.
        '''
        self.n = len(Y)
        self.Y = Y.reshape(self.n)
        if X.shape[1] >= self.n: raise ValueError("covariate dimension exceeds sample size")
        self.mX, self.sdX = np.mean(X, axis=0), np.std(X, axis=0)
        self.itcp = intercept
        if intercept:
            self.X = np.c_[np.ones(self.n), X]
            self.X1 = np.c_[np.ones(self.n,), (X - self.mX)/self.sdX]
        else:
            self.X, self.X1 = X, X/self.sdX

        self.opt.update(options)

    def mad(self, x):
        return np.median(abs(x - np.median(x)))*1.4826

    def bandwidth(self, tau):
        h0 = min((len(self.mX) + np.log(self.n))/self.n, 0.5)**0.4
        return max(0.01, h0*(tau-tau**2)**0.5)

    def smooth_check(self, x, tau=0.5, h=None, kernel='Laplacian', w=np.array([])):
        if h == None: h = self.bandwidth(tau)

        loss1 = lambda x : np.where(x >= 0, tau*x, (tau-1)*x) + 0.5 * h * np.exp(-abs(x)/h)
        loss2 = lambda x : (tau - norm.cdf(-x/h)) * x + 0.5 * h * np.sqrt(2 / np.pi) * np.exp(-(x/h) ** 2 / 2)
        loss3 = lambda x : tau * x + h * np.log(1 + np.exp(-x/h))
        loss4 = lambda x : (tau - 0.5) * x + h * (0.25 * (x/h)**2 + 0.25) * (abs(x) < h) \
                            + 0.5 * abs(x) * (abs(x) >= h)
        loss5 = lambda x : (tau - 0.5) * x + 0.5 * h * (0.75 * (x/h) ** 2 \
                            - (x/h) ** 4 / 8 + 3 / 8) * (abs(x) < h) + 0.5 * abs(x) * (abs(x) >= h)
        loss_dict = {'Laplacian': loss1, 'Gaussian': loss2, 'Logistic': loss3, \
                     'Uniform': loss4, 'Epanechnikov': loss5}
        if not w.any(): 
            return np.mean(loss_dict[kernel](x))
        else:
            return np.mean(loss_dict[kernel](x) * w)

    def boot_weight(self, weight):
        w1 = lambda n : rgt.multinomial(n, pvals=np.ones(n)/n)
        w2 = lambda n : rgt.exponential(size=n)
        w3 = lambda n : 2*rgt.binomial(1, 1/2, n)
        w4 = lambda n : rgt.normal(1, 1, n)
        w5 = lambda n : rgt.uniform(0, 2, n)
        w6 = lambda n : abs(rgt.normal(size=n)) * np.sqrt(np.pi / 2)

        boot_dict = {'Multinomial': w1, 'Exponential': w2, 'Rademacher': w3,\
                     'Gaussian': w4, 'Uniform': w5, 'Folded-normal': w6}

        return boot_dict[weight](self.n)

    def retire_weight(self, x, tau, c):
        tmp1 = tau * c * (x > c) - (1 - tau) * c * (x < -c)
        tmp2 = tau * x * (x >= 0) * (x <= c) + (1 - tau) * x * (x < 0) * (x >= -c)   
        return -(tmp1 + tmp2) / len(x)

    def conquer_weight(self, x, tau, kernel="Laplacian", w=np.array([])):
        ker1 = lambda x : 0.5 + 0.5 * np.sign(x) * (1 - np.exp(-abs(x)))
        ker2 = lambda x : norm.cdf(x)
        ker3 = lambda x : 1 / (1 + np.exp(-x))
        ker4 = lambda x : np.where(x > 1, 1, 0) + np.where(abs(x) <= 1, 0.5 * (1 + x), 0)
        ker5 = lambda x : 0.25 * (2 + 3 * x / 5 ** 0.5 - (x / 5 ** 0.5)**3 ) * (abs(x) <= 5 ** 0.5) + (x > 5 ** 0.5)
        ker_dict = {'Laplacian': ker1, 'Gaussian': ker2, 'Logistic': ker3, \
                    'Uniform': ker4, 'Epanechnikov': ker5}                       
        if not w.any():
            return (ker_dict[kernel](x) - tau) / len(x)
        else:
            return w * (ker_dict[kernel](x) - tau) / len(x)

    def retire(self, tau=0.5, robust=2, standardize=True, adjust=False):
        '''
            Robust/Huberized Expectile Regression
        '''
        if standardize: X = self.X1
        else: X = self.X

        beta0 = np.zeros(X.shape[1])
        if self.itcp: beta0[0] = np.quantile(self.Y, tau)
        res = self.Y - beta0[0]
        c = robust * self.mad(res)
        grad0 = X.T.dot(self.retire_weight(res, tau, c))
        diff_beta = -grad0
        beta1 = beta0 + diff_beta
        res, count = self.Y - X.dot(beta1), 0
 
        while np.max(np.abs(grad0)) > self.opt['tol'] and count < self.opt['max_iter']:
            c = robust * self.mad(res)
            grad1 = X.T.dot(self.retire_weight(res, tau, c))
            diff_grad = grad1 - grad0
            r0, r1 = diff_beta.dot(diff_beta), diff_grad.dot(diff_grad)
            r01 = diff_grad.dot(diff_beta)
            lr1, lr2 = r01/r1, r0/r01
            grad0, beta0 = grad1, beta1
            diff_beta = -min(lr1, lr2, self.opt['max_lr'])*grad1
            beta1 += diff_beta
            res = self.Y - X.dot(beta1)
            count += 1

        if standardize and adjust:
            beta1[1*self.itcp:] = beta1[self.itcp:] / self.sdX
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
                
        beta0 : initial estimate; default is np.array([]).
        
        res : n-vector of fitted residuals; default is np.array([]).
        
        weight : n-vector of observation weights; default is np.array([]).
        
        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale.

        Returns
        -------
        'beta' : conquer estimate.

        'res' : n-vector of fitted residuals.

        'niter' : number of iterations.

        'bw' : bandwidth.
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
        while count < self.opt['max_iter'] and np.max(np.abs(grad0)) > self.opt['tol']:
            grad1 = X.T.dot(self.conquer_weight(-res/h, tau, kernel, weight))
            diff_grad = grad1 - grad0
            r0, r1 = diff_beta.dot(diff_beta), diff_grad.dot(diff_grad)
            if r1 == 0: lr = 1
            else:
                r01 = diff_grad.dot(diff_beta)
                lr1, lr2 = r01/r1, r0/r01
                lr = min(lr1, lr2, self.opt['max_lr'])

            grad0, beta0 = grad1, beta1
            diff_beta = -lr*grad1
            beta1 += diff_beta
            res = self.Y - X.dot(beta1)
            count += 1
        
        if standardize and adjust:
            beta1[1*self.itcp:] = beta1[self.itcp:]/self.sdX
            if self.itcp: beta1[0] -= self.mX.dot(beta1[1:])

        return {'beta': beta1, 'res': res, 'niter': count, 'bw': h}


    def bw_path(self, tau=0.5, h_seq=np.array([]), L=20, \
                kernel="Laplacian", standardize=True, adjust=True):
        '''
            Solution Path of Conquer at a Sequence of Bandwidths
        
        Arguments
        ---------
        h_seq : a sequence of bandwidths.

        L : number of bandwdiths; default is 20.

        Returns
        -------
        'beta_seq' : a sequence of conquer estimates corresponding to bandwidths in descending order.

        'res_seq' : a sequence of residual vectors.

        'bw_seq' : a sequence of bandwidths in descending order.
        '''
        if not np.array(h_seq).any():
            h_seq = np.linspace(0.01, min((len(self.mX) + np.log(self.n))/self.n, 0.5)**0.4, num=L)

        if standardize: X = self.X1
        else: X = self.X

        h_seq, L = np.sort(h_seq)[::-1], len(h_seq)
        beta_seq = np.empty(shape=(X.shape[1], L))
        res_seq = np.empty(shape=(n, L))
        model = self.fit(tau, h_seq[0], kernel, standardize=standardize, adjust=False)
        beta_seq[:,0], res_seq[:,0] = model['beta'], model['res']

        for l in range(1,L):      
            model = self.fit(tau, h_seq[l], kernel, model['beta'], model['res'], standardize=standardize, adjust=False)
            beta_seq[:,l], res_seq[:,l] = model['beta'], model['res']
    
   
        if standardize and adjust:
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp:
                beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])

        return {'beta_seq': beta_seq, 'res_seq': res_seq, 'bw_seq': h_seq}
        

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
        'beta' : conquer estimate.
        
        'normal_ci' : numpy array. Normal CIs based on estimated asymptotic covariance matrix.
        '''
        if h == None: h = self.bandwidth(tau)
        X = self.X
        model = self.fit(tau, h, kernel, standardize=standardize)
        hess_weight = norm.pdf(model['res']/h)
        grad_weight = ( norm.cdf(-model['res']/h) - tau)**2
        hat_V = (X.T * grad_weight).dot(X)/self.n
        inv_J = np.linalg.inv((X.T * hess_weight).dot(X)/(self.n * h))
        ACov = inv_J.dot(hat_V).dot(inv_J)
        rad = norm.ppf(1-0.5*alpha)*np.sqrt( np.diag(ACov) / self.n )        
        ci = np.c_[model['beta'] - rad, model['beta'] + rad]

        return {'beta': model['beta'], 'normal_ci': ci}


    def mb(self, tau=0.5, h=None, kernel="Laplacian", weight="Exponential", standardize=True):
        '''
            Multiplier Bootstrap Estimates
   
        Parameters
        ----------
        tau : quantile level; default is 0.5.
        
        h : bandwidth. The default is computed by self.bandwidth(tau).

        kernel : a character string representing one of the built-in smoothing kernels; default is "Laplacian".

        weight : a character string representing one of the built-in bootstrap weight distributions; default is "Exponential".

        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.

        Returns
        -------
        mb_beta : numpy array. 1st column: conquer estimate; 2nd to last: bootstrap estimates.
        '''
        if h==None: h = self.bandwidth(tau)
        
        if weight not in self.weights:
            raise ValueError("weight distribution must be either Exponential, Rademacher, Multinomial, Gaussian, Uniform or Folded-normal")
           
        model = self.fit(tau, h, kernel, standardize=standardize, adjust=False)
        mb_beta = np.zeros([len(model['beta']), self.opt['nboot']+1])
        mb_beta[:,0], res = model['beta'], model['res']

        for b in range(self.opt['nboot']):
            model = self.fit(tau, h, kernel, beta0=mb_beta[:,0], res=res, \
                             weight=self.boot_weight(weight), standardize=standardize)
            mb_beta[:,b+1] = model['beta']

        if standardize:
            mb_beta[self.itcp:,0] = mb_beta[self.itcp:,0]/self.sdX
            if self.itcp: mb_beta[0,0] -= self.mX.dot(mb_beta[1:,0])

        ## delete NaN bootstrap estimates (when using Gaussian weights)
        mb_beta = mb_beta[:,~np.isnan(mb_beta).any(axis=0)]
        return mb_beta

    
    def mb_ci(self, tau=0.5, h=None, kernel="Laplacian", weight="Exponential", standardize=True, alpha=0.05):
        '''
            Multiplier Bootstrap Confidence Intervals

        Arguments
        ---------
        tau : quantile level; default is 0.5.

        h : bandwidth. The default is computed by self.bandwidth(tau).

        kernel : a character string representing one of the built-in smoothing kernels; default is "Laplacian".

        weight : a character string representing one of the built-in bootstrap weight distributions; default is "Exponential".

        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.

        alpha : miscoverage level for each CI; default is 0.05.

        Returns
        -------
        'boot_beta' : numpy array. 1st column: conquer estimate; 2nd to last: bootstrap estimates.
        
        'percentile_ci' : numpy array. Percentile bootstrap CI.

        'pivotal_ci' : numpy array. Pivotal bootstrap CI.

        'normal_ci' : numpy array. Normal-based CI using bootstrap variance estimates.
        '''
        if h==None: h = self.bandwidth(tau)
        
        mb_beta = self.mb(tau, h, kernel, weight, standardize)
        if weight in self.weights[:4]:
            adj = 1
        elif weight == 'Uniform':
            adj = np.sqrt(1/3)
        elif weight == 'Folded-normal':
            adj = np.sqrt(0.5*np.pi - 1)

        percentile_ci = np.c_[np.quantile(mb_beta[:,1:], 0.5*alpha, axis=1), \
                              np.quantile(mb_beta[:,1:], 1-0.5*alpha, axis=1)]
        pivotal_ci = np.c_[(1+1/adj)*mb_beta[:,0] - percentile_ci[:,1]/adj, \
                           (1+1/adj)*mb_beta[:,0] - percentile_ci[:,0]/adj]

        radi = norm.ppf(1-0.5*alpha)*np.std(mb_beta[:,1:], axis=1)/adj
        normal_ci = np.c_[mb_beta[:,0] - radi, mb_beta[:,0] + radi]

        return {'boot_beta': mb_beta, 
                'percentile_ci': percentile_ci,
                'pivotal_ci': pivotal_ci,
                'normal_ci': normal_ci}


    def qr(self, tau=0.5, lr=1, beta0=np.array([]), res=np.array([]), standardize=True, adjust=True):
        '''
            Quantile Regression via Subgradient Descent and Conquer Initialization
        
        Arguments
        ---------
        tau : quantile level; default is 0.5.

        lr : learning rate (step size); default is 1.

        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale.

        Returns
        -------
        'beta' : standard quantile regression estimate.

        'res' : n-vector of fitted residuals.

        'niter' : number of iterations.
        '''
        if not beta0.any():
            model = self.fit(tau=tau, standardize=standardize, adjust=False)
            beta0, res = model['beta'], model['res']
    
        if standardize: X = self.X1
        else: X = self.X

        sub_grad = lambda x : (x <= 0) - tau
        n, dev, count = len(res), 1, 0
        while count < self.opt['max_iter'] and dev > self.opt['tol'] * np.sum(beta0 ** 2):
            diff = lr * X.T.dot(sub_grad(res))/n
            beta0 -= diff
            dev = diff.dot(diff)
            res = self.Y - X.dot(beta0)
            count += 1

        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            if self.itcp: 
                beta0[0] -= self.mX.dot(beta0[1:])

        return {'beta': beta0, 'res': res, 'niter': count}




class high_dim(low_dim):
    '''
        Regularized Convolution Smoothed Quantile Regression via ILAMM
                        (iterative local adaptive majorize-minimization)
    '''
    weights = ['Multinomial', 'Exponential', 'Rademacher']
    penalties = ["L1", "SCAD", "MCP", "CapppedL1"]
    opt = {'phi': 0.1, 'gamma': 1.25, 'max_iter': 1e3, 'tol': 1e-5, \
           'irw_tol': 1e-4, 'nsim': 200, 'nboot': 200}

    def __init__(self, X, Y, intercept=True, options={}):

        '''
        Arguments
        ---------
        X : n by p matrix of covariates; each row is an observation vector.
           
        Y : n-dimensional vector of response variables.
            
        intercept : logical flag for adding an intercept to the model.

        options : a dictionary of internal statistical and optimization parameters.
        
            phi : initial quadratic coefficient parameter in the ILAMM algorithm; default is 0.1.
        
            gamma : adaptive search parameter that is larger than 1; default is 1.25.
        
            max_iter : maximum numder of iterations in the ILAMM algorithm; default is 1e3.
        
            tol : the ILAMM iteration stops when |beta^{k+1} - beta^k|^2/|beta^k|^2 <= tol; default is 1e-5.

            irw_tol : tolerance parameter for stopping iteratively reweighted L1-penalization; default is 1e-4.

            nsim : number of simulations for computing a data-driven lambda; default is 200.

            nboot : number of bootstrap samples for post-selection inference; default is 200.
        '''
        self.n, self.p = X.shape
        self.Y = Y.reshape(self.n)
        self.mX, self.sdX = np.mean(X, axis=0), np.std(X, axis=0)
        self.itcp = intercept
        if intercept:
            self.X = np.c_[np.ones(self.n), X]
            self.X1 = np.c_[np.ones(self.n,), (X - self.mX)/self.sdX]
        else:
            self.X, self.X1 = X, X/self.sdX

        self.opt.update(options)


    def bandwidth(self, tau):
        h0 = (np.log(self.p)/self.n)**0.25
        return max(0.05, h0*(tau-tau**2)**0.5)

    def soft_thresh(self, x, c):
        tmp = abs(x) - c
        return np.sign(x)*np.where(tmp<=0, 0, tmp)
    
    def self_tuning(self, tau=0.5, standardize=True):
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

        Returns
        -------
        lambda_sim : a numpy array of self.opt['nsim'] simulated lambda values.
        '''    
        if standardize: X = self.X1 
        else: X = self.X
        lambda_sim = np.array([max(abs(X.T.dot(tau - (rgt.uniform(0,1,self.n) <= tau))))
                               for b in range(self.opt['nsim'])])
        return lambda_sim/self.n
    

    def concave_weight(self, x, penalty="SCAD", a=None):
        if penalty == "SCAD":
            if a==None: a = 3.7
            tmp = 1 - (abs(x) - 1) / (a - 1)
            tmp = np.where(tmp <= 0, 0, tmp)
            return np.where(tmp > 1, 1, tmp)
        elif penalty == "MCP":
            if a==None: a = 3
            tmp = 1 - abs(x) / a 
            return np.where(tmp <= 0, 0, tmp)
        elif penalty == "CapppedL1":
            if a==None: a = 3
            return abs(x) <= a / 2
    

    def retire_loss(self, x, tau, c):
        out = 0.5 * (abs(x) <= c) * x**2 + (c * abs(x) - 0.5 * c ** 2) * (abs(x) > c)
        return np.mean(abs(tau - (x<0)) * out)
    

    def l1_retire(self, Lambda=np.array([]), tau=0.5, robust=3, beta0=np.array([]), \
                  res=np.array([]), standardize=True, adjust=True):
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

        phi, r0, count = self.opt['phi'], 1, 0
        while r0 > self.opt['tol']*np.sum(beta0**2) and count < self.opt['max_iter']:
            c = robust * self.mad(res)
            grad0 = X.T.dot(self.retire_weight(res, tau, c))
            loss_eval0 = self.retire_loss(res, tau, c)
            beta1 = beta0 - grad0/phi
            beta1[self.itcp:] = self.soft_thresh(beta1[self.itcp:], Lambda/phi)
            diff_beta = beta1 - beta0
            r0 = diff_beta.dot(diff_beta)
            
            res = self.Y - X.dot(beta1)
            loss_proxy = loss_eval0 + diff_beta.dot(grad0) + 0.5*phi*r0
            loss_eval1 = self.retire_loss(res, tau, c)
            
            while loss_proxy < loss_eval1:
                phi *= self.opt['gamma']
                beta1 = beta0 - grad0/phi
                beta1[self.itcp:] = self.soft_thresh(beta1[self.itcp:], Lambda/phi)
                diff_beta = beta1 - beta0
                r0 = diff_beta.dot(diff_beta)
                res = self.Y - X.dot(beta1)
                loss_proxy = loss_eval0 + diff_beta.dot(grad0) + 0.5*phi*r0
                loss_eval1 = self.retire_loss(res, tau, c)
                
            beta0, phi = beta1, self.opt['phi']
            count += 1

        if standardize and adjust:
            beta1[self.itcp:] = beta1[self.itcp:]/self.sdX
            if self.itcp: beta1[0] -= self.mX.dot(beta1[1:])

        return {'beta': beta1, 'res': res, 'niter': count, 'lambda': Lambda}
        

    def l1(self, Lambda=np.array([]), tau=0.5, h=None, kernel="Laplacian", \
           beta0=np.array([]), res=np.array([]), standardize=True, adjust=True, weight=np.array([])):
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
        
        weight : an n-vector of observation weights; default is np.array([]) (empty).

        Returns
        -------
        'beta' : a numpy array of estimated coefficients.
        
        'res' : an n-vector of fitted residuals.

        'niter' : number of iterations.

        'lambda' : lambda value.

        '''
        if not np.array(Lambda).any():
            Lambda = 0.75*np.quantile(self.self_tuning(tau,standardize), 0.9)
        if h == None: h = self.bandwidth(tau)
        
        if not beta0.any():
            model = self.l1_retire(Lambda, tau, standardize=standardize, adjust=False)
            beta0, res, count = model['beta'], model['res'], model['niter']
        else: count = 0
                    
        if standardize: X = self.X1
        else: X = self.X
        
        phi, r0 = self.opt['phi'], 1 
        while r0 > self.opt['tol']*np.sum(beta0**2) and count < self.opt['max_iter']:
            
            grad0 = X.T.dot(self.conquer_weight(-res/h, tau, kernel, weight))
            loss_eval0 = self.smooth_check(res, tau, h, kernel, weight)
            beta1 = beta0 - grad0/phi
            beta1[self.itcp:] = self.soft_thresh(beta1[self.itcp:], Lambda/phi)
            diff_beta = beta1 - beta0
            r0 = diff_beta.dot(diff_beta)          
            res = self.Y - X.dot(beta1)
            loss_proxy = loss_eval0 + diff_beta.dot(grad0) + 0.5*phi*r0
            loss_eval1 = self.smooth_check(res, tau, h, kernel, weight)
            
            while loss_proxy < loss_eval1:
                phi *= self.opt['gamma']
                beta1 = beta0 - grad0/phi
                beta1[self.itcp:] = self.soft_thresh(beta1[self.itcp:], Lambda/phi)
                diff_beta = beta1 - beta0
                r0 = diff_beta.dot(diff_beta)
                res = self.Y - X.dot(beta1)
                loss_proxy = loss_eval0 + diff_beta.dot(grad0) + 0.5*phi*r0
                loss_eval1 = self.smooth_check(res, tau, h, kernel, weight)
                
            beta0, phi = beta1, self.opt['phi']
            count += 1

        if standardize and adjust:
            beta1[self.itcp:] = beta1[self.itcp:]/self.sdX
            if self.itcp: beta1[0] -= self.mX.dot(beta1[1:])
            
        return {'beta': beta1, 'res': res, 'niter': count, 'lambda': Lambda}


    def irw(self, Lambda=None, tau=0.5, h=None, kernel="Laplacian", beta0=np.array([]), res=np.array([]),
            penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True, weight=np.array([])):
        '''
            Iteratively Reweighted L1-Penalized Conquer (irw-l1-conquer)
            
        Arguments
        ----------
        penalty : a character string representing one of the built-in concave penalties; default is "SCAD".
        
        a : the constant (>2) in the concave penality; default is 3.7.
        
        nstep : number of iterations/steps of the IRW algorithm; default is 5.

        Returns
        -------
        'beta' : a numpy array of estimated coefficients.
        
        'res' : an n-vector of fitted residuals. 

        'nirw' : number of reweighted penalization steps.

        'lambda' : lambda value.
        '''
        if Lambda == None: 
            Lambda = 0.75*np.quantile(self.self_tuning(tau,standardize), 0.9)
        if h == None: h = self.bandwidth(tau)
        
        if not beta0.any():
            model = self.l1(Lambda, tau, h, kernel, standardize=standardize, adjust=False, weight=weight)
        else:
            model = self.l1(Lambda, tau, h, kernel, beta0, res, standardize, adjust=False, weight=weight)
        beta0, res = model['beta'], model['res']

        err, count = 1, 1
        while err > self.opt['irw_tol'] and count <= nstep:
            rw_lambda = Lambda * self.concave_weight(beta0[self.itcp:]/Lambda, penalty, a)
            model = self.l1(rw_lambda, tau, h, kernel, beta0, res, standardize, adjust=False, weight=weight)
            err = np.sum((model['beta']-beta0)**2)/np.sum(beta0**2)
            beta0, res = model['beta'], model['res']
            count += 1
        
        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            if self.itcp: beta0[0] -= self.mX.dot(beta0[1:])
            
        return {'beta': beta0, 'res': res, 'nirw': count, 'lambda': Lambda}
    
     
    def irw_retire(self, Lambda=None, tau=0.5, robust=3, penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True):
        '''
            Iteratively Reweighted L1-Penalized Retire (irw-l1-retire)
        '''
        if Lambda == None: 
            Lambda = np.quantile(self.self_tuning(tau,standardize), 0.9)
        

        model = self.l1_retire(Lambda, tau, robust, standardize=standardize, adjust=False)
        beta0, res = model['beta'], model['res']
        err, count = 1, 1
        while err > self.opt['irw_tol'] and count <= nstep:
            rw_lambda = Lambda * self.concave_weight(beta0[self.itcp:]/Lambda, penalty, a)
            model = self.l1_retire(rw_lambda, tau, robust, beta0, res, standardize, adjust=False)
            err = np.sum((model['beta']-beta0)**2)/np.sum(beta0**2)
            beta0, res = model['beta'], model['res']
            count += 1
        
        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            if self.itcp: beta0[0] -= self.mX.dot(beta0[1:])
            
        return {'beta': beta0, 'res': res, 'nirw': count, 'lambda': Lambda}
    
    
    def l1_path(self, lambda_seq, tau=0.5, h=None, kernel="Laplacian", standardize=True, adjust=True):
        '''
            Solution Path of L1-Penalized Conquer

        Arguments
        ---------
        lambda_seq : a numpy array of lambda values.

        Returns
        -------
        'beta_seq' : a sequence of l1-conquer estimates. Each column corresponds to an estiamte for a lambda value.

        'res_seq' : a sequence of residual vectors. 

        'size_seq' : a sequence of numbers of selected variables. 

        'lambda_seq' : a sequence of lambda values in descending order.

        'bw' : bandwidth.
        '''
        if h == None: h = self.bandwidth(tau)
        lambda_seq = np.sort(lambda_seq)[::-1]
        beta_seq = np.empty(shape=(self.X.shape[1], len(lambda_seq)))
        res_seq = np.empty(shape=(self.n, len(lambda_seq)))
        model = self.l1(lambda_seq[0], tau, h, kernel, standardize=standardize, adjust=False)
        beta_seq[:,0], res_seq[:,0] = model['beta'], model['res']
        
        for l in range(1,len(lambda_seq)):
            model = self.l1(lambda_seq[l], tau, h, kernel, model['beta'], model['res'], standardize, adjust=False)
            beta_seq[:,l], res_seq[:,l] = model['beta'], model['res']

        if standardize and adjust:
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])

        return {'beta_seq': beta_seq, 'res_seq': res_seq, \
                'size_seq': np.sum((abs(beta_seq[self.itcp:,:])>0), axis=0), \
                'lambda_seq': lambda_seq, 'bw': h}

    
    def irw_path(self, lambda_seq, tau=0.5, h=None, kernel="Laplacian", \
                 penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True):
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
        
        nstep : number of iterations/steps of the IRW algorithm; default is 5.
        
        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale; default is TRUE.


        Returns
        -------
        'beta_seq' : a sequence of irw-l1-conquer estimates. Each column corresponds to an estiamte for a lambda value.

        'res_seq' : a sequence of residual vectors. 

        'size_seq' : a sequence of numbers of selected variables. 

        'lambda_seq' : a sequence of lambda values in descending order.

        'bw' : bandwidth.
        '''
        if h == None: h = self.bandwidth(tau)
        
        lambda_seq, nlambda = np.sort(lambda_seq)[::-1], len(lambda_seq)
        beta_seq = np.empty(shape=(self.X.shape[1], nlambda))
        res_seq = np.empty(shape=(self.n, nlambda))
        
        model = self.irw(lambda_seq[0], tau, h, kernel, penalty=penalty, a=a, nstep=nstep, \
                         standardize=standardize, adjust=False)
        beta_seq[:,0], res_seq[:,0] = model['beta'], model['res']
        
        for l in range(1, nlambda):
            model = self.irw(lambda_seq[l], tau, h, kernel, model['beta'], model['res'], \
                             penalty, a, nstep, standardize, adjust=False)
            beta_seq[:,l], res_seq[:,l] = model['beta'], model['res']
        
        if standardize and adjust:
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])
    
        return {'beta_seq': beta_seq, 'res_seq': res_seq, \
                'size_seq': np.sum((abs(beta_seq[self.itcp:,:])>0), axis=0), \
                'lambda_seq': lambda_seq, 'bw': h}


    def bic(self, tau=0.5, h=None, lambda_seq=np.array([]), nlambda=100, kernel="Laplacian", Cn=None, \
            penalty="SCAD", a=3.7, nstep=5, standardize=True, adjust=True):
        '''
            Model Selection via Bayesian Information Criterion
        
        Reference
        ---------
        Model selection via Bayesian information criterion for quantile regression models (2014)
        by Eun Ryung Lee, Hohsuk Noh and Byeong U. Park
        Journal of the American Statistical Association 109(505): 216--229.

        Arguments
        ---------
        see l1_path() and irw_path() 
        
        Cn : a positive constant (that diverges as sample size increases) in the modified BIC; default is log(p).

        Returns
        -------
        'bic_beta' : estimated coefficient vector for the BIC-selected model.

        'bic_seq' : residual vector for the BIC-selected model.

        'bic_size' : size of the BIC-selected model.

        'bic_lambda' : lambda value that corresponds to the BIC-selected model.

        'bw' : bandwidth.
        '''    

        if not lambda_seq.any():
            sim_lambda = self.self_tuning(tau=tau, standardize=standardize)
            lambda_seq = np.linspace(np.quantile(sim_lambda, 0.25), \
                                     np.quantile(sim_lambda, 0.75), \
                                     num=nlambda)
        else:
            nlambda = len(lambda_seq)

        if Cn == None: Cn = np.log(self.p)

        if penalty not in self.penalties: raise ValueError("penalty must be either L1, SCAD, MCP or CapppedL1")

        check_sum = lambda x : np.sum(np.where(x >= 0, tau * x, (tau - 1) * x))

        
        if penalty == "L1":
            model_all = self.l1_path(lambda_seq, tau, h, kernel, standardize, adjust)
        else:
            model_all = self.irw_path(lambda_seq, tau, h, kernel, penalty, a, nstep, standardize, adjust) 

        BIC = np.array([np.log(check_sum(model_all['res_seq'][:,l])) for l in range(nlambda)])
        BIC += model_all['size_seq'] * np.log(self.n) * Cn / (2 * self.n)
        bic_idx = BIC==min(BIC)
 
        return {'bic_beta': model_all['beta_seq'][:,bic_idx], \
                'bic_res':  model_all['res_seq'][:,bic_idx], \
                'bic_size': model_all['size_seq'][bic_idx], \
                'bic_lambda': model_all['lambda_seq'][bic_idx], \
                'bw': model_all['bw']}




    def boot_select(self, Lambda=None, tau=0.5, h=None, kernel="Laplacian", \
                    weight="Multinomial", alpha=0.05, penalty="SCAD", a=3.7, nstep=5, \
                    standardize=True, parallel=False, ncore=None):
        '''
            Model Selection via Bootstrap 

        Arguments
        ---------
        Lambda : scalar regularization parameter. If unspecified, it will be computed by self.self_tuning().
        
        tau : quantile level; default is 0.5.

        h : smoothing parameter/bandwidth. The default is computed by self.bandwidth().
        
        kernel : a character string representing one of the built-in smoothing kernels; default is "Laplacian".

        weight : a character string representing one of the built-in bootstrap weight distributions; default is "Multinomial".
                
        alpha : miscoverage level for each CI; default is 0.05.
        
        penalty : a character string representing one of the built-in concave penalties; default is "SCAD".
        
        a : the constant (>2) in the concave penality; default is 3.7.
        
        nstep : number of iterations/steps of the IRW algorithm; default is 5.
        
        standardize : logical flag for x variable standardization prior to fitting the model; default is TRUE.

        parallel : logical flag to implement bootstrap using parallel computing; default is FALSE.

        ncore : number of cores used for parallel computing.

        Returns
        -------
        'boot_beta' : numpy array. 1st column: penalized conquer estimate; 2nd to last: bootstrap estimates.
            
        'majority_vote' : selected model by majority vote.

        'intersection' : selected model by intersecting.
        '''
        if Lambda == None: Lambda = 0.75*np.quantile(self.self_tuning(tau, standardize), 0.9)
        if h == None: h = self.bandwidth(tau) 
        if weight not in self.weights[:3]:
            raise ValueError("weight distribution must be either Exponential, Rademacher or Multinomial")

        model = self.irw(Lambda, tau, h, kernel, penalty=penalty, a=a, nstep=nstep, \
                         standardize=standardize, adjust=False)
        mb_beta = np.zeros(shape=(self.p+self.itcp, self.opt['nboot']+1))
        mb_beta[:,0] = model['beta']
        if standardize:
            mb_beta[self.itcp:,0] = mb_beta[self.itcp:,0]/self.sdX
            if self.itcp: mb_beta[0,0] -= self.mX.dot(mb_beta[1:,0])

        if parallel:
            import multiprocessing
            max_ncore = multiprocessing.cpu_count()
            if ncore == None: ncore = max_ncore
            if ncore > max_ncore: raise ValueError("number of cores exceeds the limit")

        def bootstrap(b):
            boot_fit = self.irw(Lambda, tau, h, kernel, beta0=model['beta'], res=model['res'], \
                                penalty=penalty, a=a, nstep=nstep,
                                standardize=standardize, weight=self.boot_weight(weight))
            return boot_fit['beta']

        if not parallel:
            for b in range(self.opt['nboot']): mb_beta[:,b+1] = bootstrap(b)
        else:
            from joblib import Parallel, delayed
            num_cores = multiprocessing.cpu_count()
            boot_results = Parallel(n_jobs=ncore)(delayed(bootstrap)(b) for b in range(self.opt['nboot']))
            mb_beta[:,1:] = np.array(boot_results).T
        
        ## delete NaN bootstrap estimates (when using Gaussian weights)
        mb_beta = mb_beta[:,~np.isnan(mb_beta).any(axis=0)]
        
        ## Method 1: Majority vote among all bootstrap models
        selection_rate = np.mean(mb_beta[self.itcp:,1:]!=0, axis=1)
        model_1 = np.where(selection_rate>0.5)[0]
        
        ## Method 2: Intersection of all bootstrap models
        model_2 = np.arange(self.p)
        for b in range(len(mb_beta[0,1:])):
            boot_model = np.where(abs(mb_beta[self.itcp:,b+1])>0)[0]
            model_2 = np.intersect1d(model_2, boot_model)

        return {'boot_beta': mb_beta, 'majority_vote': model_1, 'intersection': model_2}


    def boot_inference(self, Lambda=None, tau=0.5, h=None, kernel="Laplacian", \
                       weight="Multinomial", alpha=0.05, penalty="SCAD", a=3.7, nstep=5, \
                       standardize=True, parallel=False, ncore=1):
        '''
            Post-Selection-Inference via Bootstrap

        Arguments
        ---------
        see boot_select().

        Returns
        -------
        see boot_select().

        'percentile_ci' : numpy array. Percentile bootstrap CI.

        'pivotal_ci' : numpy array. Pivotal bootstrap CI.

        'normal_ci' : numpy array. Normal-based CI using bootstrap variance estimates.
        '''
        mb_model = self.boot_select(Lambda, tau, h, kernel, weight, alpha, penalty, a, nstep, standardize, parallel)
        
        percentile_ci = np.zeros([self.p + self.itcp, 2])
        pivotal_ci = np.zeros([self.p + self.itcp, 2])
        normal_ci = np.zeros([self.p + self.itcp, 2])

        # post-selection bootstrap inference
        X_select = self.X[:, mb_model['majority_vote']+self.itcp]
        model = low_dim(X_select, self.Y, self.itcp).mb_ci(tau, kernel=kernel, weight=weight, \
                                                           alpha=alpha, standardize=standardize)

        percentile_ci[mb_model['majority_vote']+self.itcp,:] = model['percentile_ci'][self.itcp:,:]
        pivotal_ci[mb_model['majority_vote']+self.itcp,:] = model['pivotal_ci'][self.itcp:,:]
        normal_ci[mb_model['majority_vote']+self.itcp,:] = model['normal_ci'][self.itcp:,:]

        if self.itcp: 
            percentile_ci[0,:] = model['percentile_ci'][0,:]
            pivotal_ci[0,:] = model['pivotal_ci'][0,:]
            normal_ci[0,:] = model['normal_ci'][0,:]

        return {'boot_beta': mb_model['boot_beta'], \
                'percentile_ci': percentile_ci, \
                'pivotal_ci': pivotal_ci, \
                'normal_ci': normal_ci, \
                'majority_vote': mb_model['majority_vote'], \
                'intersection': mb_model['intersection']}



class cv_lambda():
    '''
        Cross-Validated Penalized Conquer 
    '''
    penalties = ["L1", "SCAD", "MCP"]
    opt = {'nsim': 200, 'phi': 0.1, 'gamma': 1.25, 'max_iter': 1e3, 'tol': 1e-4, 'irw_tol': 1e-4}

    def __init__(self, X, Y, intercept=True, options={}):
        self.n, self.p = X.shape
        self.X, self.Y = X, Y.reshape(self.n)
        self.itcp = intercept
        self.opt.update(options)

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

        sqr = high_dim(self.X, self.Y, self.itcp, self.opt)
        if h == None: h = sqr.bandwidth(tau)

        if not lambda_seq.any():
            lambda_max = np.max(sqr.self_tuning(tau, standardize))
            lambda_seq = np.linspace(0.25*lambda_max, 1.25*lambda_max, nlambda)
        else:
            nlambda = len(lambda_seq)

        if penalty not in self.penalties: raise ValueError("penalty must be either L1, SCAD or MCP")

        check_loss = lambda x : np.mean(np.where(x >= 0, tau * x, (tau - 1)*x))   # empirical check loss
        idx, folds = self.divide_sample(nfolds)
        val_err = np.zeros((nfolds, nlambda))
        for v in range(nfolds):
            X_train, Y_train = self.X[np.setdiff1d(idx,folds[v]),:], self.Y[np.setdiff1d(idx,folds[v])]
            X_val, Y_val = self.X[folds[v],:], self.Y[folds[v]]
            sqr_train = high_dim(X_train, Y_train, self.itcp, self.opt)

            if penalty == "L1":
                model = sqr_train.l1_path(lambda_seq, tau, h, kernel, standardize, adjust)
            else:
                model = sqr_train.irw_path(lambda_seq, tau, h, kernel, penalty, a, nstep, standardize, adjust)

            val_err[v,:] = np.array([check_loss(Y_val - model['beta_seq'][0,l]*self.itcp \
                                     - X_val.dot(model['beta_seq'][self.itcp:,l])) for l in range(nlambda)])
        
        cv_err = np.mean(val_err, axis=0)
        cv_min = min(cv_err)
        lambda_min = model['lambda_seq'][cv_err==cv_min][0]
        if penalty == "L1":
            cv_model = sqr.l1(lambda_min, tau, h, kernel, standardize=standardize, adjust=adjust)
        else:
            cv_model = sqr.irw(lambda_min, tau, h, kernel, penalty=penalty, a=a, nstep=nstep, \
                               standardize=standardize, adjust=adjust)

        return {'cv_beta': cv_model['beta'], \
                'cv_res': cv_model['res'], \
                'lambda_min': lambda_min, \
                'lambda_seq': model['lambda_seq'], \
                'min_cv_err': cv_min, \
                'cv_err': cv_err}



class validate_lambda(cv_lambda):
    '''
        Train Penalized Conquer on a Validation Set
    '''
    penalties = ["L1", "SCAD", "MCP"]
    
    def __init__(self, X_train, Y_train, X_val, Y_val, intercept=True, options={}):
        self.n, self.p = X_train.shape
        self.X_train, self.Y_train = X_train, Y_train.reshape(self.n)
        self.X_val, self.Y_val = X_val, Y_val.reshape(len(Y_val))
        self.itcp = intercept
        self.opt.update(options)

   
    def train(self, tau=0.5, h=None, lambda_seq=np.array([]), nlambda=20, 
                kernel="Laplacian", penalty="SCAD", a=3.7, nstep=5, standardize=True):
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
            lambda_max = np.max(sqr_train.self_tuning(tau, standardize))
            lambda_seq = np.linspace(0.25*lambda_max, lambda_max, nlambda)
        else:
            nlambda = len(lambda_seq)
            
        if h == None: h = sqr_train.bandwidth(tau)

        if penalty not in self.penalties:
            raise ValueError("penalty must be either L1, SCAD or MCP")
        elif penalty == "L1":
            model_train = sqr_train.l1_path(lambda_seq, tau, h, kernel, standardize)
        else:
            model_train = sqr_train.irw_path(lambda_seq, tau, h, kernel, penalty, a, nstep, standardize)
        
        check_loss = lambda x : np.mean(np.where(x >= 0, tau * x, (tau - 1)*x))   # empirical check loss
        val_err = np.array([check_loss(self.Y_val - model_train['beta_seq'][0,l]*self.itcp \
                            - self.X_val.dot(model_train['beta_seq'][self.itcp:,l])) for l in range(nlambda)])
        val_min = min(val_err)
        l_min = np.where(val_err==val_min)[0][0]

        return {'val_beta': model_train['beta_seq'][:,l_min], \
                'val_res': model_train['res_seq'][:,l_min], \
                'val_size': model_train['size_seq'][l_min], \
                'lambda_min': model_train['lambda_seq'][l_min], \
                'lambda_seq': model_train['lambda_seq'], \
                'min_val_err': val_min, 'val_err': val_err}
