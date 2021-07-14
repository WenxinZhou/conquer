"""
    Regularized Convolution Smoothed Quantile Regression

Reference:
High-dimensional Quantile Regression: Convolution Smoothing and Concave Regularization (2020)
by Kean Ming Tan, Lan Wang and Wenxin Zhou 

@author: Wenxin Zhou (E-mail: wez243@ucsd.edu)
"""

import numpy as np
from conquer import conquer
from scipy.stats import norm

class reg_conquer(conquer):
    weights = ['Exponential', 'Rademacher', 'Multinomial']

    def __init__(self, X, Y, intercept=True):
        self.n, self.p = X.shape
        self.Y = Y
        self.mX = np.mean(X, axis=0)
        self.sdX = np.std(X, axis=0)
        self.itcp = intercept
        if intercept:
            self.X = np.concatenate([np.ones((self.n,1)), X], axis=1)
            self.X1 = np.concatenate([np.ones((self.n,1)), (X - self.mX)/self.sdX], axis=1)
        else:
            self.X, self.X1 = X, X/self.sdX

    def h(self, tau):
        h0 = (np.log(self.p)/self.n)**0.25
        return max(0.05, h0*(tau-tau**2)**0.5)

    def soft_thresh(self, x, c):
        tmp = abs(x) - c
        return np.sign(x)*np.where(tmp<=0, 0, tmp)
    
    def self_tuning(self, tau=0.5, standardize=True, B=200):
        '''
            A Simulation-based Approach for Choosing the Penalty Level (Lambda)
        
        Reference:
        l1-penalized Quantile Regression in High-dimensinoal Sparse Models
        by Alexandre Belloni and Victor Chernozhukov
        The Annals of Statistics 39(1): 82--130.

        Arguments
        ----------
        tau : quantile level. The default is 0.5.
        
        standardize : logical flag for x variable standardization prior to fitting the model. Default is TRUE.
        
        B : number of simulations. The default is 200.

        Returns
        -------
        Lambda_sim : a numpy array (with length B) of simulated Lambda values.
        '''    
        if standardize: X = self.X1 
        else: X = self.X
        Lambda_sim = np.empty(B)

        for b in range(B):
            weight = tau - (np.random.uniform(0,1,self.n) <= tau)
            Lambda_sim[b] = 1.1*max(abs(X.T.dot(weight)/self.n))
        return Lambda_sim
    
    def concave_weight(self, x, penalty="SCAD", a=None):
        if penalty == "SCAD":
            if a==None: a = 3.7
            tmp = 1 - (abs(x)-1)/(a-1)
            tmp = np.where(tmp<=0,0,tmp)
            return np.where(tmp>1, 1, tmp)
        if penalty == "MCP":
            if a==None: a = 2
            tmp = 1 - abs(x)/a 
            return np.where(tmp<=0, 0, tmp)
        if penalty == "CapppedL1":
            if a==None: a = 2
            return 1*(abs(x) <= a/2)
    
    def smooth_check(self, x, tau=0.5, h=None, kernel='Laplacian', w=np.array([])):
        if h == None: h = self.h(tau)       
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
    
    def retire_loss(self, x, tau, c):
        out = 0.5*(abs(x)<=c)* x**2 + (c*abs(x)-0.5*c**2)*(abs(x)>c)
        return np.mean( abs(tau - (x<0))*out )
    
    def l1_retire(self, Lambda=np.array([]), tau=0.5, tune=3, beta0=np.array([]), res=np.array([]), 
                    standardize=True, adjust=True, phi=0.1, gamma=1.25, max_iter=1000, tol=1e-10):   
        if not np.array(Lambda).any(): 
            Lambda = np.median(self.self_tuning(tau,standardize))

        if standardize: X = self.X1
        else: X = self.X
        
        if not beta0.any():
            beta0 = np.zeros(X.shape[1])
            if self.itcp: beta0[0] = np.quantile(self.Y, tau)
            res = self.Y - beta0[0]

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
            standardize=True, adjust=True, weight=np.array([]), phi=0.1, gamma=1.25, max_iter=500, tol=1e-10):
        '''
            L1-penalized Convolution Smoothed Quantile Regression (L1-Conquer)
        
        Arguments
        ----------
        Lambda : regularization parameter. This should be either a scalar, or 
                 a vector of length equal to the column dimension of X. If not 
                 specified, it will be computed by self.self_tuning().

        tau : quantile level. The default is 0.5.

        h : smoothing parameter/bandwidth. The default is computed by self.h().
        
        kernel : a character string representing one of the built-in smoothing kernels. The default is "Laplacian".

        beta0 : initial estimator. If unspecified, it will be set as zero.

        res : residual vector of the initial estiamtor
        
        standardize : logical flag for x variable standardization prior to fitting the model. Default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale.            
        
        weight : n-vector of observation weights. The default is np.array([]).
        
        phi : quadratic coefficient parameter in the ILAMM algorithm. The default is 0.001.
        
        gamma : adaptive search parameter that is larger than 1. The default is 1.25.
        
        max_iter : maximum numder of iterations. The default is 1000.
        
        tol : tolerance value. The default is 1e-10.

        Returns
        -------
        beta1 : a numpy array of estimated coefficients.
        
        list : a list of residual vector, number of iterations and Lambda components.

        '''
        if not np.array(Lambda).any():
            Lambda = np.quantile(self.self_tuning(tau,standardize), 0.9)
        if h == None: h = self.h(tau)
        
        if not beta0.any():
            beta0, retire_fit = self.l1_retire(Lambda, tau, standardize=standardize, adjust=False)
            res = retire_fit[0]            
                    
        if standardize: X = self.X1
        else: X = self.X
        
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
                    penalty="SCAD", a=3.7, step=5, standardize=True, adjust=True, weight=np.array([]), tol=1e-6):
        '''
            Iteratively Reweighted L1-penalized Conquer (IRW-L1-Conquer)
            
        Arguments
        ----------
        penalty : a character string representing one of the built-in concave penalties. The default is "SCAD".
        
        a : the constant (>2) in the concave penality. The default is 3.7.
        
        step : the number of iterations/steps of the IRW algorithm. The default is 5.

        Returns
        -------
        beta1 : a numpy array of estimated coefficients.
        
        list : a list of residual vector, number of iterations and Lambda components.
        '''
        if Lambda == None: 
            Lambda = np.quantile(self.self_tuning(tau,standardize), 0.9)
        if h == None: h = self.h(tau)
        
        if not beta0.any():
            beta0, fit = self.l1(Lambda, tau, h, kernel, standardize=standardize, adjust=False, weight=weight)
        else:
            beta0, fit = self.l1(Lambda, tau, h, kernel, beta0, res, standardize, adjust=False, weight=weight)
        res = fit[0]

        err, count = 1, 1
        while err > tol and count <= step:
            rw_lambda = Lambda * self.concave_weight(beta0[self.itcp:]/Lambda, penalty, a)
            beta1, fit = self.l1(rw_lambda, tau, h, kernel, beta0, res, standardize, adjust=False, weight=weight)
            err = max(abs(beta1-beta0))
            beta0, res = beta1, fit[0]
            count += 1
        
        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            if self.itcp: beta0[0] -= self.mX.dot(beta0[1:])
            
        return beta0, [res, count, Lambda]
    
     
    def irw_retire(self, Lambda=None, tau=0.5, tune=3, penalty="SCAD", a=3.7, step=5, 
                    standardize=True, adjust=True, tol=1e-5):
        '''
            Iteratively Reweighted L1-penalized Retire (IRW-L1-Retire)
        '''
        if Lambda == None: 
            Lambda = np.quantile(self.self_tuning(tau,standardize), 0.9)
        
        beta0, fit0 = self.l1_retire(Lambda, tau, tune, standardize=standardize, adjust=False)
        res = fit0[0]
        err, count = 1, 1
        while err > tol and count <= step:
            rw_lambda = Lambda * self.concave_weight(beta0[self.itcp:]/Lambda, penalty, a)
            beta1, fit1 = self.l1_retire(rw_lambda, tau, tune, beta0, res, standardize, adjust=False)
            err = max(abs(beta1-beta0))
            beta0, res = beta1, fit1[0]
            count += 1
        
        if standardize and adjust:
            beta0[self.itcp:] = beta0[self.itcp:]/self.sdX
            if self.itcp: beta0[0] -= self.mX.dot(beta0[1:])
            
        return beta0, [res, count, Lambda]
    
    
    def l1_path(self, Lambda, tau=0.5, h=None, kernel="Laplacian", standardize=True, adjust=True):
        '''
            Solution Path of L1-penalized Conquer

        Arguments
        ----------
        Lambda : a numpy array of Lambda values.

        Returns
        -------
        beta_seq : a numpy array of a sequence of l1-conquer estimates. Each column corresponds to a Lambda value.

        list : a list of redisual sequence, a sequence of numbers of nonzeros, a sequence of Lambda values in descending order, and bandwidth.
        '''
        if h == None: h = self.h(tau)
        Lambda = np.sort(Lambda)[::-1]
        beta_seq = np.empty(shape=(self.X.shape[1], len(Lambda)))
        res_seq = np.empty(shape=(self.n, len(Lambda)))
        beta_seq[:,0], fit = self.l1(Lambda[0], tau, h, kernel, standardize=standardize, adjust=False)
        res_seq[:,0] = fit[0]
        
        for l in range(1,len(Lambda)):
            beta_seq[:,l], fit = self.l1(Lambda[l], tau, h, kernel, beta_seq[:,l-1],
                                         fit[0], standardize, adjust=False)
            res_seq[:,l] = fit[0]

        if standardize and adjust:
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])

        return beta_seq, [res_seq, np.sum((abs(beta_seq[self.itcp:,:])>0), axis=0), Lambda, h]
    
    
    def irw_path(self, Lambda, tau=0.5, h=None, kernel="Laplacian", penalty="SCAD", a=3.7, step=5, standardize=True, adjust=True):
        '''
            Solution Path of Iteratively-Reweighted L1-Conquer

        Arguments
        ----------
        Lambda : a numpy array of Lambda values.

        tau : quantile level. The default is 0.5.

        h : smoothing parameter/bandwidth. The default is computed by self.h().
        
        kernel : a character string representing one of the built-in smoothing kernels. The default is "Laplacian".
        
        penalty : a character string representing one of the built-in concave penalties. The default is "SCAD".
        
        a : the constant (>2) in the concave penality. The default is 3.7.
        
        step : the number of iterations/steps of the IRW algorithm. The default is 5.
        
        standardize : logical flag for x variable standardization prior to fitting the model. Default is TRUE.
        
        adjust : logical flag for returning coefficients on the original scale. Default is TRUE.

        Returns
        -------
        beta_seq : a numpy array of a sequence of l1-conquer estimates. Each column corresponds to a Lambda value.

        list : a list of redisual sequence, a sequence of numbers of nonzeros, a sequence of Lambda values in descending order, and bandwidth.
        '''
        if h == None: h = self.h(tau)
        Lambda = np.sort(Lambda)[::-1]
        beta_seq = np.empty(shape=(self.X.shape[1], len(Lambda)))
        res_seq = np.empty(shape=(self.n, len(Lambda)))
        beta_seq[:,0], fit = self.irw(Lambda[0], tau, h, kernel, penalty=penalty, a=a, step=step, standardize=standardize, adjust=False)
        res_seq[:,0] = fit[0]
        for l in range(1,len(Lambda)):
            beta_seq[:,l], fit = self.irw(Lambda[l], tau, h, kernel, beta_seq[:,l-1], fit[0], penalty, a, step, standardize, adjust=False)
            res_seq[:,l] = fit[0]
        
        if standardize and adjust:
            beta_seq[self.itcp:,] = beta_seq[self.itcp:,]/self.sdX[:,None]
            if self.itcp: beta_seq[0,:] -= self.mX.dot(beta_seq[1:,])
    
        return beta_seq, [res_seq, np.sum((abs(beta_seq[self.itcp:,:])>0), axis=0), Lambda, h]



    def boot_select(self, Lambda=None, tau=0.5, h=None, kernel="Laplacian", weight="Multinomial",
                    B=200, alpha=0.05, penalty="SCAD", a=3.7, step=5, standardize=True):
        '''
            Model Selection via Bootstrap 

        Parameters
        ----------
        Lambda : scalar regularization parameter. If unspecified, it will be computed by self.self_tuning().
        
        tau : quantile level. The default is 0.5.

        h : smoothing parameter/bandwidth. The default is computed by self.h().
        
        kernel : a character string representing one of the built-in smoothing kernels. The default is "Laplacian".

        weight : a character string representing one of the built-in bootstrap weight distributions. The default is "Multinomial".
        
        B : number of bootstrap replications. The default is 200.
        
        alpha : 100*(1-alpha)% CI. The default is 0.05.
        
        penalty : a character string representing one of the built-in concave penalties. The default is "SCAD".
        
        a : the constant (>2) in the concave penality. The default is 3.7.
        
        step : the number of iterations/steps of the IRW algorithm. The default is 5.
        
        standardize : logical flag for x variable standardization prior to fitting the model. Default is TRUE.

        Returns
        -------
        mb_beta : p+1/p by B+1 numpy array. 1st column: penalized conquer estimator; 2nd to last: bootstrap estimates.
            
        list : a list of two selected models.
        '''
        if Lambda == None:
            Lambda = np.quantile(self.self_tuning(tau, standardize), 0.9)
        if h == None: h = self.h(tau) 
        if weight not in self.weights[:3]:
            raise ValueError("weight distribution must be either Exponential, Rademacher or Multinomial")

        fit0 = self.irw(Lambda, tau, h, kernel, penalty=penalty, a=a, step=step, standardize=standardize, adjust=False)
        mb_beta = np.zeros(shape=(self.p+self.itcp, B+1))
        mb_beta[:,0] = fit0[0]
        if standardize:
            mb_beta[self.itcp:,0] = mb_beta[self.itcp:,0]/self.sdX
            if self.itcp: mb_beta[0,0] -= self.mX.dot(mb_beta[1:,0])
 
        for b in range(B):
            boot_fit = self.irw(Lambda, tau, h, kernel, beta0=fit0[0], res=fit0[1][0], penalty=penalty, a=a, step=step,
                                standardize=standardize, weight=self.boot_weight(weight))
            mb_beta[:,b+1] = boot_fit[0]
        
        ## delete NaN bootstrap estimates (when using Gaussian weights)
        mb_beta = mb_beta[:,~np.isnan(mb_beta).any(axis=0)]
        
        ## Method 1: Majority vote among B bootstrap models
        selection_rate = 1 - np.mean(mb_beta[self.itcp:,1:]==0, axis=1)
        model_1 = np.where(selection_rate>0.5)[0]
        
        ## Method 2: Intersection of B bootstrap models
        model_2 = np.arange(self.p)
        for b in range(len(mb_beta[0,1:])):
            boot_model = np.where(abs(mb_beta[self.itcp:,b+1])>0)[0]
            model_2 = np.intersect1d(model_2, boot_model)

        return mb_beta, [model_1, model_2]



    def boot_inference(self, Lambda=None, tau=0.5, h=None, kernel="Laplacian", weight="Multinomial",
                        B=200, alpha=0.05, penalty="SCAD", a=3.7, step=5, standardize=True):
        '''
            Post-Selection-Inference via Bootstrap
        '''
        mb_beta, mb_model = self.boot_select(Lambda, tau, h, kernel, weight, B, alpha, penalty, a, step, standardize)
        
        mb_ci = np.zeros([3, self.p + self.itcp, 2])
        X_select = self.X[:, mb_model[0]+self.itcp]
        post_sqr = conquer(X_select, self.Y, self.itcp)
        post_boot = post_sqr.mb_ci(tau, kernel=kernel, weight=weight, alpha=alpha, standardize=standardize)
        mb_ci[:,mb_model[0]+self.itcp,:] = post_boot[1][:,self.itcp:,:]
        if self.itcp: mb_ci[:,0,:] = post_boot[1][:,0,:]

        return mb_beta, [mb_ci, mb_model[0], mb_model[1]]
    



## Use a validation set to choose lambda for penalized conquer
class val_conquer():
    penalties = ["L1", "SCAD", "MCP"]
    
    def __init__(self, X_train, Y_train, X_val, Y_val, intercept=True):
        self.n, self.p = X_train.shape
        self.X_train, self.Y_train = X_train, Y_train
        self.X_val, self.Y_val = X_val, Y_val 
        self.itcp = intercept
        
    def check(self, x, tau):
        '''
            Empirical Quantile Loss (check function)
        '''
        return np.mean(0.5*abs(x)+(tau-0.5)*x)
   
    def train(self, tau=0.5, h=None, Lambda_seq=np.array([]), L=20, kernel="Laplacian", penalty="SCAD", a=3.7, step=5, standardize=True, B=200):
        '''
        Arguments
        ----------
        Lambda_seq : a numpy array of Lambda values. If unspecified, it will be determined by the self_tuning() function in reg_conquer.

        L : number of Lambda values if unspecified. The default is 20.

        penalty : a character string representing one of the built-in penalties. The default is "SCAD".

        Returns
        -------
        val_beta : a numpy array of regression estiamtes. 
        
        list : a list of fitted redisual, number of nonzeros, selected Lambda value, a sequence of validation errors and its minimum.
        '''    
        sqr_train = reg_conquer(self.X_train, self.Y_train, intercept=self.itcp)
        if not Lambda_seq.any():
            lambda_max = np.quantile(sqr_train.self_tuning(tau, standardize, B), 0.99)
            Lambda_seq = np.linspace(0.5*lambda_max, lambda_max, L)
        else:
            L = len(Lambda_seq)
            
        if h == None: h = sqr_train.h(tau)

        if penalty not in self.penalties:
            raise ValueError("penalty must be either L1, SCAD or MCP")
        elif penalty == "L1":
            train_beta, train_fit = sqr_train.l1_path(Lambda_seq, tau, h, kernel, standardize)
        else:
            train_beta, train_fit = sqr_train.irw_path(Lambda_seq, tau, h, kernel, penalty, a, step, standardize)
        val_error = np.zeros(L)
        for l in range(L):
            val_error[l] = self.check(self.Y_val - train_beta[0,l]*self.itcp - self.X_val.dot(train_beta[self.itcp:,l]), tau)

        val_min = min(val_error)
        l_min = np.where(val_error == val_min)[0][0]
        val_beta, val_res = train_beta[:,l_min], train_fit[0][:,l_min]
        
        return val_beta, [val_res, sum((abs(val_beta[self.itcp:])>0)), Lambda_seq[l_min], val_error, val_min]