import numpy as np
import numpy.random as rgt
from scipy.stats import norm
from scipy.optimize import minimize
from conquer.linear_model import low_dim

from cvxopt import solvers, matrix
solvers.options['show_progress'] = False

class QuantES(low_dim):
    '''
        Joint Quantile and Expected Shortfall Regression    
    '''
    def _spec_func(self, type=1):
        '''
            Specification Functions in Fissler and Ziegel's Joint Loss 
        '''
        if type == 1:
            f0 = lambda x : -np.sqrt(-x)
            f1 = lambda x : 0.5 / np.sqrt(-x)
            f2 = lambda x : 0.25 / np.sqrt((-x)**3)
        elif type == 2:
            f0 = lambda x : -np.log(-x)
            f1 = lambda x : -1 / x
            f2 = lambda x : 1 / x ** 2
        elif type == 3:
            f0 = lambda x : -1 / x
            f1 = lambda x : 1 / x ** 2
            f2 = lambda x : -2 / x ** 3
        elif type == 4:
            f0 = lambda x : np.log( 1 + np.exp(x))
            f1 = lambda x : np.exp(x) / (1 + np.exp(x))
            f2 = lambda x : np.exp(x) / (1 + np.exp(x)) ** 2
        elif type == 5:
            f0 = lambda x : np.exp(x)
            f1 = lambda x : np.exp(x)
            f2 = lambda x : np.exp(x) 
        else:
            raise ValueError("type must be an integer between 1 and 5")

        return f0, f1, f2 


    def twostep(self, tau=0.5, h=None, kernel='Laplacian', 
                loss='L2', robust=None, type=1,
                standardize=True, tol=None, options=None,
                ci=False, level=0.95):
        '''
            Two-Step Procedure for Joint Quantile & Expected Shortfall Regression

        Reference
        ---------
        Higher Order Elicitability and Osband's Principle (2016)
        by Tobias Fissler and Johanna F. Ziegel
        Ann. Statist. 44(4): 1680-1707

        Effciently Weighted Estimation of Tail and Interquantile Expectations (2020)
        by Sander Barendse 
        SSRN Preprint
        
        Robust Estimation and Inference 
        for Expected Shortfall Regression with Many Regressors (2022)
        by Xuming He, Kean Ming Tan and Wen-Xin Zhou
        Preprint

        Inference for Joint Quantile and Expected Shortfall Regression (2022)
        by Xiang Peng and Judy Wang
        arXiv:2208.10586

        Arguments
        ---------        
        tau : quantile level; default is 0.5.

        h : bandwidth; the default value is computed by self.bandwidth(tau).
        
        kernel : a character string representing one of the built-in smoothing kernels; 
                 default is "Laplacian".

        loss : the loss function used in stage two. There are three options.
               1. 'L2': squared/L2 loss;
               2. 'Huber': Huber loss;
               3. 'FZ': Fissler and Ziegel's joint loss.
               
        robust : robustification parameter in the Huber loss; 
                 if robust=None, it will be automatically determined in a data-driven way.

        type : an integer (from 1 to 5) that corresponds to one of the 
               five specification functions in FZ's loss.

        tol : tolerance for termination.

        options : a dictionary of solver options; see https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html.
                  Default is options={'gtol': 1e-05, 'norm': inf, 'maxiter': None, 'disp': False, 'return_all': False}
                  gtol : gradient norm must be less than gtol(float) before successful termination.
                  norm : order of norm (Inf is max, -Inf is min).
                  maxiter : maximum number of iterations to perform.
                  disp : set to True to print convergence messages.
                  return_all : set to True to return a list of the best solution 
                               at each of the iterations.

        'ci' : logical flag for computing normal-based confidence intervals.

        'level' : confidence level between 0 and 1.

        Returns
        -------
        'coef_q' : quantile regression coefficient estimate.
            
        'res_q' : a vector of fitted quantile regression residuals.

        'coef_e' : expected shortfall regression coefficient estimate.

        'robust' : robustification parameter in the Huber loss.

        'ci' : coordinates-wise (100*level)% confidence intervals.

        '''
        if loss in {'L2', 'Huber', 'TrunL2', 'TrunHuber'}:
            qrfit = self.fit(tau=tau, h=h, kernel=kernel, standardize=standardize)
            nres_q = np.minimum(qrfit['res'], 0)
        
        if loss == 'L2':
            adj = np.linalg.solve(self.X.T.dot(self.X), self.X.T.dot(nres_q) / tau)
            coef_e = qrfit['beta'] + adj
            robust = None
        elif loss == 'Huber':
            Z = nres_q + tau*(self.Y - qrfit['res'])
            X0 = self.X[:, self.itcp:]
            if robust == None:
                esfit = low_dim(tau*X0, Z, intercept=self.itcp).adaHuber(standardize=standardize)
                coef_e = esfit['beta']
                robust = esfit['robust']
                if self.itcp: coef_e[0] /= tau
            elif robust > 0:
                esfit = low_dim(tau*X0, Z, intercept=self.itcp).retire(robust=robust,
                                                                       standardize=standardize,
                                                                       scale=False)
                coef_e = esfit['beta']
                robust = esfit['robust']
                if self.itcp: coef_e[0] /= tau
            else:
                raise ValueError("robustification parameter must be positive")
        elif loss == 'FZ':
            if type in np.arange(1,4):
                Ymax = np.max(self.Y)
                Y = self.Y - Ymax
            else:
                Y = self.Y
            qrfit = low_dim(self.X[:, self.itcp:], Y, intercept=True)\
                    .fit(tau=tau, h=h, kernel=kernel, standardize=standardize)
            adj = np.minimum(qrfit['res'], 0)/tau + Y - qrfit['res']
            f0, f1, f2 = self._spec_func(type)

            fun  = lambda z : np.mean(f1(self.X.dot(z)) * (self.X.dot(z) - adj) - f0(self.X.dot(z)))
            grad = lambda z : self.X.T.dot(f2(self.X.dot(z)) * (self.X.dot(z) - adj))/self.n
            esfit = minimize(fun, qrfit['beta'], method='BFGS', jac=grad, tol=tol, options=options)
            coef_e = esfit['x']
            robust = None
            if type in np.arange(1,4):
                coef_e[0] += Ymax
                qrfit['beta'][0] += Ymax
        elif loss == 'TrunL2':
            tail = self.Y <= self.X.dot(qrfit['beta'])
            tail_X = self.X[tail,:] 
            tail_Y = self.Y[tail]
            coef_e = np.linalg.solve(tail_X.T.dot(tail_X), tail_X.T.dot(tail_Y))
            robust = None
        elif loss == 'TrunHuber':
            tail = self.Y <= self.X.dot(qrfit['beta'])
            esfit = QuantES(self.X[tail, self.itcp:], self.Y[tail], \
                            intercept=self.itcp).adaHuber(standardize=standardize)
            coef_e = esfit['beta']
            robust = esfit['robust']

        if loss in {'L2', 'Huber'} and ci:
            res_e = nres_q + tau * self.X.dot(qrfit['beta'] - coef_e)
            n, p = self.X[:,self.itcp:].shape
            X0 = np.c_[np.ones(n,), self.X[:,self.itcp:] - self.mX]
            if loss == 'L2':
                weight = res_e ** 2
            else:
                weight = np.minimum(res_e ** 2, robust ** 2)
    
            inv_sig = np.linalg.inv(X0.T.dot(X0) / n)   
            acov = inv_sig.dot((X0.T * weight).dot(X0) / n).dot(inv_sig)
            radius = norm.ppf(1/2 + level/2) * np.sqrt( np.diag(acov) / n) / tau
            ci = np.c_[coef_e - radius, coef_e + radius]

        return {'coef_q': qrfit['beta'], 'res_q': qrfit['res'], 
                'coef_e': coef_e,
                'loss': loss, 'robust': robust,
                'ci': ci, 'level': level}


    def boot_es(self, tau=0.5, h=None, kernel='Laplacian', 
                loss='L2', robust=None, standardize=True, 
                B=200, level=0.95):

        fit = self.twostep(tau, h, kernel, loss, robust, standardize=standardize)
        boot_coef = np.zeros((self.X.shape[1], B))
        for b in range(B):
            idx = rgt.choice(np.arange(self.n), size=self.n)
            boot = QuantES(self.X[idx,self.itcp:], self.Y[idx], intercept=self.itcp)
            if loss == 'L2':
                bfit = boot.twostep(tau, h, kernel, loss='L2', standardize=standardize)
            else:
                bfit = boot.twostep(tau, h, kernel, loss, \
                                    robust=fit['robust'], standardize=standardize)
            boot_coef[:,b] = bfit['coef_e']
        
        left  = np.quantile(boot_coef, 1/2-level/2, axis=1)
        right = np.quantile(boot_coef, 1/2+level/2, axis=1)
        piv_ci = np.c_[2*fit['coef_e'] - right, 2*fit['coef_e'] - left]
        per_ci = np.c_[left, right]

        return {'coef_q': fit['coef_q'],
                'coef_e': fit['coef_e'],
                'boot_coef_e': boot_coef,
                'loss': loss, 'robust': fit['robust'], 
                'piv_ci': piv_ci, 'per_ci': per_ci, 'level': level}


    def nc_twostep(self, tau=0.5, h=None, kernel='Laplacian', 
                   loss='L2', robust=None, standardize=True, 
                   ci=False, level=0.95):
        '''
            Non-Crossing Two-Step Joint Quantile & Expected Shortfall Regression

        Reference
        ---------
        Robust Estimation and Inference 
        for Expected Shortfall Regression with Many Regressors (2022)
        by Xuming He, Kean Ming Tan and Wen-Xin Zhou
        Preprint

        '''

        qrfit = self.fit(tau=tau, h=h, kernel=kernel, standardize=standardize)
        nres_q = np.minimum(qrfit['res'], 0)
        fitted_q = self.Y - qrfit['res']
        Z = nres_q/tau + fitted_q
 
        P = matrix(self.X.T.dot(self.X) / self.n)
        q = matrix(-self.X.T.dot(Z) / self.n)
        G = matrix(self.X)
        hh = matrix(fitted_q)
        l, c = 0, robust 
        
        if loss == 'L2':
            esfit = solvers.qp(P, q, G, hh, initvals={'x': matrix(qrfit['beta'])})
            coef_e = np.array(esfit['x']).reshape(self.X.shape[1],)
        else:
            rel = (self.X.shape[1] + np.log(self.n)) / self.n
            esfit = self.twostep(tau, h, kernel, loss, robust, standardize=standardize)
            coef_e = esfit['coef_e']
            res  = np.abs(Z - self.X.dot(coef_e))
            c = robust
            
            if robust == None:
                c = self._find_root(lambda t : np.mean(np.minimum((res/t)**2, 1)) - rel, 
                                    np.min(res), np.sum(res ** 2))

            sol_diff = 1
            while l < self.opt['max_iter'] and sol_diff > self.opt['tol']:
                wt = np.where(res > c, res/c, 1)
                P = matrix( (self.X.T / wt ).dot(self.X) / self.n)
                q = matrix( -self.X.T.dot(Z / wt) / self.n)
                esfit = solvers.qp(P, q, G, hh, initvals={'x': matrix(coef_e)})
                tmp = np.array(esfit['x']).reshape(self.X.shape[1],)
                sol_diff = np.max(np.abs(tmp - coef_e))
                res = np.abs(Z - self.X.dot(tmp))
                if robust == None:
                    c = self._find_root(lambda t : np.mean(np.minimum((res/t)**2, 1)) - rel, 
                                        np.min(res), np.sum(res ** 2))
                coef_e = tmp
                l += 1
            c *= tau

        if ci:
            res_e = nres_q + tau * (fitted_q - self.X.dot(coef_e))
            X0 = np.c_[np.ones(self.n,), self.X[:,self.itcp:] - self.mX]
            if loss == 'L2': weight = res_e ** 2
            else: weight = np.minimum(res_e ** 2, c ** 2)
    
            inv_sig = np.linalg.inv(X0.T.dot(X0) / self.n)   
            acov = inv_sig.dot((X0.T * weight).dot(X0) / self.n).dot(inv_sig)
            radius = norm.ppf(1/2 + level/2) * np.sqrt( np.diag(acov) / self.n) / tau
            ci = np.c_[coef_e - radius, coef_e + radius]

        return {'coef_q': qrfit['beta'], 
                'res_q': qrfit['res'], 
                'coef_e': coef_e, 'iter': l,
                'loss': loss, 'robust': c,
                'ci': ci, 'level': level}
