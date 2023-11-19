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
    def _G2(self, G2_type=1):
        '''
            Specification Function G2 in Fissler and Ziegel's Joint Loss 
        '''
        if G2_type == 1:
            f0 = lambda x : -np.sqrt(-x)
            f1 = lambda x : 0.5 / np.sqrt(-x)
            f2 = lambda x : 0.25 / np.sqrt((-x)**3)
        elif G2_type == 2:
            f0 = lambda x : -np.log(-x)
            f1 = lambda x : -1 / x
            f2 = lambda x : 1 / x ** 2
        elif G2_type == 3:
            f0 = lambda x : -1 / x
            f1 = lambda x : 1 / x ** 2
            f2 = lambda x : -2 / x ** 3
        elif G2_type == 4:
            f0 = lambda x : np.log( 1 + np.exp(x))
            f1 = lambda x : np.exp(x) / (1 + np.exp(x))
            f2 = lambda x : np.exp(x) / (1 + np.exp(x)) ** 2
        elif G2_type == 5:
            f0 = lambda x : np.exp(x)
            f1 = lambda x : np.exp(x)
            f2 = lambda x : np.exp(x) 
        else:
            raise ValueError("G2_type must be an integer between 1 and 5")

        return f0, f1, f2 


    def _FZ_loss(self, x, tau, G1=False, G2_type=1):
        '''
            Fissler and Ziegel's Joint Loss Function

        Arguments
        ---------        
        G1 : logical flag for the specification function G1 in FZ's loss. 
             G1(x)=0 if G1=False, and G1(x)=x and G1=True.

        G2_type : an integer (from 1 to 5) that indicates the type 
                  of the specification function G2 in FZ's loss.
        '''
        X = self.X
        if G2_type in {1, 2, 3}:
            Ymax = np.max(self.Y)
            Y = self.Y - Ymax
        else:
            Y = self.Y
        dim = X.shape[1]
        Yq = X @ x[:dim]
        Ye = X @ x[dim : 2*dim]
        f0, f1, _ = self._G2(G2_type)
        loss = f1(Ye) * (Ye - Yq - (Y - Yq) * (Y<= Yq) / tau) - f0(Ye)
        if G1:
            return np.mean((tau - (Y<=Yq)) * (Y-Yq) + loss)
        else:
            return np.mean(loss)


    def joint_fit(self, tau=0.5, G1=False, G2_type=1, 
                  standardize=True, refit=True, tol=None, 
                  options={'maxiter': None, 'maxfev': None, 'disp': False, 
                           'return_all': False, 'initial_simplex': None, 
                           'xatol': 0.0001, 'fatol': 0.0001, 'adaptive': False}):
        '''
            Joint Quantile & Expected Shortfall Regression via FZ Loss Minimization

        Reference
        ---------
        Higher Order Elicitability and Osband's Principle (2016)
        by Tobias Fissler and Johanna F. Ziegel
        Ann. Statist. 44(4): 1680-1707

        A Joint Quantile and Expected Shortfall Regression Framework (2019)
        by Timo Dimitriadis and Sebastian Bayer 
        Electron. J. Stat. 13(1): 1823-1871
        
        Arguments
        ---------        
        tau : quantile level; default is 0.5.

        G1 : logical flag for the specification function G1 in FZ's loss. 
             G1(x)=0 if G1=False, and G1(x)=x and G1=True.

        G2_type : an integer (from 1 to 5) that indicates the type of the specification function G2 in FZ's loss.
        
        standardize : logical flag for x variable standardization prior to fitting the quantile model; 
                      default is TRUE.
        
        refit : logical flag for refitting joint regression if the optimization is terminated early;
                default is TRUE.

        tol : tolerance for termination.

        options : a dictionary of solver options; 
                  see https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html

        Returns
        -------
        'coef_q' : quantile regression coefficient estimate.
            
        'coef_e' : expected shortfall regression coefficient estimate.

        'nit' : total number of iterations. 

        'nfev' : total number of function evaluations.

        'message' : a message that describes the cause of the termination.

        '''

        dim = self.X.shape[1]
        Ymax = np.max(self.Y)
        ### warm start with QR + truncated least squares
        qrfit = low_dim(self.X[:, self.itcp:], self.Y, intercept=True)\
                .fit(tau=tau, standardize=standardize)
        coef_q = qrfit['beta']
        tail = self.Y <= self.X.dot(coef_q)
        tail_X = self.X[tail,:] 
        tail_Y = self.Y[tail]
        coef_e = np.linalg.solve(tail_X.T.dot(tail_X), tail_X.T.dot(tail_Y))
        if G2_type in {1, 2, 3}:
            coef_q[0] -= Ymax
            coef_e[0] -= Ymax
        x0 = np.r_[(coef_q, coef_e)]

        ### joint quantile and ES fit
        fun  = lambda x : self._FZ_loss(x, tau, G1, G2_type)
        esfit = minimize(fun, x0, method='Nelder-Mead', tol=tol, options=options)
        nit, nfev = esfit['nit'], esfit['nfev']

        ### refit if convergence criterion is not met
        while refit and not esfit['success']:
            esfit = minimize(fun, esfit['x'], method='Nelder-Mead',
                             tol=tol, options=options)
            nit += esfit['nit']
            nfev += esfit['nfev']

        coef_q, coef_e = esfit['x'][:dim], esfit['x'][dim : 2*dim]
        if G2_type in {1, 2, 3}:
            coef_q[0] += Ymax
            coef_e[0] += Ymax

        return {'coef_q': coef_q, 'coef_e': coef_e,
                'nit': nit, 'nfev': nfev,
                'success': esfit['success'],
                'message': esfit['message']}


    def twostep_fit(self, tau=0.5, h=None, kernel='Laplacian', 
                    loss='L2', robust=None, G2_type=1,
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
        for Expected Shortfall Regression with Many Regressors (2023)
        by Xuming He, Kean Ming Tan and Wen-Xin Zhou
        J. R. Stat. Soc. B. 85(4): 1223-1246

        Inference for Joint Quantile and Expected Shortfall Regression (2023)
        by Xiang Peng and Judy Wang
        Stat 12(1) e619

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

        G2_type : an integer (from 1 to 5) that indicates the type of the specification function G2 in FZ's loss.

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
                esfit = low_dim(tau*X0, Z, intercept=self.itcp).adaHuber()
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
            if G2_type in np.arange(1,4):
                Ymax = np.max(self.Y)
                Y = self.Y - Ymax
            else:
                Y = self.Y
            qrfit = low_dim(self.X[:, self.itcp:], Y, intercept=True)\
                    .fit(tau=tau, h=h, kernel=kernel, standardize=standardize)
            adj = np.minimum(qrfit['res'], 0)/tau + Y - qrfit['res']
            f0, f1, f2 = self._G2(G2_type)

            fun  = lambda z : np.mean(f1(self.X.dot(z)) * (self.X.dot(z) - adj) - f0(self.X.dot(z)))
            grad = lambda z : self.X.T.dot(f2(self.X.dot(z)) * (self.X.dot(z) - adj))/self.n
            esfit = minimize(fun, qrfit['beta'], method='BFGS', jac=grad, tol=tol, options=options)
            coef_e = esfit['x']
            robust = None
            if G2_type in np.arange(1,4):
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
                            intercept=self.itcp).adaHuber()
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

        fit = self.twostep_fit(tau, h, kernel, loss, robust, standardize=standardize)
        boot_coef = np.zeros((self.X.shape[1], B))
        for b in range(B):
            idx = rgt.choice(np.arange(self.n), size=self.n)
            boot = QuantES(self.X[idx,self.itcp:], self.Y[idx], intercept=self.itcp)
            if loss == 'L2':
                bfit = boot.twostep_fit(tau, h, kernel, loss='L2', standardize=standardize)
            else:
                bfit = boot.twostep_fit(tau, h, kernel, loss, \
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


    def nc_fit(self, tau=0.5, h=None, kernel='Laplacian', 
               loss='L2', robust=None, standardize=True, 
               ci=False, level=0.95):
        '''
            Non-Crossing Joint Quantile & Expected Shortfall Regression

        Reference
        ---------
        Robust Estimation and Inference  
        for Expected Shortfall Regression with Many Regressors (2023)
        by Xuming He, Kean Ming Tan and Wen-Xin Zhou
        J. R. Stat. Soc. B. 85(4): 1223-1246
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
            esfit = self.twostep_fit(tau, h, kernel, loss, robust, standardize=standardize)
            coef_e = esfit['coef_e']
            res  = np.abs(Z - self.X.dot(coef_e))
            c = robust
            
            if robust == None:
                c = self._find_root(lambda t : np.mean(np.minimum((res/t)**2, 1)) - rel, 
                                    np.min(res)+self.opt['tol'], np.sqrt(res @ res))

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
                                        np.min(res)+self.opt['tol'], np.sqrt(res @ res))
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
                'coef_e': coef_e, 'nit': l,
                'loss': loss, 'robust': c,
                'ci': ci, 'level': level}
