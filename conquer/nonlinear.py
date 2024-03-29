import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.sparse import csc_matrix
from sklearn.metrics import pairwise_kernels as PK
from sklearn.kernel_ridge import KernelRidge as KR

# https://qpsolvers.github.io/qpsolvers/quadratic-programming.html#qpsolvers.solve_qp
# https://qpsolvers.github.io/qpsolvers/supported-solvers.html#supported-solvers
from qpsolvers import solve_qp


###############################################################################
########################## Kernel Ridge Regression ############################
###############################################################################
class KRR:
    '''
    Kernel Ridge Regression
    
    Methods:
        __init__(): Initialize the KRR object
        qt(): Fit (smoothed) quantile kernel ridge regression
        es(): Fit (robust) expected shortfall kernel ridge regression
        qt_seq(): Fit a sequence of quantile kernel ridge regressions
        qt_predict(): Compute predicted quantile at test data
        es_predict(): Compute predicted expected shortfall at test data
        qt_loss(): Check or smoothed check loss
        qt_sg(): Compute the (sub)derivative of the (smoothed) check loss
        bw(): Compute the bandwidth (smoothing parameter)
        genK(): Generate the kernel matrix for test data

    Attributes:
        params (dict): a dictionary of kernel parameters;
            gamma (float), default is 1;
            coef0 (float), default is 1;
            degree (int), default is 3.
            rbf : exp(-gamma*||x-y||_2^2)
            polynomial : (gamma*<x,y> + coef0)^degree
            laplacian : exp(-gamma*||x-y||_1)
    '''
    n_jobs = None
    min_bandwidth = 1e-4
    params = {'gamma': 1, 'coef0': 1, 'degree': 3}
    
    def __init__(self, X, Y, normalization=None, 
                 kernel='rbf', kernel_params=dict()):
        ''' 
        Initialize the KRR object

        Args:
            X (ndarray): n by p matrix of covariates;
                         n is the sample size, p is the number of covariates.
            Y (ndarray): response/target variable.
            normalization (str): method for normalizing covariates;
                                 should be one of [None, 'MinMax', 'Z-score'].
            kernel (str): type of kernel function; 
                          choose one from ['rbf', 'polynomial', 'laplacian'].
            kernel_params (dict): a dictionary of user-specified kernel parameters; 
                                  default is in the class attribute.

        Attributes:
            n (int) : number of observations
            Y (ndarray) : target variable
            nm (str) : method for normalizing covariates
            kernel (str) : type of kernel function
            params (dict) : a dictionary of kernel parameters
            X0 (ndarray) : normalized covariates
            xmin (ndarray) : minimum values of the original covariates
            xmax (ndarray) : maximum values of the original covariates
            xm (ndarray) : mean values of the original covariates
            xsd (ndarray) : standard deviations of the original covariates
            K (ndarray) : n by n kernel matrix
        '''

        self.n = X.shape[0]
        self.Y = Y.reshape(self.n)
        self.nm = normalization
        self.kernel = kernel
        self.params.update(kernel_params)

        if normalization is None:
            self.X0 = X[:]
        elif normalization == 'MinMax':
            self.xmin = np.min(X, axis=0)
            self.xmax = np.max(X, axis=0)
            self.X0 = (X[:] - self.xmin)/(self.xmax - self.xmin)
        elif normalization == 'Z-score':
            self.xm, self.xsd = np.mean(X, axis=0), np.std(X, axis=0)
            self.X0 = (X[:] - self.xm)/self.xsd
        
        # compute the kernel matrix
        self.K = PK(self.X0, metric=kernel, filter_params=True,
                    n_jobs=self.n_jobs, **self.params)


    def genK(self, x):
        ''' Generate the kernel matrix for test data '''
        if np.ndim(x) == 1:
            x = x.reshape(1, -1)
        if self.nm == 'MinMax':
            x = (x - self.xmin)/(self.xmax - self.xmin)
        elif self.nm == 'Z-score':
            x = (x - self.xm)/self.xsd
        
        # return an n * m matrix, m is test data size
        return PK(self.X0, x, metric=self.kernel, 
                  filter_params=True, n_jobs=self.n_jobs, 
                  **self.params)


    def qt_loss(self, x, h=0):
        '''
        Check or smoothed check loss
        '''
        if h == 0:
            out = np.where(x > 0, self.tau * x, (self.tau - 1) * x)
        else:
            out = (self.tau - norm.cdf(-x/h)) * x \
                  + (h/2) * np.sqrt(2/np.pi) * np.exp(-(x/h) ** 2 /2)
        return np.sum(out)

    
    def qt_sg(self, x, h=0):
        '''
        Gradient/subgradient of the (smoothed) check loss
        '''
        if h == 0:
            return np.where(x >= 0, self.tau, self.tau - 1)
        else:
            return self.tau - norm.cdf(-x / h)
        
        
    def bw(self, exponent=1/3):
        '''
        Compute the bandwidth (smoothing parameter)

        Args: 
            exponent (float): the exponent in the formula; default is 1/3.
        '''
        krr = KR(alpha = 1, kernel=self.kernel,
                 gamma = self.params['gamma'],
                 degree = self.params['degree'],
                 coef0 = self.params['coef0'])
        krr.fit(self.X0, self.Y)
        krr_res = self.Y - krr.predict(self.X0)
        return max(self.min_bandwidth, 
                   np.std(krr_res)/self.n ** exponent)


    def qt(self, tau=0.5, alpha_q=1, 
           init=None, intercept=True, 
           smooth=False, h=0., 
           method='L-BFGS-B', solver='clarabel', 
           tol=1e-8, options=None):
        '''
        Fit (smoothed) quantile kernel ridge regression

        Args:
            tau (float): quantile level between 0 and 1; default is 0.5.
            alpha_q (float): regularization parameter; default is 1.
            init (ndarray): initial values for optimization; default is None.
            intercept (bool): whether to include intercept term; 
                              default is True.
            smooth (bool): a logical flag for using smoothed check loss; 
                           default is FALSE.
            h (float): bandwidth for smoothing; default is 0.
            method (str): type of solver if smoothing (h>0) is used;
                          choose one from ['BFGS', 'L-BFGS-B'].
            solver (str): type of QP solver if check loss is used; 
                          default is 'clarabel'.
            tol (float): tolerance for termination; default is 1e-8.
            options (dict): a dictionary of solver options; default is None.
        
        Attributes:
            qt_sol (OptimizeResult): solution of the optimization problem
            qt_beta (ndarray): quantile KRR coefficients
            qt_fit (ndarray): fitted quantiles (in-sample)
        '''
        self.alpha_q, self.tau, self.itcp_q = alpha_q, tau, intercept
        if smooth and h == 0: 
            h = self.bw()
        n, self.h = self.n, h

        # compute smoothed quantile KRR estimator with bandwidth h
        if self.h > 0: 
            if intercept:
                x0 = init if init is not None else np.zeros(n + 1)
                x0[0] = np.quantile(self.Y, tau)
                res = lambda x: self.Y - x[0] - self.K @ x[1:]
                func = lambda x: self.qt_loss(res(x),h) + \
                                 (alpha_q/2) * np.dot(x[1:], self.K @ x[1:])
                grad = lambda x: np.insert(-self.K @ self.qt_sg(res(x),h) 
                                           + alpha_q*self.K @ x[1:], 
                                           0, np.sum(-self.qt_sg(res(x),h)))
                self.qt_sol = minimize(func, x0, method=method, 
                                       jac=grad, tol=tol, options=options)
                self.qt_beta = self.qt_sol.x
                self.qt_fit = self.qt_beta[0] + self.K @ self.qt_beta[1:]
            else:
                x0 = init if init is not None else np.zeros(n)
                res = lambda x: self.Y - self.K @ x
                func = lambda x: self.qt_loss(res(x), h) \
                                 + (alpha_q/2) * np.dot(x, self.K @ x)
                grad = lambda x: -self.K @ self.qt_sg(res(x), h) \
                                 + alpha_q * self.K @ x
                self.qt_sol = minimize(func, x0=x0, method=method, 
                                       jac=grad, tol=tol, options=options)
                self.qt_beta = self.qt_sol.x
                self.qt_fit = self.K @ self.qt_beta
        else: 
            # fit quantile KRR by solving a quadratic program
            C = 1 / alpha_q
            lb = C * (tau - 1)
            ub = C * tau
            x = solve_qp(P=csc_matrix(self.K), q=-self.Y, G=None, 
                         h=None, A=csc_matrix(np.ones(n)),
                         b=np.array([0.]), lb=lb * np.ones(n), 
                         ub=ub * np.ones(n), solver=solver)
            self.itcp_q = True
            self.qt_fit = self.K @ x
            b = np.quantile(self.Y - self.qt_fit, tau)
            self.qt_beta = np.insert(x, 0, b)
            self.qt_fit += b


    def qt_seq(self, tau=0.5, alphaseq=np.array([0.1]), 
               intercept=True, smooth=False, h=0., 
               method='L-BFGS-B', solver='clarabel', 
               tol=1e-8, options=None):
        '''
        Fit a sequence of (smoothed) quantile kernel ridge regressions
        '''
        alphaseq = np.sort(alphaseq)[::-1]
        args = [intercept, smooth, h, method, solver, tol, options]

        x0 = None
        x, fit = [], []
        for alpha_q in alphaseq:
            self.qt(tau, alpha_q, x0, *args)
            x.append(self.qt_beta)
            fit.append(self.qt_fit)
            x0 = self.qt_beta

        self.qt_beta = np.array(x).T
        self.qt_fit = np.array(fit).T
        self.alpha_q = alphaseq


    def qt_predict(self, x): 
        '''
        Compute predicted quantile at new input x
        
        Args:
            x (ndarray): new input.
        '''
        return self.itcp_q*self.qt_beta[0] + \
               self.qt_beta[self.itcp_q:] @ self.genK(x)


    def es(self, tau=0.5, alpha_q=1, alpha_e=1, 
           init=None, intercept=True, 
           qt_fit=None, smooth=False, h=0., 
           method='L-BFGS-B', solver='clarabel',
           robust=False, c=None, 
           qt_tol=1e-8, es_tol=1e-10, options=None):
        """ 
        Fit (robust) expected shortfall kernel ridge regression
        
        Args:
            tau (float): quantile level between 0 and 1; default is 0.5.
            alpha_t, alpha_e (float): regularization parameters; default is 1.
            init (ndarray): initial values for optimization; default is None.
            intercept (bool): whether to include intercept term; 
                              default is True.
            qt_fit (ndarray): fitted quantiles from the first step; 
                              default is None.
            smooth (bool): a logical flag for using smoothed check loss; 
                           default is FALSE.
            h (float): bandwidth for smoothing; default is 0.
            method (str): type of solver if smoothing (h>0) is used;
                          choose one from ['BFGS', 'L-BFGS-B'].
            solver (str): type of QP solver if check loss is used; 
                          default is 'clarabel'.
            robust (bool): whether to use the Huber loss in the second step; 
                           default is False.
            c (float): positive tuning parameter for the Huber estimator; 
                       default is None.
            qt_tol (float): tolerance for termination in qt-KRR; 
                            default is 1e-8.
            es_tol (float): tolerance for termination in es-KRR; 
                            default is 1e-10.
            options (dict): a dictionary of solver options; default is None.
    
        Attributes:
            es_sol (OptimizeResult): solution of the optimization problem
            es_beta (ndarray): expected shortfall KRR coefficients
            es_fit (ndarray): fitted expected shortfalls (in-sample)
            es_pred (ndarray): predicted expected shortfall at new input
        """
        if qt_fit is None:
            self.qt(tau, alpha_q, None, intercept, smooth, h, 
                    method, solver, qt_tol, options)
            qt_fit = self.qt_fit
        elif len(qt_fit) != self.n:
            raise ValueError("Length of qt_fit should be equal to \
                              the number of observations.")
        
        self.alpha_e, self.tau, self.itcp = alpha_e, tau, intercept
        n = self.n
        
        qt_nres = np.minimum(self.Y - qt_fit, 0)
        if robust == True and c is None:
            c = np.std(qt_nres) * (n/np.log(n))**(1/3) / tau
        self.c = c
        # surrogate response
        Z = qt_nres/tau + qt_fit
        if intercept:
            x0 = init if init is not None else np.zeros(n + 1)
            x0[0] = np.mean(Z)
            res = lambda x: Z - x[0] - self.K @ x[1:]
            func = lambda x: huber_loss(res(x), c) + \
                             (alpha_e/2) * np.dot(x[1:], self.K @ x[1:])
            grad = lambda x: np.insert(-self.K @ huber_grad(res(x), c)
                                       + alpha_e * self.K @ x[1:],
                                       0, -np.sum(huber_grad(res(x), c)))
            self.es_sol = minimize(func, x0, method=method, 
                                   jac=grad, tol=es_tol, options=options)
            self.es_beta = self.es_sol.x
            self.es_fit = self.es_beta[0] + self.K @ self.es_beta[1:]
        else:
            x0 = init if init is not None else np.zeros(n)
            res = lambda x: Z - self.K @ x
            func = lambda x: huber_loss(res(x), c) \
                             + (alpha_e/2) * np.dot(x, self.K @ x)
            grad = lambda x: -self.K @ huber_grad(res(x), c)  \
                             + alpha_e * self.K @ x
            self.es_sol = minimize(func, x0=x0, method=method, 
                                   jac=grad, tol=es_tol, options=options)
            self.es_beta = self.es_sol.x
            self.es_fit = self.K @ self.es_beta
        self.es_residual = Z - self.es_fit
        self.es_model = None


    def lses(self, tau=0.5, 
             alpha_q=1, alpha_e=1,
             smooth=False, h=0., 
             method='L-BFGS-B', solver='clarabel',
             qt_tol=1e-8, options=None):
        
        self.alpha_e, self.tau, self.itcp = alpha_e, tau, True
        self.qt(tau, alpha_q, None, True,
                smooth, h, method, solver, qt_tol, options) 
        n = self.n
        Z = np.minimum(self.Y - self.qt_fit, 0)/tau + self.qt_fit
        self.es_model = KR(alpha = alpha_e, kernel=self.kernel, 
                           gamma = self.params['gamma'],
                           degree = self.params['degree'],
                           coef0 = self.params['coef0'])
        self.es_model.fit(self.X0, Z)
        self.es_fit = self.es_model.predict(self.X0)
        self.es_residual = Z - self.es_fit
        self.es_beta = None


    def es_predict(self, x): 
        '''
        Compute predicted expected shortfall at new input x
        
        Args:
            x (ndarray): new input.
        '''

        if self.es_beta is not None:
            return self.itcp*self.es_beta[0] \
                   + self.es_beta[self.itcp:] @ self.genK(x)
        elif self.es_model is not None:
            if self.nm == 'MinMax':
                x = (x - self.xmin)/(self.xmax - self.xmin)
            elif self.nm == 'Z-score':
                x = (x - self.xm)/self.xsd
            return self.es_model.predict(x)

    
    def bahadur(self, x):
        '''
        Compute Bahadur representation of the expected shortfall estimator
        '''
        if np.ndim(x) == 1:
            x = x.reshape(1, -1)
        if self.nm == 'MinMax':
            x = (x - self.xmin)/(self.xmax - self.xmin)
        elif self.nm == 'Z-score':
            x = (x - self.xm)/self.xsd
        
        A = self.K/self.n + self.alpha_e * np.eye(self.n)
        return np.linalg.solve(A, self.genK(x)) \
               * self.es_residual.reshape(-1,1)


def huber_loss(u, c=None):
    ''' Huber loss '''
    if c is None:
        out = 0.5 * u ** 2
    else:
        out = np.where(abs(u)<=c, 0.5*u**2, c*abs(u) - 0.5*c**2)
    return np.sum(out)


def huber_grad(u, c=None):
    ''' Gradient of Huber loss '''
    if c is None:
        return u    
    else:
        return np.where(abs(u)<=c, u, c*np.sign(u))



###############################################################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
from enum import Enum
from sklearn.utils import shuffle


###############################################################################
######################### Neural Network Regression ###########################
###############################################################################
class ANN:
    '''
    Artificial Neural Network Regression
    '''
    optimizers = ["sgd", "adam"]
    activations = ["sigmoid", "tanh", "relu", "leakyrelu"]
    opt = {'batch_size' : 64, 'val_pct' : 0.1, 'step_size': 10,
           'activation' : 'relu', 'depth': 4, 'width': 256,  
           'optimizer' : 'adam', 'lr': 1e-3, 'lr_decay': 1., 'nepochs' : 600,
           'dropout_proportion': 0.1, 'Lambda': 0, 'weight_decay': 0.0, 
           'momentum': 0.9, 'nesterov': True}
    
    def __init__(self, X, Y, normalization=None):
        '''
        Args:
            X (ndarry): n by p matrix of covariates; 
                        n is the sample size, p is the number of covariates.
            Y (ndarry): response/target variable.
            normalization (str): method for normalizing covariates;
                                 should be one of [None, 'MinMax', 'Z-score'].

        Attributes:
            Y (ndarray): response variable.
            X0 (ndarray): normalized covariates.
            n (int): sample size.
            nm (str): method for normalizing covariates.
        '''
        self.n = X.shape[0]
        self.Y = Y.reshape(self.n)
        self.nm = normalization

        if self.nm is None:
            self.X0 = X
        elif self.nm == 'MinMax':
            self.xmin = np.min(X, axis=0)
            self.xmax = np.max(X, axis=0)
            self.X0 = (X - self.xmin)/(self.xmax - self.xmin)
        elif self.nm == 'Z-score':
            self.xm, self.xsd = np.mean(X, axis=0), np.std(X, axis=0)
            self.X0 = (X - self.xm)/self.xsd

    
    def qt(self, tau=0.5, smooth=False, h=0., 
           options=dict(), plot=False, device='cpu'):
        '''
        Fit (smoothed) quantile neural network regression

        Args: 
            tau (float): quantile level between 0 and 1; default is 0.5.
            smooth (boolean): a logical flag for using smoothed check loss; default is FALSE.
            h (float): bandwidth for smoothing; default is 0.
            options (dictionary): a dictionary of neural network and optimization parameters.
                batch_size (int): the number of training examples used in one iteration; 
                                  default is 64.
                val_pct (float): the proportion of the training data to use for validation;
                                 default is 0.1.
                step_size (int): the number of epochs of learning rate decay; default is 10.
                activation (string): activation function; default is the ReLU function.
                depth (int): the number of hidden layers; default is 4.
                width (int): the number of neurons for each layer; default is 256.
                optimizer (string): the optimization algorithm; default is the Adam optimizer.
                lr (float): , learning rate of SGD or Adam optimization; default is 1e-3.
                lr_decay (float): multiplicative factor by which the learning rate will be reduced;
                                  default is 0.95.
                nepochs (int): the number of training epochs; default is 600.
                dropout_proportion : proportion of the dropout; default is 0.1.
                Lambda (float): L_1-regularization parameter; default is 0.
                weight_decay (float): weight decay of L2 penalty; default is 0.
                momentum (float): momentum accerleration rate for SGD algorithm; 
                                  default is 0.9.
                nesterov (boolean): whether to use Nesterov gradient descent algorithm;
                                    default is TRUE.
            plot (boolean) : whether to plot loss values at iterations.
            device (string): device to run the model on; default is 'cpu'.
        '''
        self.opt.update(options)
        self.device = device
        if smooth and h == 0:
            h = max(0.01, (tau - tau**2)**0.5 / self.n ** (1/3))
        self.h = h if smooth else 0.

        # (smoothed) check loss
        def qtloss(y_pred, y):
            z = y - y_pred
            if not smooth:
                return torch.max((tau - 1) * z, tau * z).mean()
            else:
                tmp = .5 * h * torch.sqrt(2/torch.tensor(np.pi))
                return torch.add((tau - Normal(0, 1).cdf(-z/h)) * z, 
                                  tmp * torch.exp(-(z/h)**2/2)).mean()
        
        out = self.trainer(self.X0, self.Y, qtloss, device)

        # plot loss
        if plot:
            plt.plot(out['train_losses'][1:out['epoch']], label='Train Loss')
            if self.opt['val_pct'] > 0:
                plt.plot(out['val_losses'][1:out['epoch']], 
                         label='Validation Loss')
            plt.legend()
            plt.show()
        
        self.qt_model = out['model']
        self.qt_fit = out['fit']
        

    def qt_predict(self, X):
        ''' Compute predicted quantile at new input X '''
        if self.nm == 'MinMax':
            X = (X - self.xmin)/(self.xmax - self.xmin)
        elif self.nm == 'Z-score':
            X = (X - self.xm)/self.xsd
        Xnew = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.qt_pred = self.qt_model.predict(Xnew)

    
    def es(self, tau=0.5, robust=False, c=None, 
           qt_fit=None, smooth=False, h=0., 
           options=dict(), plot=False, device='cpu'):
        '''
        Fit (robust) expected shortfall neural network regression
        '''
        self.opt.update(options)
        self.tau = tau
        if qt_fit is None:
            self.qt(tau=tau, smooth=smooth, h=h, plot=False, device=device)
            qt_fit = self.qt_fit
        elif len(qt_fit) != self.n:
            raise ValueError("Length of qt_fit should be equal to \
                             the number of observations.")

        qt_nres = np.minimum(self.Y - qt_fit, 0)
        if robust == True and c is None:
            c = np.std(qt_nres) * (self.n/np.log(self.n))**(1/3) / tau
        self.c = c

        # surrogate response
        Z = qt_nres / tau + qt_fit

        # L2/Huber loss
        if not robust:
            esloss = nn.MSELoss(reduction='mean')
        else:
            esloss = nn.HuberLoss(reduction='mean', delta=c)
            
        out = self.trainer(self.X0, Z, esloss, device)

        # plot loss
        if plot:
            plt.plot(out['train_losses'][1:out['epoch']], label='Train Loss')
            if self.opt['val_pct'] > 0:
                plt.plot(out['val_losses'][1:out['epoch']], 
                         label='Validation Loss')
            plt.legend()
            plt.show()
        
        self.es_model = out['model']
        self.es_fit = out['fit']


    def es_predict(self, X):
        ''' Compute predicted expected shortfall at new input X '''
        if self.nm == 'MinMax':
            X = (X - self.xmin)/(self.xmax - self.xmin)
        elif self.nm == 'Z-score':
            X = (X - self.xm)/self.xsd
        Xnew = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.es_pred = self.es_model.predict(Xnew)
      

    def trainer(self, x, y, Loss, device='cpu'):
        '''
        Train an MLP model with given loss function
        '''
        input_dim = x.shape[1]
        tX = torch.tensor(x, dtype=torch.float32).to(device)
        tY = torch.tensor(y, dtype=torch.float32).to(device)
        shuffled_dataset = shuffle(TensorDataset(tX, tY))
        train_size = int((1 - self.opt['val_pct'])*self.n)

        train_dataset, val_dataset \
            = shuffled_dataset[:train_size], shuffled_dataset[train_size:]
        train_dl = DataLoader(train_dataset, self.opt['batch_size'], 
                              shuffle=True, drop_last=True)
        val_dl = DataLoader(val_dataset, self.opt['batch_size'], shuffle=False)
        
        # initialize the model
        model = MLP(input_dim, options = self.opt).to(device)
        
        # choose the optimizer
        if self.opt['optimizer'] == "sgd":
            optimizer = optim.SGD(model.parameters(), 
                                  lr=self.opt['lr'], 
                                  weight_decay=self.opt['weight_decay'], 
                                  nesterov=self.opt['nesterov'], 
                                  momentum=self.opt['momentum'])
        elif self.opt['optimizer'] == "adam":
            optimizer = optim.Adam(model.parameters(), 
                                   lr=self.opt['lr'], 
                                   weight_decay=self.opt['weight_decay'])
        else:
            raise Exception(self.opt['optimizer'] 
                            + "is currently not available")

        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                              step_size=self.opt['step_size'], 
                                              gamma=self.opt['lr_decay'])

        # track progress
        train_losses = []
        val_losses = []

        epoch = 0
        best_val_loss = 1e8

        # training loop
        while epoch < self.opt['nepochs']:
            train_loss = 0.
            model.train()
            for x_batch, y_batch in train_dl:
                optimizer.zero_grad()
                predictions = model(x_batch)
                loss = Loss(predictions, y_batch.view_as(predictions))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if torch.isnan(loss).any():
                    import warnings
                    warnings.warn('NaN values in the loss during training.')
                    break

            train_losses.append( train_loss / len(train_dl))
            
            if self.opt['val_pct'] > 0:
                # validation loop
                val_loss = 0.
                model.eval()
                with torch.no_grad():  # disable gradient computation
                    for x_batch, y_batch in val_dl:
                        predictions = model(x_batch)
                        loss = Loss(predictions, y_batch.view_as(predictions))
                        val_loss += loss.item()
                val_losses.append( val_loss / len(val_dl) )

                # save the best model
                if val_losses[epoch] < best_val_loss:
                    best_val_loss = val_losses[epoch]
                    torch.save(model.state_dict(), 'best_model.pth')

                # learning rate decay
                scheduler.step()
            else:
                torch.save(model.state_dict(), 'best_model.pth')
            epoch += 1

        final_model = MLP(input_dim, options=self.opt).to(device)
        final_model.load_state_dict(torch.load('best_model.pth'))
        final_model.eval()

        return {'model': final_model,
                'fit': final_model(tX).detach().cpu().numpy().reshape(-1),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epoch': epoch}
    
    
    def ls(self, options=dict(), 
           robust=False, c=None, s=1.,
           plot=False, device='cpu'):
        ''' 
        Fit least squares neural network regression 
        or its robust version with Huber loss
        '''
        
        self.opt.update(options)            
        out = self.trainer(self.X0, self.Y, nn.MSELoss(), device)

        if robust == True and c is None:
            ls_res = self.Y - out['fit']
            scale = s * np.std(ls_res) + (1 - s) * mad(ls_res)
            c = scale * (self.n / np.log(self.n))**(1/3)
            out = self.trainer(self.X0, self.Y, 
                               nn.HuberLoss(delta=c), device)
        self.c = c

        # plot loss
        if plot:
            plt.plot(out['train_losses'][1:out['epoch']], label='Train Loss')
            if self.opt['val_pct'] > 0:
                plt.plot(out['val_losses'][1:out['epoch']], 
                         label='Validation Loss')
            plt.legend()
            plt.show()
        
        self.ls_model = out['model']
        self.ls_fit = out['fit']


    def ls_predict(self, X):
        ''' Compute predicted (conditional) mean at new input X '''
        if self.nm == 'MinMax':
            X = (X - self.xmin)/(self.xmax - self.xmin)
        elif self.nm == 'Z-score':
            X = (X - self.xm)/self.xsd
        Xnew = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.ls_pred = self.ls_model.predict(Xnew)



class Activation(Enum):
    ''' Activation functions '''
    relu = nn.ReLU()
    tanh = nn.Tanh()
    sigmoid = nn.Sigmoid()
    leakyrelu = nn.LeakyReLU()



class MLP(nn.Module):
    ''' Generate a multi-layer perceptron '''
    def __init__(self, input_size, options):
        super(MLP, self).__init__()

        activation_func = Activation[options.get('activation', 'relu')].value
        dropout = options.get('dropout_proportion', 0)
        layers = [input_size] + [options['width']] * options['depth']
        
        nn_structure = []
        for i in range(len(layers) - 1):
            nn_structure.extend([
                nn.Linear(layers[i], layers[i + 1]),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                activation_func
            ])
        nn_structure.append(nn.Linear(layers[-1], 1))

        self.fc_in = nn.Sequential(*nn_structure)


    def forward(self, x):
        return self.fc_in(x)


    def predict(self, X):
        with torch.no_grad():
            self.eval()
            yhat = self.forward(X)[:, 0]
        return yhat.cpu().numpy()
    

###############################################################################
###########################    Helper Functions    ############################
###############################################################################
def mad(x):
    ''' Median absolute deviation '''
    return 1.4826 * np.median(np.abs(x - np.median(x)))