import numpy as np 
import pandas as pd 
import time
import matplotlib.pyplot as plt


class myLinearSVC(object):
    """ 
    My implementation of Linear Support Vector Machine classificer.
    Args:
        loss = string, loss function (currently only supports
        'smooth_hinge'
        tol = float, tolerance for convergence
        C = float, regularization penalty
        verbose = bool, output during training
        max_iter = int, maximum interations till convergence
        betas = list, coefficients from each iteration of training
        objs = list, objective values from each iteration of training
        errors = list, training errors from each iteration of training
    """
    def __init__(self, loss='smooth_hinge', tol=0.0001, C=1.0, 
                 verbose=False, max_iter=1000, random_seed=0):
        self.loss = loss
        self.tol = tol
        self.C = C
        self.verbose = verbose
        self.max_iter = max_iter
        self.betas = None 
        self.objs = None 
        self.errors = None
        if loss == 'smooth_hinge':
            self.h = 0.5

    def compute_obj(self, X, y, betas):
        """
        Computes the objective using smoothed hinge loss.
        INPUTS:
            X = dim(n, d)
            y = dim(n, )
            betas = dim(d, )
        OUTPUT:
            grad = dim(d, )
        """
        # n, d = X.shape
        if self.loss == 'smooth_hinge':
            yt = y*X.dot(betas)
            ell = (1+self.h-yt)**2 / (4*self.h)*(np.abs(1-yt) <= self.h) + (1-yt)*(yt < (1-self.h))
            return np.mean(ell) + self.C*np.dot(betas, betas)
        elif self.loss == 'squared_hinge':
            raise ValueError('squared hinge loss is not implemented yet!')

    def compute_grad(self, X, y, betas):
        """
        Computes gradient using smoothed hinge loss.
        INPUTS:
            X = dim(n, d)
            y = dim(n, )
            beta = dim(d, )
        OUTPUT:
            grad = dim(d, )
        """
        n, d = X.shape
        if self.loss == 'smooth_hinge':
            yt = y*X.dot(betas)
            ell_prime = -(1+self.h-yt) / (2*self.h)*y*(np.abs(1-yt) <= self.h) - y*(yt < (1-self.h))
            return np.mean(ell_prime[:, np.newaxis]*X, axis=0) + 2*self.C*betas
        elif self.loss == 'squared_hinge':
            raise ValueError('squared hinge loss is not implemented yet!')

    def backtracking(self, X, y, betas, t, alpha, b, max_iter=100, verbose=False):
        """
        Implements backtracking rule to find step size.
        INPUTS:
            X = current point, dim(n, d)
            y = response at current point, dim(n, 1)
            betas = regression coefficients, dim(d, 1)
            t = starting (maximum) step size (scalar)
            alpha = constant used to define backtracking 
                conidtion, (scalar)
            b = factor to decrease t by when backtracking 
                condition not met, (scalar)
            max_iter = max number of iterations to run the 
                algorithm, (scalar)
        OUPUT:
            t = step size to be used, (scalar)
        """
        # find the gradient of x and its L2 norm
        grad = self.compute_grad(X, y, betas)
        l2_norm_grad = np.linalg.norm(grad)
        # loop to find t
        found_t = False
        i = 0 
        while (found_t is False and i < max_iter):
            # check backtrack condition
            lhs = self.compute_obj(X, y, betas - t*grad)
            rhs = self.compute_obj(X, y, betas) - alpha*t*l2_norm_grad**2    
            if verbose:
                print("iteration:", i)
                print("LHS:", lhs)
                print("RHS:", rhs)
                print("step size:", t)
                print("\n")
            if (lhs < rhs):
                found_t = True
            elif i == max_iter - 1:
                return t
            else:
                t *= b
                i += 1
        return t

    def compute_classification_error(self, X, y, betas):
        """
        Misclassification error calculation for (-1, +1) classification.
        INPUTS:
            X = predictors, dim(n, d)
            y = response, dim(n, 1)
            betas = regression coefficients, dim(d, 1)
        OUPUT:
            error = misclassfication error, (scalar)
        """
        # make predictions and convert to classification
        y_dec = np.dot(X, betas)
        y_pred = np.sign(y_dec)
        # find error 
        num_incorr = np.count_nonzero(y - y_pred)
        error_rate = num_incorr/float(X.shape[0])
        return error_rate

    def compute_iter_error(self, X, y):
        """
        Computes misclassification error at each 
        INPUTS:
            X, y = numpy arrays, dataset
        OUPUT:
            errors = errors at each iteration of training
        """
        n, d = X.shape
        errors = []
        for i in range(len(self.betas)):
            if i%1 == 0:
                print("Iteration Num:", i+1, end="\r")
                error = self.compute_classification_error(X, y, self.betas[i])
                errors.append(error)
        return np.array(errors)

    def fit(self, X, y, t_init=1.0, alpha=0.5, b=0.5, 
            max_iter_bt=100, C=None):
        """
        INPUTS:
            X = matrix of predictors, dim(n, d)
            y = vector of responses, dim(n, 1)
            t_init = initial step size, (scalar)
            alpha = constant used to define backtracking 
                conidtion, (scalar)
            b = factor to decrease t by when backtracking 
                condition not met, (scalar)
            max_iter = max number of iterations for gradient 
                descent algorithm, (scalar)
            max_iter_bt = max numer of iterations for backtracking 
                algorithm, (scalar)
        OUTPUTS:
            betas_vals = regression coefficients at each gradient 
                descent iteration, dim(i, d)
            obj_vals = objective value at each gradient descent 
                iteration, dim(i, 1) 
        """
        if C is not None:
            self.C = C
        # get dimensions, initialize betas, gradient, and objective
        n, d = X.shape
        betas = np.zeros(d)
        thetas = np.zeros(d)
        grad_thetas = self.compute_grad(X, y, thetas)
        error = self.compute_classification_error(X, y, betas)
        obj = self.compute_obj(X, y, betas)
        betas_vals = [betas]
        obj_vals = [obj]
        errors = [error]
        t = t_init
        # perform gradient descent
        iter = 0
        while np.linalg.norm(grad_thetas) > self.tol and iter < self.max_iter:
            # find t
            t = self.backtracking(X, y, thetas, t_init, alpha, b, max_iter_bt)
            # update betas and thetas and get error
            betas_new = thetas - t*grad_thetas
            thetas = betas_new + iter/(iter+3.0)*(betas_new - betas)
            error = self.compute_classification_error(X, y, betas_new)
            obj = self.compute_obj(X, y, betas_new)
            if self.verbose:
                print("Iteration Num: {}, Cost: {}, Error: {}, Step Size: {}".format(iter, 
                        round(obj,5), round(error, 3), round(t, 3)), end="\r")
            # append to beta_vals and obj_vals
            betas_vals.append(betas_new)
            obj_vals.append(obj)
            errors.append(error)
            # recompute grad and obj and increment counter
            grad_thetas = self.compute_grad(X, y, thetas)
            betas = betas_new
            iter += 1
        # print final output
        if self.verbose:
            print("\nFINAL: Iteration Num: {}, Cost: {}, Error: {}, Step Size: {}".format(
                iter, round(obj,5), round(error, 3), round(t, 3)))
        # store results in class object
        self.betas = betas_vals
        self.objs = obj_vals
        self.errors = errors

    def predict(self, X):
        """
        Predicts on new dataset.
        """
        y_dec = np.dot(X, self.betas[-1])
        y_pred = np.sign(y_dec)
        return y_pred

    def decision_function(self, X):
        """
        Gets decision function on a dataset.
        """
        y_dec = np.dot(X, self.betas[-1])
        return y_dec

    def display_results(self, X, y):
        """
        Displays error plots for training and validation data.
            Args:
                X = numpy array, validation data
                y = numpy array, validation data
            Returns:
                None
        """
        # check if model exists in the class object
        if self.betas is None:
            raise ValueError('you must fit the model first!')
        print('final objective:', round(self.objs[-1], 5))
        print('final train error:', round(self.errors[-1], 5))
        iters = np.arange(0, len(self.objs), 1)
        errors_val = self.compute_iter_error(X, y)
        print('final validation error:', round(errors_val[-1], 5))
        # plot results
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
        plt.subplot(121)
        plt.plot(iters, self.objs)
        plt.xlabel("Iteration Number", fontsize=16)
        plt.ylabel("Objective Function Cost", fontsize=16)
        plt.subplot(122)
        plt.plot(iters, self.errors)
        plt.plot(iters, errors_val)
        plt.xlabel("Iteration Number", fontsize=16)
        plt.ylabel("Classification Error", fontsize=16)
        plt.legend(["Train Error", "Validation Error"])