"""
This module contains class and functions necessary for multiclass
classification using parallelization.
"""
import numpy as np
import scipy
import sklearn
from sklearn import preprocessing
import matplotlib.pyplot as plt
import time
import multiprocessing
import itertools
from multiprocessing.dummy import Pool as ThreadPool
# my class import
import mklearn


def predict_binary_class(X, Xscaler, betas, class_pair, method):
    """
    Makes binary predictions for a given model and class for
    either one-vs-one or one-vs-rest classification.
    """
    X_std = Xscaler.transform(X)
    if method == 'ovo':
        y_pred = np.dot(X_std, betas)
        y_pred = np.sign(y_pred)
        y_pred[y_pred == 1] = class_pair[0]
        y_pred[y_pred == -1] = class_pair[1]
        return y_pred
    elif method == 'ovr':
        y_pred = np.dot(X, betas)
        return y_pred

    
def predict_class(y_binary_preds, method):
    """
    Predicts class based on binary predictions using
    one-vs-one or one-vs-rest multiclass classification.
    """
    if method == 'ovo':
        y_preds = np.vstack(y_binary_preds)
        # get most commonly occuring class for each observation
        y_pred_final = np.ravel(scipy.stats.mode(y_preds, axis=0)[0])
        return y_pred_final
    elif method == 'ovr':
        y_preds = np.vstack(y_binary_preds)
        # get the class with highest probability
        y_pred_final = np.argmax(y_preds, axis=0)
        return y_pred_final, y_preds

    
def predict_multiclass(X, scalers, models, classes, 
                       n_threads, method, function):
    """
    Performs full multiclass classification using either one-vs-one
    or one-vs-rest classification. It also supports predictions using
    multiprocessing. 
    """
    print('Performing multiclass prediction using {} threads...'.format(n_threads))
    params = []
    i = 0
    if method == 'ovo':
        for class_pair in itertools.combinations(classes, 2):
            params.append((X, scalers[i], models[i], class_pair, method))
            i += 1
    elif method == 'ovr':
        for class_label in classes:
            params.append((X, scalers[i], models[i], class_label, method))
            i += 1
    # predict binary class by multiprocessing
    s_time = time.time()
    pool = ThreadPool(n_threads)
    binary_preds = pool.starmap(function, params)
    pool.close()
    pool.join()
    # predict multiclass
    y_pred = predict_class(binary_preds, method)
    e_time = time.time()
    print('Multiclass prediction complete. Elapsed time: {}s'.format(
        round(e_time - s_time, 3)))
    return y_pred


def compute_multi_classification_error(y, y_pred):
    """
    Computes classification error between predictions
    and labels.
    """
    y_diff = y - y_pred
    y_diff[y_diff != 0] = 1
    return np.mean(y_diff)


class Multiclass(object):
    """
    This class performs multiclass predictions using mklearn's 
    LinearSVC class.
    Args:
        model = mklearn.myLinearSVC object, estimator for multiclass
        classes = numpy array or list, list of classes in dataset
        multiclass = string, 'ovo' or 'ovr'
        n_threads = int, number of threads for multiprocessing
        Cs = list, list of penalties for cross-validation
        k = int, number of folds in cross-validation
        verbose = bool, controls training output
        fitted_betas = list, contains coefficients of each model
        scalers = list, contains scalers for each dataset used in ovo 
        or ovr model
    """
    def __init__(self, model, classes, multiclass='ovo', n_threads=5, 
                 Cs=1.0, k=1, verbose=False):
        self.model = model
        self.classes = classes
        self.multiclass = multiclass
        self.n_threads = n_threads
        self.Cs = Cs
        self.k = k
        self.verbose = verbose
        self.fitted_betas = None
        self.scalers = None
        
    def standardize(self, X):
        """
        Method to standardize data.
        """
        Xscaler = preprocessing.StandardScaler().fit(X)
        X_std = Xscaler.transform(X)
        return X_std, Xscaler
        
    def get_data(self, X, y, classes):
        """
        Method to get appropriate data for ovo or ovr 
        multiclassification.
        """
        if self.multiclass == 'ovo':
            # get indicies
            c1, c2 = classes[0], classes[1]
            c1_ind, c2_ind = np.where(y == c1)[0], np.where(y == c2)[0]
            # get data subsets
            X_sub = np.concatenate([X[c1_ind,:], X[c2_ind,:]])
            y_sub = np.concatenate([y[c1_ind], y[c2_ind]])
            # relabel y
            y_sub[y_sub == c1] = 1.0
            y_sub[y_sub == c2] = -1.0
            # standardize X
            X_sub_std, Xscaler = self.standardize(X_sub)
            return X_sub_std, y_sub, Xscaler
        elif self.multiclass == 'ovr':
            # relabel y
            y_relab = y.copy()
            y_relab[y_relab != classes] = -1.0
            y_relab[y_relab == classes] = 1.0
            X_std, Xscaler = self.standardize(X) 
            return X, y_relab, Xscaler
        
    def train_model(self, X, y, C):
        """
        This method trains a single linear SVM model.
        Args:
            X, y = numpy arrays, dataset
            C = int, regularization penalty
        Returns:
            model = myLinearSVC object, trained model
        """
        model_train = self.model
        print('fitting with optimal penalty {}'.format(C))
        model_train.fit(X=X, y=y, C=C)
        return model_train
        
    def kfold_shuffle(self, X, y):
        """
        This method shuffles input dataset into k-folds.
        Args:
            X, y = numpy arrays, dataset
        Returns:
            kfolds_idx = list, each element responds to indices 
            in single fold
        """
        # randomly shuffle X and y
        assert X.shape[0] == y.shape[0]
        n = y.shape[0]
        # split indices
        idx = np.random.permutation(y.shape[0])
        idx_s = int(n/self.k)
        kfolds_idx = []
        for fold in range(self.k):
            fold_idx = idx[idx_s*fold:idx_s*(fold+1)]
            kfolds_idx.append(fold_idx)
        return kfolds_idx
    
    def single_kfolds(self, X, y, kfolds_idx, C):
        """
        This method performs single k-folds cross-validation.
        Args:
            X, y = numpy arrays, dataset cleaned for ovo or ovr
            kfolds_idx = list, indices of the folds
            C = regularization penalty
        Returns:
            avg_kfold_error = float, average error
        """
        # train model on training set
        kfold_errors = []
        for fold in range(len(kfolds_idx)):
            # get training/validation set indices
            train_idx = kfolds_idx.copy()
            del train_idx[fold]
            train_idx = np.concatenate(train_idx)
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[kfolds_idx[fold]], y[kfolds_idx[fold]]
            # fit models
            model = self.model
            model.fit(X=X_train, y=y_train, C=C)
            error = model.compute_classification_error(
                X_val, y_val, model.betas[-1])
            kfold_errors.append(error)
        avg_kfold_error = np.mean(kfold_errors)
        return avg_kfold_error
    
    def single_cv(self, X, y):
        """
        This method runs an entire instance of single cross-validation
        for a model. 
        Args:
            X, y = numpy arrays, dataset cleaned for ovo or ovr
        Returns
            errors = numpy array, errors for each penalty tested
        """
        # get dimensions
        assert X.shape[0] == y.shape[0]
        n, d = X.shape
        errors = []
        # get kfolds indices for cross-validation
        kfolds_idx = self.kfold_shuffle(X, y)
        print("Performing cross-validation...\n")
        # perform k-folds cv
        if type(self.Cs) != list:
            print("Regularization penalty value:", self.Cs)
            error = self.single_kfolds(X, y, kfolds_idx, self.Cs)
            print("CV Error:", round(error, 3))
            errors.append(error)
        else:
            for f, C in enumerate(self.Cs):
                print("Regularization penalty value:", C)
                error = self.single_kfolds(X, y, kfolds_idx, C)
                print("CV Error:", round(error, 3))
                errors.append(error)
        return np.array(errors)

    def fit_single_model(self, X, y, c_pair):
        """
        This method fits a single linear SVM model with cross-validation.
        Args:
            X, y = numpy arrays, dataset
            c_pair = tuple or int, tuple for ovo and int for ovr
        Returns:
            model_opt = mklearn.myLinearSVC object, optimum model found
            from cross-validation
        """
        print('Fitting {} model for pair {}'.format(
            self.multiclass, c_pair))
        # get relabeled data subset
        X_sub, y_sub, Xscaler = self.get_data(X, y, c_pair)
        if not self.scalers:
            self.scalers = [Xscaler]
        else:
            self.scalers.append(Xscaler)
        # no cross-validation case
        if self.k == 1:
            print('k = 1, no cross-validation case...')
            model_opt = self.train_model(X_sub, y_sub, self.model.C)
            error = model_opt.compute_classification_error(
                X_sub, y_sub, model_opt.betas[-1])
        # perform k-fold cross-validation to find optimal lambda
        else:
            print('Performing {}-fold cross-validation...'.format(self.k))
            cv_errors = self.single_cv(X_sub, y_sub)
            best_C_ind = np.argmin(cv_errors)
            if len(cv_errors) == 1:
                C_opt = self.Cs
            else:
                C_opt = self.Cs[best_C_ind]
            print('Best lambda: {}'.format(round(C_opt, 3)))
            # fit model using the optimal lambda found from cv
            model_opt = self.train_model(X_sub, y_sub, C_opt)
            error = cv_errors[best_C_ind]
        print('Model fit complete. Final objective cost: {}'.format(
            model_opt.objs[-1]))
        print('Final training error: {}\n'.format(error))
        return model_opt
    
    def fit(self, X, y):
        """
        This method fits multiclass model with cross-validation.
        Args:
            X, y = numpy arrays, dataset
        """
        print('{} multiclass using {}-folds cross-validation'.format(
            self.multiclass, self.k))
        t_start = time.time()
        self.fitted_betas = []
        if self.multiclass == 'ovo':
            classes_combs = list(itertools.combinations(self.classes, 2))
            for class_pair in classes_combs:
                model = self.fit_single_model(X, y, class_pair)
                self.fitted_betas.append(model.betas[-1])
        elif self.multiclass == 'ovr':
            for class_label in self.classes:
                model = self.fit_single_model(X, y, class_label)
                self.fitted_betas.append(model.betas[-1])
        t_end = time.time() 
        print('\nTotal elapsed time: {}s'.format(round(t_end - t_start), 5))
        
       
            
        
        
        
    