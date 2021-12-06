# -*- coding: utf-8 -*-
"""
learning_models.py
This file contains all the models used for wine quality prediction.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn import model_selection,preprocessing
from sklearn import linear_model,tree,ensemble,neighbors,svm,gaussian_process


class Parameters(object):
    """
    Definition of the parameters used in the learning task.
    """
    
    def __init__(self):
        self.SEED = 123456 # random seed
        self.NUM_FOLDS = 5 # number of folds in cross validation
        self.ERROR_SCORE = 'neg_mean_squared_error' # error score
        self.ERROR_FUNC = lambda y,yhat: np.mean((y-yhat)**2)
    
    def set_param(self,param_dic):
        """
        Specify the value of certain parameters.\\
        :param param_dic: dict, {name: value} 
        """
        for item in param_dic.keys():
            self.__dict__[item] = param_dic[item]       


class RegressionModel(object):
    """
    The base class of different learning models.
    """
    
    def __init__(self,model_name=None,sklearn_model=None,parameters=None):
        self.name = [model_name,self.__class__.__name__][model_name is None]
        self.sklearn_model = sklearn_model
        self.param = [parameters,Parameters()][parameters is None]
        self.hyper_param_grid = {}
        self.trained_model = None
        
    def train(self,y_data,X_data):
        """
        Train the data
        """
        y_data, X_data = self.preprocess(y_data, X_data)
        if len(self.hyper_param_grid.keys())>0:
            self.trained_model = model_selection.GridSearchCV(self.sklearn_model,self.hyper_param_grid)
            self.trained_model.fit(X_data,y_data)
        else:
            self.trained_model = self.sklearn_model
            self.trained_model.fit(X_data,y_data)
        return
        
    def predict(self,y_data,X_data):
        """
        Test the model with OOS data
        """
        y_data, X_data = self.preprocess(y_data, X_data)
        if self.trained_model is not None:
            y_pred = self.trained_model.predict(X_data)
            y_pred = self.postprocess(y_pred)
            self.y_pred = y_pred
            self.y_test = y_data
            return y_pred
        else:
            print("The model has not been trained yet.")
            return None
    
    def error_score(self,method='mse'):
        """
        Output the error score of the prediction
        """
        if 'y_pred' not in self.__dict__.keys():
            print("Please apply predict() to the test set first.")
        elif method == 'mse':
            return np.mean((self.y_test-self.y_pred)**2)
        elif method == 'mae':
            return np.mean(np.abs(self.y_test-self.y_pred))
        else:
            print("Unknow error metric '{}'".format(method))

    def preprocess(self,y_data,X_data):
        """
        Pre-process the data.
        """
        return [y_data, X_data]
    
    def postprocess(self,y_data):
        """
        Post-process the data.
        """
        return y_data

# Linear Model

class LeastSquares(RegressionModel):
    
    def __init__(self, *args, **kwargs):
        RegressionModel.__init__(self, *args, **kwargs)
        self.sklearn_model = linear_model.LinearRegression(fit_intercept=True)
        

class Polynomial(LeastSquares):
    
    def __init__(self, *args, **kwargs):
        RegressionModel.__init__(self, *args, **kwargs)
        self.sklearn_model = linear_model.LinearRegression(fit_intercept=False)
    
    def preprocess(self,y_data,X_data):
        X_data = preprocessing.PolynomialFeatures(degree=2).fit_transform(X_data)
        return [y_data, X_data]


class Lasso(RegressionModel):
    
    def __init__(self, *args, **kwargs):
        RegressionModel.__init__(self, *args, **kwargs)
        self.sklearn_model = linear_model.Lasso(normalize=True)
        self.hyper_param_grid = {'alpha':[0.01,0.05,0.1,0.5,1]}


class Ridge(RegressionModel):
    
    def __init__(self, *args, **kwargs):
        RegressionModel.__init__(self, *args, **kwargs)
        self.sklearn_model = linear_model.Ridge(normalize=True)
        self.hyper_param_grid = {'alpha':[0.01,0.05,0.1,0.5,1]}


class Huber(RegressionModel):
    
    def __init__(self, *args, **kwargs):
        RegressionModel.__init__(self, *args, **kwargs)
        self.sklearn_model = linear_model.HuberRegressor()
        self.hyper_param_grid = {'epsilon':[1,1.25,1.5,1.75,2]}


class LinearSVM(RegressionModel):
    
    def __init__(self, *args, **kwargs):
        RegressionModel.__init__(self, *args, **kwargs)
        self.sklearn_model = svm.LinearSVR()
        self.hyper_param_grid = {'epsilon':[0,0.5,1],\
                                 'C':[1,2,5,10,20]}


class DecisionTree(RegressionModel):
    
    def __init__(self, *args, **kwargs):
        RegressionModel.__init__(self, *args, **kwargs)
        self.sklearn_model = tree.DecisionTreeRegressor()
        self.hyper_param_grid = {'max_depth':[2,4,6,8,10]}


class Bagging(RegressionModel):
    
    def __init__(self, *args, **kwargs):
        RegressionModel.__init__(self, *args, **kwargs)
        self.sklearn_model = ensemble.BaggingRegressor()
        self.hyper_param_grid = {'base_estimator':[tree.DecisionTreeRegressor(max_depth=i) for i in [2,4,6,8,10]],\
                                 'n_estimators':[20,50,100]}


class RandomForest(RegressionModel):
    
    def __init__(self, *args, **kwargs):
        RegressionModel.__init__(self, *args, **kwargs)
        self.sklearn_model = ensemble.RandomForestRegressor()
        self.hyper_param_grid = {'max_depth':[2,4,6,8,10],\
                                 'n_estimators':[20,50,100],\
                                 'max_features':['auto','sqrt','log2']}


class GBDT(RegressionModel):
    
    def __init__(self, *args, **kwargs):
        RegressionModel.__init__(self, *args, **kwargs)
        self.sklearn_model = ensemble.GradientBoostingRegressor()
        self.hyper_param_grid = {'max_depth':[2,4,6,8,10],\
                                 'n_estimators':[20,50,100]}


class AdaBoost(RegressionModel):
    
    def __init__(self, *args, **kwargs):
        RegressionModel.__init__(self, *args, **kwargs)
        self.sklearn_model = ensemble.AdaBoostRegressor()
        self.hyper_param_grid = {'base_estimator':[tree.DecisionTreeRegressor(max_depth=i) for i in [2,4,6,8,10]],\
                                 'n_estimators':[20,50,100]}


class KNearestNeighbors(RegressionModel):
    
    def __init__(self, *args, **kwargs):
        RegressionModel.__init__(self, *args, **kwargs)
        self.sklearn_model = neighbors.KNeighborsRegressor()
        self.hyper_param_grid = {'n_neighbors':[3,5,7,10]}
        
    def preprocess(self,y_data,X_data):
        if 'scalar' not in self.__dict__.keys():
            self.scalar = preprocessing.StandardScaler()
            self.scalar.fit(X_data)
        X_data = self.scalar.transform(X_data)
        return [y_data,X_data]


class GaussianSVM(RegressionModel):
    
    def  __init__(self, *args, **kwargs):
        RegressionModel.__init__(self, *args, **kwargs)
        self.sklearn_model = svm.SVR()
        self.hyper_param_grid = {'epsilon':[0,0.5,1],\
                                 'C':[1,2,5,10,20],\
                                 'gamma':['scale','auto']}


class GaussianProcess(RegressionModel):
    
    def __init__(self,*args,**kwargs):
        RegressionModel.__init__(self, *args, **kwargs)
        self.sklearn_model = gaussian_process.GaussianProcessRegressor()
    
