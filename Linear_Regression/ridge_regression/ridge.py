#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

from operator import itemgetter


class Model(object):
    """
     Ridge Regression.
    """
    
    # def self(self):
    #     return self.beta, self.bias

    def fit(self, X, y, alpha=0):
        """
        Fits the ridge regression model to the training data.

        Arguments
        ----------
        X: nxp matrix of n examples with p independent variables
        y: response variable vector for n examples
        alpha: regularization parameter.
        """
       	# Your code here
        X = np.array(X) #data
        y = np.array(y) #predictions
        alpha = np.array(alpha) #regularization parameter
   
        
        x_bias = np.ones((np.shape(X)[0],1)) # initiate a column of ones
        X_new = np.hstack([x_bias,X]) # add the column of ones to the data
        
        # $beta^* = (X^TX   +  \lambda \Iv)^{-1} (X^T y) $
        lambdaI_term = np.identity(X_new.shape[1])
        lambdaI_term[0, 0] = 0  # since w0 is not regularized

        inverting_term = np.dot(X_new.T, X_new) + alpha * lambdaI_term
        
        second_term = np.dot(X_new.T, y)
        beta = np.dot(np.linalg.inv(inverting_term), second_term)
        bias, beta = beta[0], beta[1:]
        
        self.beta = beta
        self.bias = bias
        
    def predict(self, X):
        """
        Predicts the dependent variable of new data using the model.

        Arguments
        ----------
        X: nxp matrix of n examples with p covariates

        Returns
        ----------
        response variable vector for n examples
        """
       	# Your code here
        predictions = np.dot(X, self.beta) + self.bias
        return predictions

    def rmse(self, X, y):
        """
        Returns the RMSE(Root Mean Squared Error) when the model is validated.
            
        Arguments
        ----------
        X: nxp matrix of n examples with p covariates
        y: response variable vector for n examples
            
        Returns
        ----------
        RMSE when model is used to predict y
        """
        predictions = self.predict(X)
        # Know: rmse = sqrt( 1/n * sum( (predictions - y)^2 ) )
        rmse = np.sqrt( np.mean(np.square(predictions - y) ))
       	# Your code here
        return rmse
#run command:
#python ridge.py --X_train_set=data/Xtraining.csv --y_train_set=data/Ytraining.csv --X_val_set=data/Xvalidation.csv --y_val_set=data/Yvalidation.csv --y_test_set=data/Ytesting.csv --X_test_set=data/Xtesting.csv

if __name__ == '__main__':

    #Read command line arguments
    parser = argparse.ArgumentParser(description='Fit a Ridge Regression Model')
    parser.add_argument('--X_train_set', required=True, help='The file which contains the covariates of the training dataset.')
    parser.add_argument('--y_train_set', required=True, help='The file which contains the response of the training dataset.')
    parser.add_argument('--X_val_set', required=True, help='The file which contains the covariates of the validation dataset.')
    parser.add_argument('--y_val_set', required=True, help='The file which contains the response of the validation dataset.')
    parser.add_argument('--X_test_set', required=True, help='The file which containts the covariates of the testing dataset.')
    parser.add_argument('--y_test_set', required=True, help='The file which containts the response of the testing dataset.')
                        
    args = parser.parse_args()

    #Parse training dataset
    X_train = np.genfromtxt(args.X_train_set, delimiter=',')
    y_train = np.genfromtxt(args.y_train_set,delimiter=',')
    
    #Parse validation set
    X_val = np.genfromtxt(args.X_val_set, delimiter=',')
    y_val = np.genfromtxt(args.y_val_set, delimiter=',')
    
    #Parse testing set
    X_test = np.genfromtxt(args.X_test_set, delimiter=',')
    y_test = np.genfromtxt(args.y_test_set, delimiter=',')
    
    #find the best regularization parameter
	# Your code here
    b = np.linspace(-5,0, 6)
    a = np.linspace(1,9, 9)
    alpha = np.outer(a,10**b)
    alpha = alpha.reshape(len(a) * len(b))
    
    #plot rmse versus lambda
	# Your code here
    rmse_list = []
    beta_list = []
    bias_list = []
    
    for i in range(len(alpha)):
        model = Model()
        model.fit(X_train, y_train, alpha[i])
        rmse = model.rmse(X_val, y_val)
        rmse_list.append(rmse)
        beta = model.beta
        beta_list.append(beta[:10])
        bias_list.append(model.bias)
    
    alpha_optimal = alpha[np.argmin(rmse_list)]
   
   
    plt.figure()
    plt.plot(alpha, rmse_list, 'ro')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('lambda')
    plt.xscale('log')
    plt.ylabel('rmse')
    plt.title(fr'RMSE vs $\lambda$ with $\lambda^*$  value of {alpha_optimal:.4f}, RMSE of {min(rmse_list):.4f}')
    plt.legend()
    plt.savefig('rmse_vs_lambda.png', dpi=300)

    
    plt.figure()
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'pink']
    for i in range(beta_list[0].shape[0]):
        plt.plot(alpha, [beta[i] for beta in beta_list], 'o', linestyle='None', color=colors[i], label=f'beta {i}')
    
    plt.plot(alpha, bias_list, 's', label='bias')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel(r'$\lambda$')
    plt.xscale('log')
    plt.ylabel(r'$\beta$')
    plt.title(r'$\beta$ vs $\lambda$')
    plt.legend()
    plt.savefig('beta_lambda.png', dpi=300)
    
    #plot predicted versus real value
    model = Model()
    
    model.fit(X_train, y_train, alpha_optimal)
    predictions = model.predict(X_test)
    # print(predictions)
    
    plt.figure()
    plt.plot(y_test, predictions, 'ro')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('true values of glucose concentration')
    plt.ylabel('predictions of glucose concentration')
    plt.title('predictions vs true values')
    plt.legend()
    plt.savefig('predvstest.png', dpi=300)
    plt.show()
    
    #plot regression coefficients
    # Your code here
    L, N= 20, len(X_val) # 20 datasets with 50 samples each
    Nsub = 50 # 50 samples per dataset
    dataset = []
    labels = []
    np.random.seed(3)
    for i in range(L):
        # sample Nsub = 50 points with replacement from X_train
        data = np.random.choice(X_train.shape[0], Nsub, replace=True) # with replacement
        data_i = X_train[data]
        labels_i = y_train[data]
        dataset.append(data_i)
        labels.append(labels_i)
    
    model = Model()
    variance_list = []
    
    for l in range(len(alpha)):
        prediction_alpha = []
        for i in range(L):
            model.fit(dataset[i], labels[i], alpha[l])
            predictions = model.predict(X_val) # this is y^hat_n
            prediction_alpha.append(predictions)
        
        prediction_alpha = np.array(prediction_alpha)
        mean = np.sum(prediction_alpha, axis=0) / L

        variance = np.sum((prediction_alpha - mean)**2, axis=0) / L
        variance = np.sum(variance, axis=0) /N
        variance_list.append(variance)

    plt.figure()
    plt.plot(alpha, variance_list, 'ro',alpha=.8)
    # plt.plot(alpha, variance_list,alpha=.8)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel(r'$\lambda$')
    plt.ylabel('variance')
    plt.xscale('log')
    plt.title(r'variance vs $\lambda$')
    plt.legend()
    plt.savefig('variance_lambda_val.png', dpi=300)
    plt.show()
            
            
            
    
    