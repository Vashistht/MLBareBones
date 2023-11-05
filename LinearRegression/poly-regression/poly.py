#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import pandas as pd 
from matplotlib import pyplot as plt

from operator import itemgetter


class Model(object):
    """
     Polynomial Regression.
    """

    def Phi(self, X, k):
        self.phi_basis = np.linspace(0,k, k+1)
        Phi_big = np.ones((X.shape[0], k+1))
        self.k = k
        for i in range(X.shape[0]):
            Phi_big[i] = np.power(X[i],  self.phi_basis)
        
        return Phi_big
        
    def fit(self, X, y, k):
        """
        Fits the polynomial regression model to the training data.

        Arguments
        ----------
        X: nx1 matrix of n examples
        y: response variable vector for n examples
        k: polynomial degree
        """
        Phi_big = self.Phi(X, k)
        # \wv^{LMS} = (\Phiv^T\Phiv)^{-1} \Phiv^T \yv
        firstterm = np.linalg.inv(np.dot(Phi_big.T, Phi_big))
        secondterm = np.dot(Phi_big.T, y)
        beta = np.dot(firstterm, secondterm)
        self.beta = beta


    def predict(self, X):
        """
        Predicts the dependent variable of new data using the model.

        Arguments
        ----------
        X: nx1 matrix of n examples

        Returns
        ----------
        response variable vector for n examples
        """
        
        # \wv\cdot \phiv(x_i)
        y_pred = np.dot(self.Phi(X,self.k), self.beta)
        return y_pred

    def rmse(self, X, y):
        """
        Returns the RMSE(Root Mean Squared Error) when the model is validated.
        
        Arguments
        ----------
        X: nx1 matrix of n examples
        y: response variable vector for n examples
        
        Returns
        ----------
        RMSE when model is used to predict y
        """
        # E = \sum_{i=1}^N [\wv\cdot \phiv(x_i) - y_n]^2
        predictions = self.predict(X)
        difference = predictions - y
        rmse = np.sqrt( np.mean(np.square(difference) ))
        return rmse

    def polynomial_fit(self, X):
        print("fit", self.beta[0], self.beta[1])
        ylist=[]
        for xi in range(len(X)):
            yi = 0
            for i in range(self.k + 1):
                yi += self.beta[i] * np.power(xi, i)
            ylist.append(yi)
        print(ylist)
        return ylist

#run command:
#python poly.py --data=data/poly_reg_data.csv

if __name__ == '__main__':

    #Read command line arguments
    parser = argparse.ArgumentParser(description='Fit a Polynomial Regression Model')
    parser.add_argument('--data', required=True, help='The file which contains the dataset.')
                        
    args = parser.parse_args()

    input_data = pd.read_csv(args.data)
    
    n = len(input_data['y'])
    n_train = 25
    n_val = n - n_train

    x = input_data['x']
    x = np.array(x)
    x_train = x[:n_train][:,None]
    x_val = x[n_train:][:,None]
    print("xval", x_val.shape)

    y= input_data['y']
    y = np.array(y)
    y_train = y[:n_train][:,None]
    y_val = y[n_train:][:,None]

    #plot validation rmse versus k
	# Your code here
    klist = np.linspace(1,10, 10, dtype=int)
    rmse_list_val = []
    model = Model() 

    for k in klist:
        model.fit(x_train, y_train, k)
        rmse = model.rmse(x_val, y_val)
        # print(rmse)
        rmse_list_val.append(rmse)
    
    k_optimal_val = klist[np.argmin(rmse_list_val)]
    
    plt.figure()
    plt.plot(klist, rmse_list_val, 's')
    plt.plot(klist, rmse_list_val)
    plt.scatter(k_optimal_val, rmse_list_val[np.argmin(rmse_list_val)], edgecolors='red', facecolors='none', s=150, linewidth=2, zorder=5, label=f'Optimal k = {k_optimal_val}')  # Empty circle marker
    plt.xlabel("k")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylabel("RMSE")
    plt.title(f"RMSE vs Degree of polynomial (k) for val dataset")
    plt.legend()
    plt.savefig("rmse_k_val.png", dpi=300)
    plt.show()
    
    #plot training rmse versus k
	# Your code here
    rmse_list_train = []

    for k in klist:
        model.fit(x_train, y_train, k)
        rmse = model.rmse(x_train, y_train)
        rmse_list_train.append(rmse)
    
    k_optimal_train = klist[np.argmin(rmse_list_train)]
    
    plt.figure()
    plt.plot(klist, rmse_list_train, 's')
    plt.plot(klist, rmse_list_train)
    plt.scatter(k_optimal_train, rmse_list_train[np.argmin(rmse_list_train)], edgecolors='red', facecolors='none', s=150, linewidth=2, zorder=5, label=f'Optimal k = {k_optimal_train}')  # Empty circle marker
    plt.xlabel("k")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylabel("RMSE")
    plt.legend()
    plt.title(f"RMSE vs Degree of polynomial (k) for train dataset")    
    plt.savefig("rmse_k_train.png", dpi=300)
    plt.show()
    
    #plot fitted polynomial curve versus k as well as the scattered training data points 
    klist_fit = [1,3, 5,10]
    
    colors = ['b', 'r', 'purple', 'c']
    y_fit_list = []
    
    plt.scatter(x_train, y_train, label = 'train_data', alpha=.7)
    for j in range(len(klist_fit)):
        model = Model()
        model.fit(x_train, y_train, k=klist_fit[j])
        x_cont = np.linspace(np.min(x_train), np.max(x_train), 100)
        y_k = model.predict(x_cont)
        plt.plot(x_cont, y_k, color = colors[j], label = f'k = {klist_fit[j]}')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title("Fitted k deg. polynomial, original y_train values for x_train")
    plt.legend(loc='upper right')
    plt.savefig("fitted_poly.png", dpi=300)
    
    plt.show()  
