import tensorflow as tf
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np



#this function is used for splitting the datasets into training,cross_Validation and test datasets
def split_data(X,Y):
    X_train, X_, Y_train, Y_ = train_test_split(X, Y, test_size = 0.4, random_state = 1)
    X_cv,X_test,Y_cv,Y_test = train_test_split(X_, Y_, test_size = 0.5, random_state = 1)
    del X_,Y_
    return X_train,Y_train,X_cv,Y_cv,X_test,Y_test



#this function plots the training error and cross_validation error based on the different degrees of polynomial
def plot_error_rate_polydegree(degrees,X_train,X_cv,Y_train,Y_cv):
    training_error = []
    cv_error = []
    degrees_history = []
    models = []
    degrees_collection = range(1,degrees)
    for degree in degrees_collection:
        poly = PolynomialFeatures(degree=degree,include_bias=False)
        sc = StandardScaler()
        lr = LinearRegression()
        X_train_poly = poly.fit_transform(X_train)  #polynomial feature engineering for training data
        X_train_scaled = sc.fit_transform(X_train_poly)  #standard scaling for training data

        X_cv_poly = poly.transform(X_cv)    #polynomial feature engineering for cross_validation dataset
        X_cv_scaled = sc.transform(X_cv_poly)  #standard scaling for polynomial dataset

        lr.fit(X_train_scaled,Y_train)  #training the model
        predicted_Y_train = lr.predict(X_train_scaled)
        training_error.append(mean_squared_error(predicted_Y_train,Y_train) / 2)

        predicted_Y_cv = lr.predict(X_cv_scaled)
        cv_error.append(mean_squared_error(predicted_Y_cv,Y_cv) / 2)
        degrees_history.append(degree)
        models.append(lr)

    plt.plot(degrees_history,training_error,label="Training Error",c='b',marker='o')
    plt.plot(degrees_history,cv_error,label="CV Error",c='r',marker='o')
    plt.legend()
    plt.xlabel("Degrees")
    plt.ylabel("Mean Squared Error")
    plt.show()



#this function plot the training as well as cross_Validation error based on different regularization parameters
def plot_regularized_error(reg_params,degree,X_train,X_cv,Y_train,Y_cv):
    training_error = []
    cv_error = []
    reg_history = []
    for reg in reg_params:
        sc = StandardScaler()
        poly = PolynomialFeatures(degree=degree,include_bias=False)
        #for training data
        X_train_poly = poly.fit_transform(X_train)
        X_train_scaled = sc.fit_transform(X_train_poly)
        #for cross_validation data
        X_cv_poly = poly.transform(X_cv)
        X_cv_scaled = sc.transform(X_cv_poly)
        
        lr = LinearRegression()
        lr=Ridge(alpha=reg)  #applying the regularization
        lr.fit(X_train_scaled,Y_train)   #training the model
        Y_train_predicted=lr.predict(X_train_scaled)
        training_error.append(mean_squared_error(Y_train_predicted,Y_train) / 2)

        Y_cv_predicted=lr.predict(X_cv_scaled)
        cv_error.append(mean_squared_error(Y_cv_predicted,Y_cv) / 2)
        reg_history.append(reg)
    plt.plot(reg_history,training_error,label="Training Error",c='b',marker='o')
    plt.plot(reg_history,cv_error,label="CV Error",c='r',marker='o')
    plt.xlabel('Regularization Parameter')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()

def plot_learning_curve(degree,X_train,X_cv,Y_train,Y_cv):
    training_error = []
    cv_error = []
    precents =[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    sample_hist = []  #sample_hist stores the total sample size of both training as well as cross_validation datasets at each iteration
    for percent in precents:
        train_sample_size = round(len(X_train) * (percent/100))   #getting the trainig sample size based on current percent
        cv_sample_size = round(len(X_cv) * (percent/100))         #same here for the cross_validation dataset
        sc = StandardScaler()
        poly = PolynomialFeatures(degree=degree,include_bias=False)
        X_train_poly = poly.fit_transform(X_train[:train_sample_size])
        X_train_scaled = sc.fit_transform(X_train_poly)
        X_cv_poly = poly.transform(X_cv[:cv_sample_size])
        X_cv_scaled = sc.transform(X_cv_poly)
        lr = LinearRegression()
        lr.fit(X_train_scaled,Y_train[:train_sample_size])   #training the model, the target variable size must match with the X size 
        Y_train_predicted=lr.predict(X_train_scaled)
        training_error.append(mean_squared_error(Y_train_predicted,Y_train[:train_sample_size]) / 2)

        Y_cv_predicted=lr.predict(X_cv_scaled)
        cv_error.append(mean_squared_error(Y_cv_predicted,Y_cv[:cv_sample_size]) / 2)

        sample_hist.append(train_sample_size+cv_sample_size)  #total sample at each iteration is the sum of training sample and cross_validation sample
    plt.plot(sample_hist,training_error,label="Training Error",c='b',marker='o')
    plt.plot(sample_hist,cv_error,label="CV Error",c='r',marker='o')
    plt.xlabel('Total number of samples')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()



def gen_data(m, seed=1, scale=0.7):
    #generating a data set based on a x^2 with added noise
    c = 0
    x_train = np.linspace(0,49,m)
    np.random.seed(seed)
    y_ideal = x_train**2 + c
    y_train = y_ideal + scale * y_ideal*(np.random.sample((m,))-0.5)
    x_ideal = x_train #for redraw when new data included in X
    return x_train, y_train, x_ideal, y_ideal



class Lin_model:
    def __init__(self,degree,regularization=False,lambda_=0):  #this function designs our linear model
        self.linear_model = LinearRegression()
        if regularization:
           self.linear_model=Ridge(alpha=lambda_)   #if regularization is used then we use the lambda or regularization term
        self.poly = PolynomialFeatures(degree=degree,include_bias=False)
        self.scalar = StandardScaler()

    def fit(self,X_train,y_train):
        X_train_mapped = self.poly.fit_transform(X_train.reshape(-1,1))  #as our training data is in 1D shaped, so we are converting them into 2D
        X_train_scaled = self.scalar.fit_transform(X_train_mapped)
        self.linear_model.fit(X_train_scaled,y_train)  #training the model

    def predict(self,X_test):
        X_test_mapped = self.poly.transform(X_test.reshape(-1,1))  #same here
        X_test_scaled = self.scalar.transform(X_test_mapped)
        y_hat = self.linear_model.predict(X_test_scaled)
        return y_hat
    def mse(self,y_test,y_hat):
        mean_s_error = mean_squared_error(y_test,y_hat) / 2
        return mean_s_error
