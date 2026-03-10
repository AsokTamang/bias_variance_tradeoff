import tensorflow as tf
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



#this function is used for splitting the datasets into training,cross_Validation and test datasets
def split_data(X,Y):
    X_train, X_, Y_train, Y_ = train_test_split(X, Y, test_size = 0.4, random_state = 1)
    X_cv,X_test,Y_cv,Y_test = train_test_split(X_, Y_, test_size = 0.5, random_state = 1)
    del X_,Y_
    return X_train,X_cv,X_test,Y_train,Y_cv,Y_test



def plot_error_rate_polydegree(degrees,X_train,X_cv,Y_train,Y_cv):
    for i in range(degrees):

