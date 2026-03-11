import tensorflow as tf
from sklearn.linear_model import LinearRegression
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



def plot_error_rate_polydegree(degrees,X_train,X_cv,Y_train,Y_cv,baseline):
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
    plt.plot(degrees_history,np.repeat(baseline,len(degrees_history)),linestyle='--',label='Baseline')
    plt.legend()
    plt.xlabel("Degrees")
    plt.ylabel("Mean Squared Error")
    plt.show()








