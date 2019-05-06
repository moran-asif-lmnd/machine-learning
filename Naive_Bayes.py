import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import math


def load_data(test_size):
    diabetes_data = pd.read_csv("pima-indians-diabetes.csv").values #values turn it into numpy
    X = diabetes_data[:, :-1]
    y = diabetes_data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    return X_train, X_test, y_train, y_test


def logical_indexing(X,y):
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    return X_0, X_1


def calc_mean_std(x):
    return np.mean(x,axis=0), np.std(x,axis=0) #axis=0 by column


def calculate_probability(x, mean, std, prior):
    prob_array = (1 / (np.sqrt(2 * math.pi) * std)) * np.exp(-(x - mean) ** 2 / (2 * std ** 2))
    prob_array = np.reshape(prob_array, (prob_array.shape[0], 1))
    prob = np.prod(prob_array, axis=0)
    return prob * prior


def make_predictions(X_test, mean_label_0, std_label_0, mean_label_1, std_label_1, prior_0, prior_1):
    probabilities = []
    for x in X_test:
        prob_0 = calculate_probability(x, mean_label_0, std_label_0, prior_0)
        prob_1 = calculate_probability(x, mean_label_1, std_label_1, prior_1)
        if prob_0 > prob_1:
            probabilities.append(0)
        else:
            probabilities.append(1)
    return probabilities


def get_accuracy(test, predictions):
    correct = 0
    for x in range(len(test)):
        if test[x] == predictions[x]:
            correct += 1
    return (correct/float(len(test))) * 100.0


def main():
    X_train, X_test, y_train, y_test = load_data(0.25)

    X_train_0, X_train_1 = logical_indexing(X_train,y_train)
    prior_0 = len(X_train_0)/len(X_train)
    prior_1 = len(X_train_1)/len(X_train)
    print ("prior_0={}, prior_1={}".format(prior_0,prior_1))

    mean_label_0, std_label_0 = calc_mean_std(X_train_0)
    mean_label_1, std_label_1 = calc_mean_std(X_train_1)
    print("mean_label_0={}, std_label_0 = {}".format(mean_label_0, std_label_0))
    print("mean_label_1={}, std_label_1 = {}".format(mean_label_1, std_label_1))

    y_pred = make_predictions(X_test, mean_label_0, std_label_0, mean_label_1, std_label_1, prior_0, prior_1)
    print("y_pred={}".format(y_pred))

    accuracy = get_accuracy(y_test,y_pred)
    print("accuracy={}".format(accuracy))


main()

