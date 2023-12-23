import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier

def main():

    # Open file
    pneumonia = np.load('./AMLS_23-24_SN23135632/Datasets/pneumoniamnist.npz')
    
    # Extract training, validation, test images
    x_train = pneumonia['train_images']
    x_val = pneumonia['val_images']
    x_test = pneumonia['test_images']

    y_train = pneumonia['train_labels']
    y_val = pneumonia['val_labels']
    y_test = pneumonia['test_labels']

    # Reshape the 2d image into a 1d image    
    X_train = x_train.reshape(x_train.shape[0], x_train.shape[1]**2)
    X_test = x_test.reshape(x_test.shape[0], x_test.shape[1]**2)

    # Create a Naive Bayes Model

    # Create a Logistic Regression model

    # Create a Stochastic Gradient Descent Classifier (hinge loss = Linear SVM)

    # Create a (Linear) Perceptron model

    # Create a K-Nearest Neigbours model

    # Create a Multilayer Perceptron model

    # Create a Linear Support Vector Machine model

    # Create a Decision Tree Classifier
    
    # Create an AdaBoost Classifier

    return
