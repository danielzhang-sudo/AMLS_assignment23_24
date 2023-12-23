import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, classification_report

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
    nb_model = GaussianNB()

    # Train the model
    fit_nb = nb_model.fit(X_train, y_train)

    # Create a Logistic Regression model
    logreg_model = LogisticRegression()

    # Train the model
    fit_logreg = logreg_model.fit(X_train, y_train)

    # Create a Stochastic Gradient Descent Classifier (hinge loss = Linear SVM)
    sgd_model = SGDClassifier()

    # Train the model
    fit_sgd = sgd_model.fit(X_train, y_train)

    # Create a (Linear) Perceptron model
    percep_model = Perceptron()

    # Train the model
    fit_percep = percep_model.fit(X_train, y_train)

    # Create a K-Nearest Neigbours model
    knn_model = KNeighborsClassifier()

    # Train the model
    fit_knn = knn_model.fit(X_train, y_train)

    # Create a Multilayer Perceptron model
    mlp_model = MLPClassifier()

    # Train the model
    fit_mlp = mlp_model.fit(X_train, y_train)

    # Create a Linear Support Vector Machine model
    svc_model = LinearSVC()

    # Train the model
    fit_svc = svc_model.fit(X_train, y_train)

    # Create a Decision Tree Classifier
    tree_model = DecisionTreeClassifier()

    # Train the model
    fit_tree = tree_model.fit(X_train, y_train)

    """
    # Create an AdaBoost Classifier
    adaboost_model = AdaBoostClassifier()

    # Train the model
    fit_adaboost = adaboost_model.fit(X_train, y_train)

    # Create a Bagging Classifier
    bagging_model = BaggingClassifier()

    # Train the model
    fit_bagging = bagging_model.fit(X_train, y_train)
    """

    return