import itertools
import pickle
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
    pneumonia = np.load('./AMLS_23-24_SN23135632/Datasets/pathmnist.npz')
    
    # Extract training, validation, test images
    x_train = pneumonia['train_images']
    x_val = pneumonia['val_images']
    x_test = pneumonia['test_images']

    y_train = pneumonia['train_labels']
    y_val = pneumonia['val_labels']
    y_test = pneumonia['test_labels']

    # Create models
    models = create_models()


    return

def create_models():
    nb_model = GaussianNB()
    logreg_model = LogisticRegression(multi_class='multinomial')
    sgd_model = SGDClassifier()
    percep_model = Perceptron()
    knn_model = KNeighborsClassifier()
    mlp_model = MLPClassifier()
    svm_model = LinearSVC()
    tree_model = DecisionTreeClassifier()
    adaboost_model = AdaBoostClassifier()
    bagging_model = BaggingClassifier()

    models = [nb_model, logreg_model, sgd_model, percep_model, knn_model, mlp_model, svm_model, tree_model, adaboost_model, bagging_model]
    return models