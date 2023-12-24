import itertools
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, classification_report

def main():
    is_validation=False

    # Open file
    pneumonia = np.load('./AMLS_23-24_SN23135632/Datasets/pathmnist.npz')
    
    # Extract training, validation, test images
    x_train = pneumonia['train_images']
    x_val = pneumonia['val_images']
    x_test = pneumonia['test_images']

    y_train = pneumonia['train_labels']
    y_val = pneumonia['val_labels']
    y_test = pneumonia['test_labels']

    # Change RGB image to grayscale image
    x_train = rgb2gray(x_train)
    x_val = rgb2gray(x_val)
    x_test = rgb2gray(x_test)

    # Reshape to 1d image array
    X_train = x_train.reshape(x_train.shape[0], x_train.shape[1]**2)
    X_test = x_test.reshape(x_test.shape[0], x_test.shape[1]**2)
    X_val = x_val.reshape(x_val.shape[0], x_val.shape[1]**2)

    # Create models
    models = create_models()

    # Train models
    fit_models = train(models, X_train, y_train)

    models_name = ['Naive Bayes.jpg', 'LogReg.jpg', 'SGD.jpg', 'Perceptron.jpg', 'K_NN.jpg', 'MLPerceptron.jpg', 'SVC.jpg', 'Decision Tree.jpg', 'Adaboost.jpg', 'Bagging.jpg']

    # Validate models
    if len(fit_models) == len(models_name):
        for i in range(len(fit_models)):
            validation(fit_models[i], models_name[i], X_val, y_val, is_validation=is_validation)

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

def train(models, X_train, y_train):
    fit_models = []
    for model in models:
        fit_model = model.fit(X_train, y_train)
        fit_models.append(fit_model)

    return fit_models

def validation(model, model_name, X_val, y_val, is_validation):

    if not is_validation:
        report = open('multiclass-report.txt', 'a')
    else:
        report = open('multiclass-validation_report.txt', 'a')
        
    report.write(f'{model_name} score is: {model.score(X_val, y_val)}\n')

    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

    report.write(classification_report(y_val, y_pred, target_names=classes))
    report.write('\n')
    report.write("*****************************************************\n\n\n")

    plt.figure(figsize=(5,5))
    plt.title('confusion matrix')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd' #'.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.title(model_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    if is_validation:
        filename = 'multiclass-' + model_name + '-validation'
    else:
        filename = 'multiclass-' + model_name + '-test'
    plt.savefig(filename) 

    report.close()

    return