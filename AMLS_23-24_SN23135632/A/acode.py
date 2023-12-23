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

    is_validation = True

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
    X_val = x_val.reshape(x_val.shape[0], x_val.shape[1]**2)

    # Create the models
    
    models_name = ['Naive Bayes.jpg', 'LogReg.jpg', 'SGD.jpg', 'Perceptron.jpg', 'K_NN.jpg', 'MLPerceptron.jpg', 'SVC.jpg', 'Decision Tree.jpg', 'Adaboost.jpg', 'Bagging.jpg']

    models = create_models(tune_parameters=is_validation)
    # nb_model, logreg_model, sgd_model, percep_model, knn_model, mlp_model, svc_model, tree_model = models

    # Train the models
    fit_models = train(models, X_train, y_train)
    # fit_nb, fit_logreg, fit_sgd, fit_percep, fit_knn, fit_mlp, fit_svc, fit_tree = fit_models

    """
    fit_nb = nb_model.fit(X_train, y_train)
    fit_logreg = logreg_model.fit(X_train, y_train)
    fit_sgd = sgd_model.fit(X_train, y_train)
    fit_percep = percep_model.fit(X_train, y_train)
    fit_knn = knn_model.fit(X_train, y_train)
    fit_mlp = mlp_model.fit(X_train, y_train)
    fit_svc = svc_model.fit(X_train, y_train)
    fit_tree = tree_model.fit(X_train, y_train)
    """

    """
    

    # Train the model
    fit_adaboost = adaboost_model.fit(X_train, y_train)
    fit_bagging = bagging_model.fit(X_train, y_train)

    """

    # models = [fit_nb, fit_logreg, fit_sgd, fit_percep, fit_knn, fit_mlp, fit_svc, fit_tree]
    # models_name = ['Naive Bayes.jpg', 'LogReg.jpg', 'SGD.jpg', 'Perceptron.jpg', 'K_NN.jpg', 'MLPerceptron.jpg', 'SVC.jpg', 'Decision Tree.jpg']

    # metrics(models, models_name, X_test, y_test)

    # Evaluate the models
    if len(fit_models) == len(models_name):
        for i in range(len(fit_models)):
            validation(fit_models[i], models_name[i], X_val, y_val, is_validation=is_validation)

    if len(fit_models) == len(models_name):
        for i in range(len(fit_models)):
            validation(fit_models[i], models_name[i], X_test, y_test, is_validation=False)

    return

def create_models(tune_parameters):
    if not tune_parameters:
        nb_model = GaussianNB() # Naive Bayes model
        logreg_model = LogisticRegression() # Logistic Regression model
        sgd_model = SGDClassifier() # Stochastic Gradient Descent Classifier (hinge loss = Linear SVM)
        percep_model = Perceptron() # (Linear) Perceptron model
        knn_model = KNeighborsClassifier() # K-Nearest Neigbours model
        mlp_model = MLPClassifier() # Multilayer Perceptron model
        svc_model = LinearSVC() # Linear Support Vector Machine Classifier
        tree_model = DecisionTreeClassifier() # Decision Tree Classifier
        adaboost_model = AdaBoostClassifier() #  AdaBoost Classifier
        bagging_model = BaggingClassifier() # Bagging Classifier
    else:
        nb_model = GaussianNB() # Naive Bayes model
        logreg_model = LogisticRegression(penalty='l2') # Default L2 penalty # best
        sgd_model = SGDClassifier(loss='hinge', penalty='l2') # default loss=hinge, penalty=l2 # best
        percep_model = Perceptron(penalty='l2') # default penalty=none # improves #best
        knn_model = KNeighborsClassifier(n_neighbors=7, weights='distance') # default neighbours=5, weights=uniform #improves #best
        mlp_model = MLPClassifier(hidden_layer_sizes=(50,)) # default hidden_layer_sizes=(100,), activation=relu # improves # best
        svc_model = LinearSVC(penalty='l2', loss='squared_hinge') # default penalty=l2, loss=squared_hinge # improves #best
        tree_model = DecisionTreeClassifier(criterion='gini', splitter='best') # default criterion=gini, aplitter=best # best gini
        adaboost_model = AdaBoostClassifier(learning_rate=0.5) # default n_estimators=50, lr=1, #best 
        bagging_model = BaggingClassifier(n_estimators=50) # default n_estimators=10 # best

    models = [nb_model, logreg_model, sgd_model, percep_model, knn_model, mlp_model, svc_model, tree_model, adaboost_model, bagging_model]

    return models

def train(models, X_train, y_train):
    fit_models = []

    for model in models:
        fit_model = model.fit(X_train, y_train)
        fit_models.append(fit_model)

    return fit_models

"""
def metrics(models, models_name, x_test, y_test):
    report = open('report.txt', 'w')

    mdl = 0
    for model in models:
        
        report.write(f'{models_name[mdl]} score is: {model.score(x_test, y_test)}\n')

        # print(model.score(x_test, y_test))

        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)

        classes = ['normal', 'pneumonia']

        report.write(classification_report(y_test, y_pred, target_names=classes))
        report.write('\n')
        report.write("**************************************************************\n\n\n\n")

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

        plt.title('Naïve Bayes')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.savefig(models_name[mdl])
        mdl=mdl+1

    report.close()
"""

def validation(model, model_name, X_val, y_val, is_validation):

    if not is_validation:
        report = open('report.txt', 'a')
    else:
        report = open('validation_report.txt', 'a')
        
    report.write(f'{model_name} score is: {model.score(X_val, y_val)}\n')

    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)

    classes = ['normal', 'pneumonia']

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

    plt.title('Naïve Bayes')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    if is_validation:
        filename = model_name + '-validation'
    else:
        filename = model_name + '-test'
    plt.savefig(filename) 

    report.close()

    return