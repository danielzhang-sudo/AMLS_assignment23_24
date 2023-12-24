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

def helper(task, mode, alg, ckpt_path, X_train, y_train, x, y, classes):
    if mode == "training":
        # Create model
        model = create_model(task, mode, alg)
        # Train model
        fit_model = training(model, x, y)
        ckpt_filename = f'./{task}/weights/{alg}_weights.sav'
        pickle.dump(fit_model, open(ckpt_filename, 'wb'))
    elif mode == "validation":
        # Create new model with different hyperparameters
        model = create_model(mode, alg)
        # Train the new models 
        fit_model = training(model, X_train, y_train)
        # Test new model on validation set
        testing(model, task, mode, alg, x, y, classes)
        ckpt_filename = f'./{alg}/{alg}_validation_weights.sav'
        pickle.dump(fit_model, open(ckpt_filename, 'wb'))
    elif mode == "testing":
        loaded_model = pickle.load(open(ckpt_path, 'rb'))
        testing(loaded_model, task, mode, alg, x, y, classes)

def create_model(task, mode, alg):
    if mode == "validation":
        if alg == "nb":
            model = GaussianNB()
        elif alg == "logreg":
            model = LogisticRegression()
        elif alg == "sgd":
            model = SGDClassifier()
        elif alg == "percep":
            model = Perceptron()
        elif alg == "knn":
            model = KNeighborsClassifier()
        elif alg == "mlp":
            model = MLPClassifier()
        elif alg == "svc":
            model = LinearSVC()
        elif alg == "tree":
            model = DecisionTreeClassifier()
        elif alg == "adaboost":
            model = AdaBoostClassifier()
        elif alg == "bagging":
            model = BaggingClassifier()
        else:
            model = GaussianNB()
    else:
        if alg == "nb":
            model = GaussianNB()
        elif alg == "logreg":
            model = LogisticRegression(penalty='l2')
        elif alg == "sgd":
            model = SGDClassifier(loss='hinge', penalty='l2')
        elif alg == "percep":
            model = Perceptron(penalty='l2')
        elif alg == "knn":
            model = KNeighborsClassifier(n_neighbors=7, weights='distance')
        elif alg == "mlp":
            model = MLPClassifier(hidden_layer_sizes=(50,))
        elif alg == "svc":
            model = LinearSVC(penalty='l2', loss='squared_hinge')
        elif alg == "tree":
            model = DecisionTreeClassifier(criterion='gini', splitter='best')
        elif alg == "adaboost":
            model = AdaBoostClassifier(learning_rate=0.5)
        elif alg == "bagging":
            model = BaggingClassifier(n_estimators=50)
        else:
            model = GaussianNB()

    return model

def training(model, X_train, y_train):
    fit_model = model.fit(X_train, y_train)
    return fit_model

def testing(model, task, mode, model_name, X, y, classes):
    report = ''
    filename = ''

    if mode == "validation":
        report = open(f'{task}/validation_report.txt', 'a')
        filename = model_name + '-validation'
    elif mode == "testing":
        report = open('testing_report.txt', 'a')
        filename = f'./{task}/figures/' + model_name + '-test'
        
    report.write(f'{model_name} score is: {model.score(X, y)}\n')

    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    report.write(classification_report(y, y_pred, target_names=classes))
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
    
    plt.savefig(filename) 

    report.close()