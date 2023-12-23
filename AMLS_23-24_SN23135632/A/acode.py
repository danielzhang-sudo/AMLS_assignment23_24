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

    models = [fit_nb, fit_logreg, fit_sgd, fit_percep, fit_knn, fit_mlp, fit_svc, fit_tree]
    models_name = ['Naive Bayes.jpg', 'LogReg.jpg', 'SGD.jpg', 'Perceptron.jpg', 'K_NN.jpg', 'MLPerceptron.jpg', 'SVC.jpg', 'Decision Tree.jpg']

    metrics(models, models_name, X_test, y_test)

    return


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