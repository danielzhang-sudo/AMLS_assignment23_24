import itertools
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, classification_report, plot_roc_curve

def helper(task, mode, alg, ckpt_path, X_train, y_train, x, y, classes):
    if mode == "training":
        # Create model
        scaler = StandardScaler()
        pca = PCA(n_components=None)
        X_pca = pca.fit_transform(X_train)
        model, param_grid = create_model(task, mode, alg)
        # Train model
        # fit_model = training(mode, model, param_grid, alg, X_train, y_train)
        fit_model = training(mode, model, alg, X_train, y_train)
        ckpt_filename = f'./{task}/weights/{alg}_weights.sav'
        pickle.dump(fit_model, open(ckpt_filename, 'wb'))
    elif mode == "validation":
        # Create new model with different hyperparameters
        model = create_model(task, mode, alg)
        # Train the new models 
        fit_model = training(model, X_train, y_train)
        # Test new model on validation set
        testing(fit_model, task, mode, alg, x, y, classes)
        ckpt_filename = f'./{task}/weights/{alg}_validation_weights.sav'
        pickle.dump(fit_model, open(ckpt_filename, 'wb'))
    elif mode == "testing":
        loaded_model = pickle.load(open(ckpt_path, 'rb'))
        testing(loaded_model, task, mode, alg, x, y, classes)

def create_model(task, mode, alg):
    param_grid = {}
    if mode == "training":
        if alg == "nb":
            model = GaussianNB()
        elif alg == "logreg":
            model = LogisticRegression(penalty='l2', class_weight='balanced', max_iter=500)
        elif alg == "sgd":
            model = SGDClassifier()
        elif alg == "percep":
            model = Perceptron()
        elif alg == "knn":
            model = KNeighborsClassifier()
        elif alg == "mlp":
            model = MLPClassifier()
        elif alg == "svc":
            model = LinearSVC(penalty='l2', loss='squared_hinge', class_weight='balanced', dual=False)
            # model = SVC(kernel='linear', gamma='auto')
            """
            model = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("svc", model)])
            param_grid = {
                "pca__n_components": [0.5, 0.75, 0.9, None],
                "svc__kernel": ["linear", "poly", "rbf"]
            }
            """
        elif alg == "tree":
            model = DecisionTreeClassifier()
        elif alg == "ranfo":
            model = RandomForestClassifier(criterion="entropy", class_weight='balanced', n_jobs=10)
            """
            model = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("ranfo", model)])
            param_grid = {
                "pca__n_components": [0.5, 0.75, 0.9, None],
                "ranfo__criterion": ["gini", "entropy"] 
            }
            """
        elif alg == "adaboost":
            model = AdaBoostClassifier()
        elif alg == "bagging":
            model = BaggingClassifier()
        else:
            model = GaussianNB()
    elif mode == "validation":
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
            model = SVC(kernel='rbf',gamma='auto',penalty='l2', loss='squared_hinge')
            # model = Pipeline(steps=[("pca", pca), ("svc", model)])
        elif alg == "tree":
            model = DecisionTreeClassifier(criterion='gini', splitter='best')
        elif alg == "ranfo":
            model = RandomForestClassifier(criterion='entropy')
            # model = Pipeline(steps=[("pca", pca), ("ranfo", model)])
        elif alg == "adaboost":
            model = AdaBoostClassifier(learning_rate=0.5)
        elif alg == "bagging":
            model = BaggingClassifier(n_estimators=50)
        else:
            model = GaussianNB()
    return model, param_grid

# def training(mode, model, param_grid, alg, X_train, y_train):
def training(mode, model, alg, X_train, y_train):
    #search = GridSearchCV(model, param_grid, n_jobs = 5)
    # fit_model = search.fit(X_train, y_train)
    fit_model = model.fit(X_train, y_train)
    if alg == "nb" or alg == 'logreg' or alg == 'sgd' or alg == 'percep' or alg == 'knn' or alg == 'mlp' or alg == 'svc' or alg == 'tree' or alg == "ranfo" or alg == 'adaboost' or alg == 'bagging':
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(fit_model, X_train, y_train, cv=10,return_times=True)
        plt.title('Learning curve')
        plt.plot(train_sizes,np.mean(train_scores,axis=1), label="Training score")
        plt.plot(train_sizes,np.mean(test_scores, axis=1), label="Validation score")
        plt.xlabel("Training set size")
        plt.ylabel("Accuracy score")
        plt.legend(loc='best')
        plt.savefig(f'./A/figures/{alg}_accuracy.png')
    # print("Best parameter (CV score=%0.3f):" % search.best_score_)
    # print(search.best_params_)
    if alg == "nb" or alg == 'sgd' or alg == 'percep' or alg == 'knn' or alg == 'mlp' or alg == 'tree' or alg == 'adaboost' or alg == 'bagging':
        plt.title('Loss curve')
        plt.plot(fit_model.loss_curve_, label="Loss curve")
        plt.xlabel()
        plt.ylabel("Loss score")
        plt.legend(loc="best")
        plt.savefig(f'./A/figures/{alg}_{mode}_loss.png')
    return fit_model

def testing(model, task, mode, model_name, X, y, classes):
    report = ''
    filename = ''

    if mode == "validation":
        report = open(f'./{task}/validation_report.txt', 'a')
        filename = f'./{task}/figures/' + model_name + '-validation'
    elif mode == "testing":
        report = open(f'./{task}/testing_report.txt', 'a')
        filename = f'./{task}/figures/' + model_name + '-test'

    if model_name != 'cnn':  
        report.write(f'{model_name} score is: {model.score(X, y)}\n')

    y_pred = model.predict(X)
    print(y_pred)

    if model_name == 'cnn':
        y_pred = np.argmax (y_pred, axis = 1)

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

    plt.figure()
    plot_roc_curve(model, X, y)
    plt.savefig(f'./{task}/figures/' + model_name + '-roc')

    report.close()