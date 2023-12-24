# AMLS_assignment23_24

### Description of the project

In this project we develop several models for classification tasks. We use two datasets, on for binary classification and another for multiclass classification. The structure of the project is the following:

```
AMLS_ASSIGNMENT23_24/
|- A/
|  |- task_A.py (code for Task A)
|  |- figures (Figures saved from Task A)
|  |  |- nb-test.png
|  |  |- nb-validation-png
|  |  |- ...
|  |- weights (Weights if trained models in Task A)
|  |  |- nb_weights.sav
|  |  |- ...
|  |- reports/ (Metrics reports of models in Task A)
|  |  |- testing_report.txt
|  |  |- validation_report.txt
|- B/
|  |- task_B.py (code for Task B)
|  |- figures (Figures saved from Task B)
|  |  |- nb-test.png
|  |  |- nb-validation-png
|  |  |- ...
|  |- weights (Weights if trained models in Task B)
|  |  |- nb_weights.sav
|  |  |- ...
|  |- reports/ (Metrics reports of models in Task B)
|  |  |- testing_report.txt
|  |  |- validation_report.txt
|- Datasets/ (Datasets folder)
|- helper.py (helper functions for Task A and B)
|- main.py (main python file)
|- README.md (Description of the project)
```

### How to run

To run the code in your local machine, first install the dependecies with `pip install -f requirements.txt`. Next open the terminal in the project directory and execute the following:
`python3 main.py --task [A/B] --mode [training/validation/testing] --alg [nb/logreg/sgd/percep/knn/mlp/svc/tree/adaboost/bagging] --ckpt_path [path_to_model_weights]`

When running in testing mode, it is necessary to include the path to the saved model to load the weights, otherwise it is an optional argument.

The code trains, validates or tests one model at a time and saves figures and reports about its results in their respective folders.