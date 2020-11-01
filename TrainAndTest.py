import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import  GridSearchCV, KFold
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
from elapsed_time import timeit
from ScoringFunctions import plot_confusion_matrix

@timeit
def train_and_test(*, x_train, x_test, y_train, y_test, grid_hyper_param, n_splits=10, kernel, **kwargs):

    # Configure the outer cross-validation class
    cv_model = KFold(n_splits=n_splits)

    if (kernel == "linear"):
        svmModel = LinearSVC(
            random_state=0,
            tol=1e-5,
            dual=False,
            max_iter = 5000
        )
    else:
        svmModel = SVC(kernel=kernel)

    gridSearch = GridSearchCV(
        svmModel,
        grid_hyper_param,
        scoring='accuracy',
        cv=cv_model,
        refit=True
    )

    searchResult = gridSearch.fit(x_train, y_train.ravel())
    best_model = searchResult.best_estimator_
    bestHyperParameter = gridSearch.best_params_['C']

    y_predictions = best_model.predict(x_test)
    score = accuracy_score(y_test, y_predictions)
    return score, bestHyperParameter, y_predictions

def plot_report(test, predicted, target_names, title, save_fig=False):
    print(classification_report(test, predicted,target_names = target_names))
    np.set_printoptions(precision=2)
    plot_confusion_matrix(
        test,
        predicted,
        classes=target_names,
        normalize=True,
        title=title
    )
    if (save_fig):
        plt.savefig('fig.png')

