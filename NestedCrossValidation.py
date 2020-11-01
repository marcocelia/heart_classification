import numpy as np
from numpy import mean
from numpy import std
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import  GridSearchCV, KFold
from sklearn.metrics import accuracy_score
import pandas as pd
from elapsed_time import timeit

@timeit
def nested_cross_validation(*,x_feat, y_target, grid_hyper_param, n_splits=10, kernel, **kwargs):

    # Configure the outer cross-validation class
    outerCV = KFold(n_splits=n_splits)

    # Array to store outer cross validation scores
    outerScores=np.array([])

    for train_outer,test_outer in outerCV.split(x_feat):

        curr_x_outer_trainset = x_feat[train_outer]
        curr_y_outer_trainset = y_target[train_outer]
        curr_x_outer_testset = x_feat[test_outer]
        curr_y_outer_testset = y_target[test_outer]

        # Configure the inner cross validation class

        innerCV = KFold(n_splits=n_splits, shuffle=True, random_state=1)

        if (kernel == "linear"):
            svmModel = LinearSVC(
                random_state=0,
                tol=1e-5,
                dual=False,
                max_iter = 5000
            )
        else:
            svmModel = SVC(kernel=kernel)

        # The inner cross validation is performed by the GridSearch function
        gridSearch = GridSearchCV(
            svmModel,
            grid_hyper_param,
            scoring='accuracy',
            cv=innerCV,
            refit=True
        )

        # Execute search
        searchResult = gridSearch.fit(
            curr_x_outer_trainset,
            curr_y_outer_trainset.ravel()
        )

        # Given that refit=True, gridSearch.fit returns the best
        # hyperParameter over the all curr_x_outer_trainset
        best_model = searchResult.best_estimator_
        best_c = gridSearch.best_params_['C']

        # We can now evaluate current model on the hold out dataset curr_x_outer_testset
        current_y_Predictions = best_model.predict(curr_x_outer_testset)

        # Compute accuracy
        currentScore = accuracy_score(
            curr_y_outer_testset,
            current_y_Predictions
        )

        # Append to array of scores
        outerScores = np.append(outerScores, currentScore)

        # Report progress
        if (kernel == "rbf"):
            best_gamma = gridSearch.best_params_['gamma']
            print('Accuracy = %.3f, Best C = %.3f, Best Gamma = %.3f' % (currentScore, best_c, best_gamma))
        else:
            print('Accuracy = %.3f, Best C = %.3f' % (currentScore, best_c))

    print('Accuracy = %.3f (%.3f)' % (mean(outerScores), std(outerScores)))
    return (mean(outerScores), std(outerScores))




