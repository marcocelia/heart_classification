import numpy as np
from numpy import mean
from numpy import std
from sklearn.svm import LinearSVC
from sklearn.model_selection import  GridSearchCV, KFold
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_excel ('heart.xlsx', sheet_name ='heart_train')

X_features = np.array(df.drop([df.columns[0], df.columns[-1]],axis=1).to_numpy())

y_target = np.array(df[["Label"]].to_numpy())


# Grid values for C

gridHyperPar = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}

# Configure the outer cross-validation class

outerCV = KFold(n_splits=10)

# Array to store outer cross validation scores

outerScores=np.array([])


for trainOuter,testOuter in outerCV.split(X_features):

    current_X_Outer_TrainingSet = X_features[trainOuter]

    current_X_Outer_TestSet = X_features[testOuter]

    current_y_Outer_TrainingSet = y_target[trainOuter]

    current_y_Outer_TestSet = y_target[testOuter]

    # Configure the inner cross validation class

    innerCV = KFold(n_splits=3, shuffle=True, random_state=1)

    # svmModel = SVC(kernel="linear")

    svmModel = LinearSVC(random_state=0,
                     tol=1e-5,dual=False,
                     max_iter = 5000)


    # The inner cross validation is performed by the
    # by the GridSearch function


    gridSearch = GridSearchCV(svmModel,
                              gridHyperPar,
                              scoring='accuracy',
                              cv=innerCV, refit=True)

    # Execute search

    searchResult = gridSearch.fit(current_X_Outer_TrainingSet,
                                  current_y_Outer_TrainingSet.ravel())


    # Given that refit=True, gridSearch.fit returns the best
    # hyperParameter over the all current_X_Outer_TrainingSet

    best_model = searchResult.best_estimator_

    bestHyperParameter = gridSearch.best_params_['C']

    # We can now evaluate current model
    # on the hold out dataset current_X_Outer_TestSet

    current_y_Predictions = best_model.predict(current_X_Outer_TestSet)

    # Compute accuracy

    currentScore = accuracy_score(current_y_Outer_TestSet,
                                 current_y_Predictions)

    # Append to array of scores

    outerScores = np.append(outerScores, currentScore)

	# Report progress

    print('Accuracy = %.3f, Best Hyper Parameter = %.3f' % (currentScore, bestHyperParameter))

print('Accuracy = %.3f (%.3f)' % (mean(outerScores), std(outerScores)))




