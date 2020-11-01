import numpy as np
from split_dataset import split_dataset
from NestedCrossValidation import nested_cross_validation
from TrainAndTest import train_and_test, plot_report

## Analisys Parameters
kernel = 'linear'

if kernel == 'linear':
    gridHyperPar = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
elif kernel == 'rbf':
    gridHyperPar = {"C": [0.001, 0.01, 0.1, 1, 10, 100], "gamma": [0.00001, .0001, .001, .01, .1]}

n_splits = 10
elapsed_time = {}

print("\nPHASE: SPLIT DATASET")

train_sample, test_sample = split_dataset(dataset_name='heart', label_name='Label', train_per=0.8)

print("\nPHASE: NESTED CROSS VALIDATION")

x_train = np.array(train_sample.drop(columns=['Label'],axis=1).to_numpy())
y_train = np.array(train_sample[["Label"]].to_numpy())

accuracy, std_dev = nested_cross_validation(
    x_feat = x_train,
    y_target = y_train,
    grid_hyper_param = gridHyperPar,
    n_splits=n_splits,
    kernel=kernel,
    func_id=f'nested_cross_validation_{kernel}',
    log_time=elapsed_time
)

print("\nPHASE: TRAIN AND TEST")

x_test = np.array(test_sample.drop(columns=['Label'],axis=1).to_numpy())
y_test = np.array(test_sample[["Label"]].to_numpy())

score, bestHyperParameter, y_predictions = train_and_test(
    x_train = x_train,
    x_test = x_test,
    y_train = y_train,
    y_test = y_test,
    grid_hyper_param = gridHyperPar,
    n_splits=n_splits,
    kernel=kernel,
    func_id=f'train_and_test_{kernel}',
    log_time=elapsed_time
)
print('\nAccuracy = %.3f, Best Hyper Parameter = %.3f\n' % (score, bestHyperParameter))
print(elapsed_time)

plot_report(
    y_test,
    y_predictions,
    target_names=['Absence','Presence'],
    title='SVM\nConfusion matrix, without normalization',
    save_fig=True
)