import numpy as np
from split_dataset import split_dataset
from NestedCrossValidation import nested_cross_validation

train_sample, test_sample = split_dataset(dataset_name='heart', label_name='Label', train_per=0.8)

X_features = np.array(train_sample.drop(columns=['Label'],axis=1).to_numpy())
y_target = np.array(train_sample[["Label"]].to_numpy())

# Grid values for C
gridHyperPar = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
elapsed_time = {}
accuracy, std_dev = nested_cross_validation(
    x_feat = X_features,
    y_target = y_target,
    grid_hyper_param = gridHyperPar,
    n_splits=10,
    kernel='linear',
    func_id='nested_cross_validation_linear',
    log_time=elapsed_time
)

# Grid values for C and gamma
# gridHyperPar = {"C": [0.001, 0.01, 0.1, 1, 10, 100], "gamma": [0.00001, .0001, .001, .01, .1]}
# accuracy, std_dev = nested_cross_validation(
#     x_feat = X_features,
#     y_target = y_target,
#     grid_hyper_param = gridHyperPar,
#     n_splits=10,
#     kernel='rbf',
#     func_id='nested_cross_validation_rbf',
#     log_time=elapsed_time
# )
print(elapsed_time)
