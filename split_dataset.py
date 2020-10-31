import numpy as np
import pandas as pd

def split(dataset, n_train):
    rand_range = np.arange(dataset.index[0], dataset.index[-1])
    train_index = np.random.choice(rand_range, n_train, replace=False)
    test_index = ~dataset.index.isin(train_index)
    return dataset.loc[train_index], dataset.loc[test_index]

def split_dataset(*,dataset_name,label_name,train_per,write_file=True):
    dataset_file = f'{dataset_name}.xlsx'
    dataset = pd.read_excel (dataset_file, sheet_name=f'{dataset_name}_standardized')
    n_samples = len(dataset)
    test_per = 1 - train_per
    n_samples_train = (int)(train_per*n_samples)

    dataset_p = dataset[ dataset[label_name] == 1 ]    # samples for Presence
    n_samples_p = len(dataset_p)

    dataset_a = dataset[ dataset[label_name] == -1 ]   # samples for Absence
    n_samples_a = len(dataset_a)

    per_a = n_samples_a/n_samples             # percentage of samples for Absence
    per_p = 1 - per_a                         # percentage of samples for Presence

    # Use same percentage of samples from the original dataset, alternatively
    # train test can be perfectly balanced with 50% of samples per class
    split_perc_a = per_a # 0.5
    split_perc_p = per_p # 0.5

    n_train_a = (int)(split_perc_a*n_samples_train)     # number of train samples for Absence
    n_train_p = n_samples_train - n_train_a             # number of train samples for Presence

    print(f"Number of train samples from class A: {n_train_a} out of {n_samples_a}")
    print(f"Number of train samples from class P: {n_train_p} out of {n_samples_p}")

    np.random.seed(123)

    train_sample_a, test_sample_a = split(dataset_a, n_train_a)
    train_sample_p, test_sample_p = split(dataset_p, n_train_p)

    train_sample = pd.concat([train_sample_a, train_sample_p], axis=0)
    test_sample = pd.concat([test_sample_a, test_sample_p], axis=0)

    if (write_file):
        with pd.ExcelWriter(dataset_file, mode='a') as writer:
            train_sample.to_excel(writer, sheet_name=f'{dataset_name}_train')
            test_sample.to_excel(writer, sheet_name=f'{dataset_name}_test')

    return train_sample, test_sample


train_sample, test_sample = split_dataset(dataset_name='heart', label_name='Label', write_file=False)
