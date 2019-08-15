import numpy as np
from sklearn.model_selection import KFold


def split_train_validation(data, n_splits, shuffle=False):
    """
    Split the data into n_splits training and test sets.
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle)

    train_inds = []
    val_inds = []

    for train_ind, val_ind in kf.split(data):
        train_inds.append(train_ind)
        val_inds.append(val_ind)

    train_inds, val_inds = make_homogenous_validation_sets(train_inds, val_inds)

    return train_inds, val_inds

def make_homogenous_validation_sets(train_inds, val_inds):
    """
    Ensure that the validation sets have equal amount of points.
    """
    min_val_inds = val_inds[0].shape[0]
    for i in range(len(val_inds)):
        if val_inds[i].shape[0] < min_val_inds:
            min_val_inds = val_inds[i].shape[0]

    for i in range(len(val_inds)):
        if val_inds[i].shape[0] > min_val_inds:
            n_inds_to_move = int(val_inds[i].shape[0]-min_val_inds)
            train_inds[i] = np.concatenate((train_inds[i], val_inds[i][0:n_inds_to_move]))
            val_inds[i] = val_inds[i][n_inds_to_move::]

    return train_inds, val_inds

def get_train_validation_set(data, train_ind, val_inds):
    """
    Get the train and test set given their indices.
    """
    training_data = data[train_ind]
    validation_data = data[val_inds]

    return training_data, validation_data
