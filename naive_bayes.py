"""This module includes methods for training and predicting using naive Bayes."""
import numpy as np
import sys


def naive_bayes_train(train_data, train_labels, params):
    """Train naive Bayes parameters from data.

    :param train_data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type train_data: ndarray
    :param train_labels: length n numpy vector with integer labels
    :type train_labels: array_like
    :param params: learning algorithm parameter dictionary. (Optional. Can be empty)
    :type params: dict
    :return: model learned with the priors and conditional probabilities of each feature
    :rtype: model
    """
    labels = np.unique(train_labels)
    d = train_data.shape[0]
    model = {
        'prior': np.zeros([labels.size, 2]),
        'prob': np.zeros([labels.size, d])
    }
    model['prior'][:, 0] = labels
    for c in range(labels.size):
        mask = train_labels == labels[c]
        class_data = train_data[:, mask]
        model['prob'][c] = np.log((class_data.sum(axis=1) + sys.float_info.min) /
                                  (mask.sum() + sys.float_info.min * 2))
        model['prior'][c, 1] = np.log(np.mean(mask))
    return model


def naive_bayes_predict(data, model):
    """Use trained naive Bayes parameters to predict the class with highest conditional likelihood.

    :param data: d x n numpy matrix (ndarray) of d binary features for n examples
    :type data: ndarray
    :param model: learned naive Bayes model
    :type model: dict
    :return: length n numpy array of the predicted class labels
    :rtype: array_like
    """
    return model['prior'][:, 0][np.argmax(np.matmul(model['prob'], data) +
                                          model['prior'][:, 1].reshape(-1, 1), axis=0)]
