import os
from six.moves import cPickle as pickle

from helpers import dataHelper


def getData():
    pickle_file = os.path.join('data', 'notMNIST.pickle')

    if not os.path.exists(pickle_file):
        reloadData()
    try:
        f = open(pickle_file, 'rb')
        loadedData = pickle.load(f)
        f.close()
        return loadedData
    except Exception as e:
        print('Unable to load data from', pickle_file, ':', e)
        raise

def reloadData():
    train_filename = dataHelper.maybe_download('data/notMNIST_large.tar.gz', 247336696)
    test_filename = dataHelper.maybe_download('data/notMNIST_small.tar.gz', 8458043)

    train_folders = dataHelper.maybe_extract(train_filename)
    test_folders = dataHelper.maybe_extract(test_filename)

    train_datasets = dataHelper.maybe_pickle(train_folders, 45000)
    test_datasets = dataHelper.maybe_pickle(test_folders, 1800)

    train_size = 200000
    valid_size = 10000
    test_size = 10000

    valid_dataset, valid_labels, train_dataset, train_labels = dataHelper.merge_datasets(
        train_datasets, train_size, valid_size)
    _, _, test_dataset, test_labels = dataHelper.merge_datasets(test_datasets, test_size)

    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)

    train_dataset, train_labels = dataHelper.randomize(train_dataset, train_labels)
    test_dataset, test_labels = dataHelper.randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = dataHelper.randomize(valid_dataset, valid_labels)

    pickle_file = os.path.join('data', 'notMNIST.pickle')

    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
