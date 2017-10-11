from __future__ import print_function
import numpy as np
from netlearner.utils import create_dir
from netlearner.utils import quantile_transform, log_transform
from preprocess.unsw import generate_dataset
from visualize.feature_plots import plot_feature_histogram
from visualize.feature_plots import plot_feature_with_labels

generate_dataset(one_hot_encode=False)
raw_train_dataset = np.load('UNSW/train_dataset.npy')
train_labels = np.load('UNSW/train_labels.npy')
raw_valid_dataset = np.load('UNSW/valid_dataset.npy')
valid_labels = np.load('UNSW/valid_labels.npy')
raw_test_dataset = np.load('UNSW/test_dataset.npy')
test_labels = np.load('UNSW/test_labels.npy')

# plot_single_feature_histogram(train_dataset, 'sjit')

print('Plot original dataset')
path = 'UNSW/Histogram/Original'
create_dir(path)
plot_feature_histogram(raw_train_dataset, path)
path = 'UNSW/FeatureComp/Original'
create_dir(path)
plot_feature_with_labels(raw_train_dataset, train_labels, path)

columns = np.array(range(1, 6) + range(8, 16) + range(17, 19) +
                   range(23, 25) + [26])
print('Plot quantile transformed dataset')
[train_dataset, valid_dataset, test_dataset] = quantile_transform(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset, columns)
path = 'UNSW/Histogram/QuantileTransformed'
create_dir(path)
plot_feature_histogram(train_dataset, path)
path = 'UNSW/FeatureComp/QuantileTransformed'
create_dir(path)
plot_feature_with_labels(train_dataset, train_labels, path)

print('Plot for log1p transformed dataset')
[train_dataset, valid_dataset, test_dataset] = log_transform(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset, columns)
path = 'UNSW/Histogram/LogTransformed'
create_dir(path)
plot_feature_histogram(train_dataset, path)
path = 'UNSW/FeatureComp/LogTransformed'
create_dir(path)
plot_feature_with_labels(train_dataset, train_labels, path)
