import numpy as np
from visualize.feature_metrics import plot_feature_importance
from visualize.feature_metrics import plot_pca_components

from preprocess.unsw import generate_dataset
from netlearner.utils import min_max_scale, augment_quantiled
from netlearner.utils import quantile_transform

generate_dataset(one_hot_encode=False)
raw = np.load('UNSW/train_dataset.npy')
train_labels = np.load('UNSW/train_labels.npy')
y = np.argmax(train_labels, 1)

plot_feature_importance(raw, y, 'UNSW', 'raw')
columns = np.array(range(1, 6) + range(8, 16) + range(17, 19) +
                   range(23, 25) + [26])
minmax, _, _ = min_max_scale(raw, None, None)
augment = augment_quantiled(raw, None, None, columns)
replace = quantile_transform(minmax, None, None, columns)
plot_pca_components(minmax, y, 'UNSW', 'raw')
plot_pca_components(augment, y, 'UNSW', 'augment_quantile')
plot_pca_components(replace, y, 'UNSW', 'quantile_transform')
