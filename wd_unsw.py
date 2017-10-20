from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
from preprocess.full_unsw import get_feature_names, symbolic_features
from preprocess.full_unsw import generate_header

CSV_COLUMNS, symbolic_names, continuous_names, discrete_names = \
    get_feature_names('UNSW/feature_names.csv')
print(symbolic_names)
print(continuous_names)
print(discrete_names)
header = generate_header(CSV_COLUMNS)
print('Feature names:', header)
