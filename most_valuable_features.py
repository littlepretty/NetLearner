from __future__ import print_function, division
import numpy as np


weights = np.transpose(np.load('w0.npy'))
print(weights.shape)

feature_names = ["" for i in range(125)]
prev = 0
prev_name = ''
for line in open('feature_names.txt'):
    if line.startswith('#'):
        continue
    words = line.split()
    index = int(words[0])
    feature_name = words[1][:-1]
    feature_type = words[2]
    if prev_name != '':
        for i in range(prev, index + 1):
            if prev + 1 < index:
                feature_names[i] = prev_name + '_' + str(i - prev)
            else:
                feature_names[i] = prev_name

    prev = index
    prev_name = feature_name

feature_names[-1] = prev_name

print(feature_names, len(feature_names))

sorted_indices = np.argsort(np.absolute(weights), axis=1)
print(sorted_indices[:, 120:124])
