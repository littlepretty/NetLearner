import numpy as np
import pickle
import sys

num_runs = int(sys.argv[1])
h = int(sys.argv[2])
f = open('result_runs%d_U%d.pkl' % (num_runs, h), 'rb')
data = pickle.load(f)
c = 10

shared_unsw_train = np.sort(data['unsw']['train'])[::-1]
shared_nsl_train = np.sort(data['nsl']['train'])[::-1]
shared_unsw_test = np.sort(data['unsw']['test'])[::-1]
shared_nsl_test = np.sort(data['nsl']['test'])[::-1]

train_avgs = [np.mean(shared_unsw_train), np.mean(shared_nsl_train)]
train_stds = [np.std(shared_unsw_train), np.std(shared_nsl_train)]
test_avgs = [np.mean(shared_unsw_test[:c]), np.mean(shared_nsl_test)]
test_stds = [np.std(shared_unsw_test), np.std(shared_nsl_test)]
# print(train_avgs)
print('Use %d shared neurons for %d runs:' % (h, len(shared_unsw_test)))
print(test_avgs)
f.close()
