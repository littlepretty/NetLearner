import numpy as np
import pickle
import sys

num_runs = sys.argv[1]
U = sys.argv[2]

f = open('result_runs%s_U%s.pkl' % (num_runs, U), 'rb')
data = pickle.load(f)
f.close()
modnet_unsw_train = np.sort(data['modnet']['unsw']['train'])[::-1]
modnet_nsl_train = np.sort(data['modnet']['nsl']['train'])[::-1]
modnet_unsw_test = np.sort(data['modnet']['unsw']['test'])[::-1]
modnet_nsl_test = np.sort(data['modnet']['nsl']['test'])[::-1]

unified_unsw_train = np.sort(data['unified']['unsw']['train'])[::-1]
unified_nsl_train = np.sort(data['unified']['nsl']['train'])[::-1]
unified_unsw_test = np.sort(data['unified']['unsw']['test'])[::-1]
unified_nsl_test = np.sort(data['unified']['nsl']['test'])[::-1]

train_avgs = [np.mean(modnet_unsw_train), np.mean(unified_unsw_train),
              np.mean(modnet_nsl_train), np.mean(unified_nsl_train)]
train_stds = [np.std(modnet_unsw_train), np.std(unified_unsw_train),
              np.std(modnet_nsl_train), np.std(unified_nsl_train)]
test_avgs = [np.mean(modnet_unsw_test[:10]), np.mean(unified_unsw_test[:10]),
             np.mean(modnet_nsl_test[:10]), np.mean(unified_nsl_test[:10])]
test_stds = [np.std(modnet_unsw_test), np.std(unified_unsw_test),
             np.std(modnet_nsl_test), np.std(unified_nsl_test)]
# pprint(train_avgs)
print('Use %s shared neurons for %d runs:' % (U, len(unified_unsw_test)))
print(test_avgs)
# print(test_stds)
