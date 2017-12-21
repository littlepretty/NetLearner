import numpy as np
import pickle
from pprint import pprint

num_runs = 10
h = 480
f = open('result_runs%d_U%d.pkl' % (num_runs, h), 'rb')
data = pickle.load(f)

shared_unsw_train = data['unsw']['train']
shared_nsl_train = data['nsl']['train']
shared_unsw_test = data['unsw']['test']
shared_nsl_test = data['nsl']['test']

train_avgs = [np.mean(shared_unsw_train), np.mean(shared_nsl_train)]
train_stds = [np.std(shared_unsw_train), np.std(shared_nsl_train)]
test_avgs = [np.mean(shared_unsw_test), np.mean(shared_nsl_test)]
test_stds = [np.std(shared_unsw_test), np.std(shared_nsl_test)]
pprint(train_avgs)
pprint(test_avgs)
pprint(test_stds)
