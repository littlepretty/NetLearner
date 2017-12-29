import numpy as np
import pickle

num_runs = 10
hs = [320, 480, 512, 540]
for h in hs:
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
    # print(train_avgs)
    print('Use %d shared neurons for %d runs:' % (h, len(shared_unsw_test)))
    print(test_avgs)
    f.close()
