import numpy as np
import pickle

num_runs = 10
Us = [320, 480, 512, 540]
for U in Us:
    f = open('result_runs%d_U%d.pkl' % (num_runs, U), 'rb')
    data = pickle.load(f)
    f.close()
    modnet_unsw_train = data['modnet']['unsw']['train']
    modnet_nsl_train = data['modnet']['nsl']['train']
    modnet_unsw_test = data['modnet']['unsw']['test']
    modnet_nsl_test = data['modnet']['nsl']['test']

    unified_unsw_train = data['unified']['unsw']['train']
    unified_nsl_train = data['unified']['nsl']['train']
    unified_unsw_test = data['unified']['unsw']['test']
    unified_nsl_test = data['unified']['nsl']['test']

    train_avgs = [np.mean(modnet_unsw_train), np.mean(unified_unsw_train),
                  np.mean(modnet_nsl_train), np.mean(unified_nsl_train)]
    train_stds = [np.std(modnet_unsw_train), np.std(unified_unsw_train),
                  np.std(modnet_nsl_train), np.std(unified_nsl_train)]
    test_avgs = [np.mean(modnet_unsw_test), np.mean(unified_unsw_test),
                 np.mean(modnet_nsl_test), np.mean(unified_nsl_test)]
    test_stds = [np.std(modnet_unsw_test), np.std(unified_unsw_test),
                 np.std(modnet_nsl_test), np.std(unified_nsl_test)]
    # pprint(train_avgs)
    print('Use %s shared neurons for %d runs:' % (U, len(unified_unsw_test)))
    print(test_avgs)
    # print(test_stds)
