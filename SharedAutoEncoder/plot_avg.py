from numpy import mean, std
import pickle

num_runs = 10
Us = [640, 540, 512, 480, 320]
for U in Us:
    f = open('result_runs%s_U%d.pkl' % (num_runs, U), 'rb')
    data = pickle.load(f)
    f.close()
    ae_unsw_train = data['ae']['unsw']['train']
    ae_nsl_train = data['ae']['nsl']['train']
    ae_unsw_test = data['ae']['unsw']['test']
    ae_nsl_test = data['ae']['nsl']['test']

    sae_unsw_train = data['sae']['unsw']['train']
    sae_nsl_train = data['sae']['nsl']['train']
    sae_unsw_test = data['sae']['unsw']['test']
    sae_nsl_test = data['sae']['nsl']['test']

    ae_unified_unsw_train = data['ae_unified']['unsw']['train']
    ae_unified_nsl_train = data['ae_unified']['nsl']['train']
    ae_unified_unsw_test = data['ae_unified']['unsw']['test']
    ae_unified_nsl_test = data['ae_unified']['nsl']['test']

    sae_unified_unsw_train = data['sae_unified']['unsw']['train']
    sae_unified_nsl_train = data['sae_unified']['nsl']['train']
    sae_unified_unsw_test = data['sae_unified']['unsw']['test']
    sae_unified_nsl_test = data['sae_unified']['nsl']['test']

    unsw_train_avgs = [mean(ae_unsw_train), mean(sae_unsw_train),
                       mean(ae_unified_unsw_train),
                       mean(sae_unified_unsw_train)]
    nsl_train_avgs = [mean(ae_nsl_train), mean(sae_nsl_train),
                      mean(ae_unified_nsl_train), mean(sae_unified_nsl_train)]
    unsw_test_avgs = [mean(ae_unsw_test), mean(sae_unsw_test),
                      mean(ae_unified_unsw_test), mean(sae_unified_unsw_test)]
    nsl_test_avgs = [mean(ae_nsl_test), mean(sae_nsl_test),
                     mean(ae_unified_nsl_test), mean(sae_unified_nsl_test)]

    unsw_train_stds = [std(ae_unsw_train), std(sae_unsw_train),
                       std(ae_unified_unsw_train), std(sae_unified_unsw_train)]
    nsl_train_stds = [std(ae_nsl_train), std(sae_nsl_train),
                      std(ae_unified_nsl_train), std(sae_unified_nsl_train)]
    unsw_test_stds = [std(ae_unsw_test), std(sae_unsw_test),
                      std(ae_unified_unsw_test), std(sae_unified_unsw_test)]
    nsl_test_stds = [std(ae_nsl_test), std(sae_nsl_test),
                     std(ae_unified_nsl_test), std(sae_unified_nsl_test)]
    print('Use %d shared neurons for %s runs: ' %
          (U, len(sae_unified_unsw_test)))
    # pprint(unsw_train_avgs)
    # pprint(nsl_train_avgs)
    print(unsw_test_avgs)
    print(nsl_test_avgs)
