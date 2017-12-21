import numpy as np
import logging
from scipy import stats


def extract_avg(filename, num_runs=10):
    f = open(filename, 'r')
    f.readline()
    f.readline()
    f.readline()

    accu = {x: [] for x in range(1, 5)}
    accu[5] = {'unsw': [], 'nsl': []}
    accu[6] = {'unsw': [], 'nsl': []}

    for _ in range(num_runs):
        f.readline()
        f.readline()
        f.readline()

        line = f.readline().strip().split()
        if len(line) == 0:
            break

        accu[6]['unsw'].append(float(line[-1]))
        line = f.readline().strip().split()
        accu[6]['nsl'].append(float(line[-1]))

        line = f.readline().strip().split()
        accu[3].append(float(line[-1]))
        line = f.readline().strip().split()
        accu[1].append(float(line[-1]))

        f.readline()
        line = f.readline().strip().split()
        accu[5]['unsw'].append(float(line[-1]))
        line = f.readline().strip().split()
        accu[5]['nsl'].append(float(line[-1]))

        line = f.readline().strip().split()
        accu[4].append(float(line[-1]))
        line = f.readline().strip().split()
        accu[2].append(float(line[-1]))

    # list_mean_accuracies(accu)
    mean_improvement(accu, key1=2, key2=1, dataset='nsl')
    mean_improvement(accu, key1=4, key2=3, dataset='unsw')
    mean_improvement(accu, key1=5, key2=6, dataset='nsl')
    mean_improvement(accu, key1=5, key2=6, dataset='unsw')


def mean_improvement(accu, key1, key2, dataset):
    if dataset in accu[key1]:
        improve_sum = np.subtract(accu[key1][dataset], accu[key2][dataset])
        improve = np.divide(improve_sum, accu[key2][dataset])
        (t, p_value) = stats.ttest_ind(accu[key1][dataset],
                                       accu[key2][dataset], equal_var=False)
    else:
        improve_sum = np.subtract(accu[key1], accu[key2])
        improve = np.divide(improve_sum, accu[key2])
        (t, p_value) = stats.ttest_ind(accu[key1], accu[key2], equal_var=False)

    absolute_mean = np.mean(improve_sum) * 100
    relative_mean = np.mean(improve) * 100
    logger.info('\t%s accu%s over accu%s' % (dataset, key1, key2))
    logger.info('\t\tmean absolute accu increase: %.6f%%' % absolute_mean)
    logger.info('\t\tmean relative accu increase: %.6f%%' % relative_mean)
    logger.info('\t\tp-value in t-test: %.6f%%' % (p_value * 100))


def list_mean_accuracies(accu):
    for (key, value) in accu.items():
        if key in range(1, 5):
            logger.info("accu%s: avg %.6f, std %.6f for %d runs" %
                        (key, np.mean(value), np.std(value), len(value)))
        else:
            for (x, y) in value.items():
                logger.info("accu%s[%s]: avg %.6f, std %.6f for %d runs" %
                            (key, x, np.mean(y), np.std(y), len(y)))


if __name__ == '__main__':
    configs = [320, 400, 480, 500, 512, 570, 640]
    dirname = 'H1X2Config'
    logfiles = ['%s/accuracy_%d.log' % (dirname, x) for x in configs]

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    hdlr = logging.FileHandler('stat_accuracy.log')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    for (i, d) in enumerate(configs):
        logger.info('When use config %d' % d)
        extract_avg(logfiles[i])
