from numpy import mean, std, arange
import pickle
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, height,
                '%.2f' % height, ha='center', va='bottom')


num_runs = 10
U = 320

f = open('SharedLayer/result_runs%d_U%d.pkl' % (num_runs, U), 'rb')
data = pickle.load(f)
f.close()
shared_unsw_train = data['unsw']['train']
shared_nsl_train = data['nsl']['train']
shared_unsw_test = data['unsw']['test']
shared_nsl_test = data['nsl']['test']

f = open('ModalityNets/result_runs%d_U%d.pkl' % (num_runs, U), 'rb')
data = pickle.load(f)
modnet_unsw_train = data['modnet']['unsw']['train']
modnet_nsl_train = data['modnet']['nsl']['train']
modnet_unsw_test = data['modnet']['unsw']['test']
modnet_nsl_test = data['modnet']['nsl']['test']

unified_unsw_train = data['unified']['unsw']['train']
unified_nsl_train = data['unified']['nsl']['train']
unified_unsw_test = data['unified']['unsw']['test']
unified_nsl_test = data['unified']['nsl']['test']
f.close()

# pprint(len(modnet_unsw_test))
# pprint(len(modnet_nsl_test))
# pprint(len(unified_unsw_test))
# pprint(len(unified_nsl_test))
# pprint(len(shared_unsw_test))
# pprint(len(shared_nsl_test))
# pprint(len(ae_unsw_test))
# pprint(len(ae_nsl_test))
# pprint(len(sae_unsw_test))
# pprint(len(sae_nsl_test))
# pprint(len(ae_unified_unsw_test))
# pprint(len(ae_unified_nsl_test))
# pprint(len(sae_unified_unsw_test))
# pprint(len(sae_unified_nsl_test))

unsw_train_avgs = [mean(modnet_unsw_train), mean(unified_unsw_train),
                   mean(shared_unsw_train)]
nsl_train_avgs = [mean(modnet_nsl_train), mean(unified_nsl_train),
                  mean(shared_nsl_train)]
unsw_test_avgs = [mean(modnet_unsw_test), mean(unified_unsw_test),
                  mean(shared_unsw_test)]
nsl_test_avgs = [mean(modnet_nsl_test), mean(unified_nsl_test),
                 mean(shared_nsl_test)]

unsw_train_stds = [std(modnet_unsw_train), std(unified_unsw_train),
                   std(shared_unsw_train)]
nsl_train_stds = [std(modnet_nsl_train), std(unified_nsl_train),
                  std(shared_nsl_train)]
unsw_test_stds = [std(modnet_unsw_test), std(unified_unsw_test),
                  std(shared_unsw_test)]
nsl_test_stds = [std(modnet_nsl_test), std(unified_nsl_test),
                 std(shared_nsl_test)]

pprint(unsw_train_avgs)
pprint(nsl_train_avgs)
pprint(unsw_test_avgs)
pprint(nsl_test_avgs)

matplotlib.rc('font', size=18)
ind = arange(len(unsw_train_avgs))
width = 0.36
tick_labels = ('modality', 'unified', 'shared')
fig, ax = plt.subplots()
rects1 = ax.bar(ind, [x * 100 for x in unsw_train_avgs], width, color='b',
                yerr=unsw_train_stds)
rects2 = ax.bar(ind + width, [x * 100 for x in unsw_test_avgs],
                width, color='r', yerr=unsw_test_stds)

ax.set_ylabel('Train/Test Accuracy(%) for UNSW-NB15 dataset')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(tick_labels)
ax.legend((rects1[0], rects2[0]), ('Train', 'Test'))

autolabel(rects1)
autolabel(rects2)

plt.ylim((80, 101))
plt.grid()
plt.show()

ind = arange(len(nsl_train_avgs))
width = 0.36
fig, ax = plt.subplots()
rects1 = ax.bar(ind, [x * 100 for x in nsl_train_avgs], width, color='b',
                yerr=[x * 100 for x in nsl_train_stds])
rects2 = ax.bar(ind + width, [x * 100 for x in nsl_test_avgs],
                width, color='r', yerr=[x * 100 for x in nsl_test_stds])

ax.set_ylabel('Train/Test Accuracy(%) for NSL-KDD dataset')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(tick_labels)
ax.legend((rects1[0], rects2[0]), ('Train', 'Test'))

autolabel(rects1)
autolabel(rects2)

plt.ylim((72, 102))
plt.grid()
plt.show()
