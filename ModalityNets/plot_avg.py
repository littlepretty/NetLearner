import numpy as np
import pickle

mn_unsw = [86.80, 86.77, 87.32, 87.13, 86.64, 87.14, 88.34,
           87.15, 87.14, 85.93, 87.64, 86.70]
mn_nsl = [79.87, 79.72, 79.40, 79.66, 80.65, 80.97, 79.89,
          80.10, 79.22, 81.54, 80.99, 80.66]
master_unsw = [88.43, 86.49, 86.67, 86.37, 86.79, 86.83, 86.94,
               87.24, 88.51, 86.70, 86.42, 87.68]
master_nsl = [81.06, 80.51, 79.80, 79.05, 81.19, 81.01, 78.39,
              79.91, 79.35, 80.72, 80.77, 80.23]
print(len(mn_unsw), len(mn_nsl), len(master_unsw), len(master_nsl))

print('Modaility UNSW', np.mean(mn_unsw), np.std(mn_unsw))
print('Modaility NSL', np.mean(mn_nsl), np.std(mn_nsl))
print('Master UNSW', np.mean(master_unsw), np.std(master_unsw))
print('Master NSL', np.mean(master_nsl), np.std(master_nsl))
