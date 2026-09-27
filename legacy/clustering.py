'''
Estimate the bandwidth to use with the mean-shift algorithm.
That this function takes time at least quadratic in n_samples. 
For large datasets, it’s wise to set that parameter to a small value.

X : array-like, shape=[n_samples, n_features]
Input points.

quantile : float, default 0.3
should be between [0, 1] 0.5 means that the median of all pairwise distances is used.

n_samples : int, optional
The number of samples to use. If not given, all samples are used.

random_state : int, RandomState instance or None (default)
The generator used to randomly select the samples from input points for bandwidth estimation. 
Use an int to make the randomness deterministic.

n_jobs : int or None, optional (default=None)
The number of parallel jobs to run for neighbors search. 
None means 1 unless in a joblib.parallel_backend context. 
-1 means using all processors. See Glossary for more details.
'''

from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

centers = [[1, 1], [-1, -1], [1, -1]]
mats, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)
#mats = {a : [np.random.random((32,16,16))] for a in range(0,100)} #creat 100 random matrix by size 16*16*32


# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(mats, quantile=0.2, n_samples=None, random_state=0, n_jobs=-1)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(mats)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(mats[my_members, 0], mats[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()