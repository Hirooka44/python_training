# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
課題7: iris.csvを読み込みオートスケーリングをしてから、scikit-learnを使ってウォード法による階層的クラスタリングを行え。
クラスター数を変えて、結果を主成分分析(PCA)により可視化をせよ。可視化の様子からいえることを考察せよ。
iris_withspecies.csvも参考にすること。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA

number_of_clusters = 3
do_autoscaling = True  # True or False

# load data set
raw_data = pd.read_csv('iris.csv', encoding='SHIFT-JIS', index_col=0)
raw_data_with_y = pd.read_csv('iris_withspecies.csv', encoding='SHIFT-JIS', index_col=0)
y = raw_data_with_y[raw_data_with_y.columns[0]]

# autoscaling
if do_autoscaling:
    autoscaled_data = (raw_data - raw_data.mean(axis=0)) / raw_data.std(axis=0, ddof=1)
else:
    autoscaled_data = raw_data

# clustering
clustering_results = linkage(autoscaled_data, metric='euclidean', method='ward')
cluster_number = fcluster(clustering_results, number_of_clusters, criterion='maxclust')
# dendrogram plot
dendrogram(clustering_results, labels=raw_data.index, color_threshold=0, orientation='left')
plt.show()

# visualize clustering result with PCA
pca_results = PCA()  # PCA
scoreT = pca_results.fit_transform(autoscaled_data)
plt.scatter(scoreT[:, 0], scoreT[:, 1], c=cluster_number, cmap=plt.get_cmap('jet'))
for number_of_samples in np.arange(0, raw_data.index.shape[0] - 1):
    plt.text(scoreT[number_of_samples, 0], scoreT[number_of_samples, 1], raw_data.index[number_of_samples],
             horizontalalignment='left', verticalalignment='top')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()

plt.scatter(scoreT[:, 0], scoreT[:, 1], c=cluster_number, cmap=plt.get_cmap('jet'))
for number_of_samples in np.arange(0, raw_data.index.shape[0] - 1):
    plt.text(scoreT[number_of_samples, 0], scoreT[number_of_samples, 1], y[number_of_samples + 1],
             horizontalalignment='left', verticalalignment='top')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()
