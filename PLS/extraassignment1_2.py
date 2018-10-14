# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
発展課題1: scikit-learnを使わずに、固有値問題に基づく方法とNIPALSアルゴリズムによる方法の2通りでPCAを行うプログラムを
作成せよ。scikit-learnのPCAの結果と等しくなることを確認せよ。
発展課題2: オートスケーリングをする場合としない場合とで主成分分析の結果を比較し、結果を考察せよ。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

data_flag = 1  # 1:iris.csv, 2:random
do_autoscaling = True  # True:autoscaling or False: no autoscaling

if data_flag == 1:
    # load data set
    raw_data = pd.read_csv('iris.csv', encoding='SHIFT-JIS', index_col=0)
elif data_flag == 2:
    # make data set
    sample_number = 30
    gamma = 1
    x_base = np.arange(-1, 1, 0.01, dtype=float)
    gram_matrix = np.exp(-gamma * ((x_base[:, np.newaxis] - x_base) ** 2))
    gram_matrix = gram_matrix + np.identity(gram_matrix.shape[0]) * 1e-14
    L = np.linalg.cholesky(gram_matrix)
    np.random.seed(10000)
    data = L.dot(np.random.randn(x_base.shape[0], sample_number)).transpose()
    np.random.seed()
    raw_data = pd.DataFrame(np.c_[data[:, 0], data[:, 30]])
    raw_data.columns = ['x1', 'x2']
    raw_data.index = [str(i) for i in range(0, sample_number)]
    plt.scatter(raw_data.x1, raw_data.x2)
    for number_of_samples in np.arange(0, raw_data.shape[0] - 1):
        plt.text(raw_data.x1[number_of_samples], raw_data.x2[number_of_samples], raw_data.index[number_of_samples],
                 horizontalalignment='left', verticalalignment='top')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# autoscaling
if do_autoscaling:
    autoscaled_data = (raw_data - raw_data.mean(axis=0)) / raw_data.std(axis=0, ddof=1)
else:
    autoscaled_data = raw_data

# PCA (scikit-learn)
# pca_results = PCA(n_components =3)
pca_results = PCA()
scoreT_scikit_learn = pca_results.fit_transform(autoscaled_data)
loading_vector_scikit_learn = pca_results.components_.transpose()
# plot
plt.scatter(scoreT_scikit_learn[:, 0], scoreT_scikit_learn[:, 1])
for number_of_samples in np.arange(0, raw_data.shape[0] - 1):
    plt.text(scoreT_scikit_learn[number_of_samples, 0], scoreT_scikit_learn[number_of_samples, 1],
             raw_data.index[number_of_samples],
             horizontalalignment='left', verticalalignment='top')
plt.xlabel('PC1 (scikit-learn)')
plt.ylabel('PC2 (scikit-learn)')
plt.show()

# PCA (eigenvalues and eigen_vectors)
eigenvalue_eig, loading_vector_eig = np.linalg.eig(autoscaled_data.transpose().dot(autoscaled_data))
scoreT_eig = np.array(autoscaled_data.dot(loading_vector_eig))
# plot
plt.scatter(scoreT_eig[:, 0], scoreT_eig[:, 1])
for number_of_samples in np.arange(0, raw_data.shape[0] - 1):
    plt.text(scoreT_eig[number_of_samples, 0], scoreT_eig[number_of_samples, 1], raw_data.index[number_of_samples],
             horizontalalignment='left', verticalalignment='top')
plt.xlabel('PC1 (eigen)')
plt.ylabel('PC2 (eigen)')
plt.show()

# PCA (NIPALS)
max_component_number = np.linalg.matrix_rank(autoscaled_data)
scoreT_nipals = np.empty((autoscaled_data.shape[0], autoscaled_data.shape[1]))
loading_vector_nipals = np.empty((autoscaled_data.shape[1], max_component_number))
X = autoscaled_data
for component_number in np.arange(0, max_component_number):
    max_variance_index_number = np.where(X.var(axis=1) == X.var(axis=1).max())[0][0]
    loading_vector_tmp = X[max_variance_index_number - 1:max_variance_index_number].transpose()
    scoreT_tmp_old = X.dot(loading_vector_tmp) / (
        loading_vector_tmp.transpose().dot(loading_vector_tmp)).values.flatten()
    repeat_calculation = True
    while repeat_calculation:
        loading_vector_tmp = X.transpose().dot(scoreT_tmp_old) / (
            scoreT_tmp_old.transpose().dot(scoreT_tmp_old).values.flatten())
        loading_vector_tmp = loading_vector_tmp / np.linalg.norm(loading_vector_tmp)
        scoreT_tmp = X.dot(loading_vector_tmp) / (
            loading_vector_tmp.transpose().dot(loading_vector_tmp)).values.flatten()
        if np.linalg.norm(scoreT_tmp - scoreT_tmp_old) < 10 ** -12:
            repeat_calculation = False
        scoreT_tmp_old = scoreT_tmp

    scoreT_nipals[:, component_number] = np.array(scoreT_tmp).reshape(scoreT_tmp.shape[0])
    loading_vector_nipals[:, component_number] = np.array(loading_vector_tmp).reshape(loading_vector_tmp.shape[0])
    X = X - scoreT_tmp.dot(loading_vector_tmp.transpose())
# plot
plt.scatter(scoreT_nipals[:, 0], scoreT_nipals[:, 1])
for number_of_samples in np.arange(0, raw_data.shape[0] - 1):
    plt.text(scoreT_nipals[number_of_samples, 0], scoreT_nipals[number_of_samples, 1],
             raw_data.index[number_of_samples],
             horizontalalignment='left', verticalalignment='top')
plt.xlabel('PC1 (NIPALS)')
plt.ylabel('PC2 (NIPALS)')
plt.show()
