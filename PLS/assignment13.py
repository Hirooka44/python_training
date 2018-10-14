# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
課題13: iris_withspecies.csvを読み込み、ランダムに90サンプルを選択してトレーニングデータ(モデル構築用データ)とし、残
りをテストデータ(モデル検証用データ)として、LDAモデルの作成およびモデルを用いた予測を行え。
この際、setosaを１つのクラス、versicolorとvirginicaとを合わせたものをもう一つのクラスとして、２クラス分類をすること。
またオートスケーリングを行うこと。さらに、トレーニングデータとテストデータそれぞれにおいて、実際のクラスと計算もしくは
予測されたクラスとの間で混同行列を計算せよ。
"""

import pandas as pd
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

do_autoscaling = True  # True or False
number_of_training_data = 90  # if this is the number of all samples, there are no test samples.

# load data set
raw_data_with_y = pd.read_csv('iris_withspecies.csv', encoding='SHIFT-JIS', index_col=0)
raw_data_with_y.iloc[50:, 0] = 'versicolor+virginica'

y = raw_data_with_y.iloc[:, 0]
X = raw_data_with_y.iloc[:, 1:]
if number_of_training_data < X.shape[0]:
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=number_of_training_data, random_state=0)
else:
    Xtrain = X.copy()
    ytrain = y.copy()
    Xtest = X.copy()
    ytest = y.copy()

# import random
# training_sample_number = random.sample(range(raw_data_with_y.shape[0]),number_of_training_data)
# test_sample_number = list(set(range(raw_data_with_y.shape[0]))-set(training_sample_number))
# ytrain = raw_data_with_y.iloc[training_sample_number,0]
# Xtrain = raw_data_with_y.iloc[training_sample_number,1:]
# ytest = raw_data_with_y.iloc[test_sample_number,0]
# Xtest = raw_data_with_y.iloc[test_sample_number,1:]

# autoscaling
if do_autoscaling:
    autoscaled_Xtrain = (Xtrain - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)
    autoscaled_Xtest = (Xtest - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)
else:
    autoscaled_Xtrain = Xtrain.copy()
    autoscaled_Xtest = Xtest.copy()

# LDA
lda_results = LinearDiscriminantAnalysis()
lda_results.fit(autoscaled_Xtrain, ytrain)

# confusion matrix between actual y and calculated y
calculated_ytrain = lda_results.predict(autoscaled_Xtrain)
confusion_matrix_train = metrics.confusion_matrix(ytrain, calculated_ytrain, labels=sorted(set(ytrain)))
print('training samples')
print(sorted(set(ytrain)))
print(confusion_matrix_train)

# confusion matrix between actual y and predicted y
if number_of_training_data < X.shape[0]:
    predicted_ytest = lda_results.predict(autoscaled_Xtest)
    confusion_matrix_test = metrics.confusion_matrix(ytest, predicted_ytest, labels=sorted(set(ytrain)))
    print('')
    print('test samples')
    print(sorted(set(ytrain)))
    print(confusion_matrix_test)
