# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
課題8: iris_withspecies.csvを読み込みオートスケーリングをしてから、scikit-learnを使って線形判別分析
(Linear Discriminant Analysis, LDA)を行え。この際、setosaを１つのクラス、versicolorとvirginicaとを合わせたものを
もう一つのクラスとして、２クラス分類をすること。さらに、実際のクラスと計算されたクラスとの間で混同行列を計算せよ。
"""

import pandas as pd
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

do_autoscaling = True  # True or False

# load data set
raw_data_with_y = pd.read_csv('iris_withspecies.csv', encoding='SHIFT-JIS', index_col=0)
raw_data_with_y.iloc[50:, 0] = 'versicolor+virginica'
y = raw_data_with_y.iloc[:, 0]
X = raw_data_with_y.iloc[:, 1:]

# autoscaling
if do_autoscaling:
    autoscaled_X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
else:
    autoscaled_X = X.copy()

# LDA
lda_results = LinearDiscriminantAnalysis()
lda_results.fit(autoscaled_X, y)

# confusion matrix between actual y and calculated y
calculated_y = lda_results.predict(autoscaled_X)
confusion_matrix = metrics.confusion_matrix(y, calculated_y, labels=sorted(set(y)))
print(sorted(set(y)))
print(confusion_matrix)
