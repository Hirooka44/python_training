# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
課題17: iris_withspecies.csvを読み込み、ランダムに90サンプルを選択してトレーニングデータ(モデル構築用データ)とし、
残りをテストデータ(モデル検証用データ)として、SVMの作成およびSVMを用いた予測を行え。
この際、setosaを１つのクラス、versicolorとvirginicaとを合わせたものをもう一つのクラスとして、２クラス分類をすること。
またオートスケーリングを行うこと。さらに、トレーニングデータとテストデータそれぞれにおいて、実際のクラスと計算もしくは
予測されたクラスとの間で混同行列を計算せよ。カーネル関数として線形カーネルとガウシアンカーネルを用いて結果を比較すること。
ハイパーパラメータは5-fold クロスバリデーション後の正解率が高くなるように設定すること。
"""

import numpy as np
import pandas as pd
from sklearn import metrics, svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, train_test_split

classification_method_flag = 3  # 1:LDA, 2:linear SVM, 3:nonlinear SVM(Gaussian kernel)
do_autoscaling = True  # True or False
number_of_training_data = 90  # if this is the number of all samples, there are no test samples.
linear_svm_cs = 2 ** np.arange(-10, 11, dtype=float)
nonlinear_svm_cs = 2 ** np.arange(-10, 11, dtype=float)
nonlinear_svm_gammas = 2 ** np.arange(-20, 10, dtype=float)
fold_number = 5

# load data set
raw_data_with_y = pd.read_csv('iris_withspecies.csv', encoding='SHIFT-JIS', index_col=0)
raw_data_with_y.iloc[50:, 0] = 'versicolor+virginica'

# For SVM
raw_data_with_y.loc[raw_data_with_y.iloc[:, 0] == 'setosa', raw_data_with_y.columns[0]] = 1
raw_data_with_y.loc[raw_data_with_y.iloc[:, 0] == 'versicolor+virginica', raw_data_with_y.columns[0]] = -1

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

# For SVM
ytrain = list(ytrain)
ytest = list(ytest)

# autoscaling
if do_autoscaling:
    autoscaled_Xtrain = (Xtrain - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)
    autoscaled_Xtest = (Xtest - Xtrain.mean(axis=0)) / Xtrain.std(axis=0, ddof=1)
else:
    autoscaled_Xtrain = Xtrain.copy()
    autoscaled_Xtest = Xtest.copy()

if classification_method_flag == 1:  # LDA
    classification_model = LinearDiscriminantAnalysis()
elif classification_method_flag == 2:  # linear SVM
    linear_svm_in_cv = GridSearchCV(svm.SVC(kernel='linear'), {'C': linear_svm_cs}, cv=fold_number)
    linear_svm_in_cv.fit(autoscaled_Xtrain, ytrain)
    optimal_linear_svm_c = linear_svm_in_cv.best_params_['C']
    classification_model = svm.SVC(kernel='linear', C=optimal_linear_svm_c)
elif classification_method_flag == 3:  # nonlinear SVM
    nonlinear_svm_in_cv = GridSearchCV(svm.SVC(kernel='rbf'), {'C': nonlinear_svm_cs, 'gamma': nonlinear_svm_gammas},
                                       cv=fold_number)
    nonlinear_svm_in_cv.fit(autoscaled_Xtrain, ytrain)
    optimal_nonlinear_svm_c = nonlinear_svm_in_cv.best_params_['C']
    optimal_nonlinear_svm_gamma = nonlinear_svm_in_cv.best_params_['gamma']
    classification_model = svm.SVC(kernel='rbf', C=optimal_nonlinear_svm_c, gamma=optimal_nonlinear_svm_gamma)

classification_model.fit(autoscaled_Xtrain, ytrain)

# confusion matrix between actual y and calculated y
calculated_ytrain = classification_model.predict(autoscaled_Xtrain)
confusion_matrix_train = metrics.confusion_matrix(ytrain, calculated_ytrain, labels=sorted(set(ytrain)))
print('training samples')
print(sorted(set(ytrain)))
print(confusion_matrix_train)

# confusion matrix between actual y and predicted y
if number_of_training_data < X.shape[0]:
    predicted_ytest = classification_model.predict(autoscaled_Xtest)
    confusion_matrix_test = metrics.confusion_matrix(ytest, predicted_ytest, labels=sorted(set(ytrain)))
    print('')
    print('test samples')
    print(sorted(set(ytrain)))
    print(confusion_matrix_test)
