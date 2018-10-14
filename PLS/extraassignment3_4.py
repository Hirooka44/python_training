# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
発展課題3: scikit-learnを使わずに、NIPALSアルゴリズムによる方法でPLSを行うプログラムを作成せよ。
scikit-learnのPLSの結果と等しくなることを確認せよ。
発展課題4: scikit-learnを使わずに、SIMPLSアルゴリズムによる方法でPLSを行うプログラムを作成せよ。
scikit-learnのPLSの結果と等しくなることを確認せよ。
"""

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

regression_method_flag = 2  # 1:PLS(scikit-learn), 2:PLS(nipals), 3:PLS(simpls),
pls_component_number = 2  # constant component number
threshold_of_rate_of_same_value = 0.79
do_autoscaling = True  # True or False

# load data set
raw_data_with_y = pd.read_csv('logSdataset1290.csv', encoding='SHIFT-JIS', index_col=0)

raw_data_with_y = raw_data_with_y.loc[:, raw_data_with_y.mean().index]  # 平均を計算できる変数だけ選択
# raw_data_with_y = raw_data_with_y.loc[raw_data_with_y.mean(axis=1).index,:] #平均を計算できるサンプルだけ選択

raw_data_with_y = raw_data_with_y.replace(np.inf, np.nan).fillna(np.nan)  # infをnanに置き換えておく
raw_data_with_y = raw_data_with_y.dropna(axis=1)  # nanのある変数を削除
# raw_data_with_y = raw_data_with_y.dropna() #nanのあるサンプルを削除

# logSdataset1290.csv用 変数選択により精度向上
# raw_data_with_y = raw_data_with_y.drop(['Ipc','Kappa3'],axis=1) #Ipc:1139だけ外れ値をもつ記述子、
# Kappa3:889だけ外れ値をもつ記述子

# delete duplicates
# raw_data_with_y = raw_data_with_y.loc[~raw_data_with_y.index.duplicated(keep='first'),:] #重複したサンプルの最初を残す
# raw_data_with_y = raw_data_with_y.loc[~raw_data_with_y.index.duplicated(keep='last'),:] #重複したサンプルの最後を残す
raw_data_with_y = raw_data_with_y.loc[~raw_data_with_y.index.duplicated(keep=False), :]  # 重複したサンプルはすべて削除
y = raw_data_with_y[raw_data_with_y.columns[0]]
rawX = raw_data_with_y[raw_data_with_y.columns[1:]]
rawX_tmp = rawX.copy()

# delete descriptors with high rate of the same values
rate_of_same_value = list()
num = 0
for X_variable_name in rawX.columns:
    num += 1
    #    print( '{0} / {1}'.format(num,rawX.shape[1]) )
    same_value_number = rawX[X_variable_name].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / rawX.shape[0]))
deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)

"""
# delete descriptors with zero variance
deleting_variable_numbers = np.where( rawX.var() == 0 )
"""

if len(deleting_variable_numbers[0]) == 0:
    X = rawX
else:
    X = rawX.drop(rawX.columns[deleting_variable_numbers], axis=1)
    print('Variable numbers zero variance: {0}'.format(deleting_variable_numbers[0] + 1))

print('# of X-variables: {0}'.format(X.shape[1]))

# autoscaling
if do_autoscaling:
    autoscaled_X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    autoscaled_y = (y - y.mean()) / y.std(ddof=1)
else:
    autoscaled_X = X
    autoscaled_y = y

if regression_method_flag == 1:  # PLS(scikit-learn)
    regression_model = PLSRegression(n_components=pls_component_number)
    regression_model.fit(autoscaled_X, autoscaled_y)
    calculated_y = np.ndarray.flatten(regression_model.predict(autoscaled_X))
elif regression_method_flag == 2:  # PLS(nipals)
    autoscaled_X_tmp = autoscaled_X.copy()
    autoscaled_y_tmp = autoscaled_y.copy()
    weight_w = np.empty((autoscaled_X.shape[1], pls_component_number))
    scoreT = np.empty((autoscaled_X.shape[0], pls_component_number))
    x_loading_p = np.empty((autoscaled_X.shape[1], pls_component_number))
    y_loading_q = np.empty((1, pls_component_number))
    for pls_component in range(0, pls_component_number):
        weight_w_tmp = autoscaled_X_tmp.transpose().dot(autoscaled_y) / np.linalg.norm(
            autoscaled_X_tmp.transpose().dot(autoscaled_y))
        score_t_tmp = autoscaled_X_tmp.dot(weight_w_tmp)
        x_loading_p_tmp = autoscaled_X_tmp.transpose().dot(score_t_tmp) / np.linalg.norm(
            score_t_tmp.transpose().dot(score_t_tmp))
        y_loading_q_tmp = autoscaled_y.transpose().dot(score_t_tmp) / np.linalg.norm(
            score_t_tmp.transpose().dot(score_t_tmp))
        weight_w[:, pls_component] = np.array(weight_w_tmp).reshape(weight_w_tmp.shape[0])
        scoreT[:, pls_component] = np.array(score_t_tmp).reshape(score_t_tmp.shape[0])
        x_loading_p[:, pls_component] = np.array(x_loading_p_tmp).reshape(x_loading_p_tmp.shape[0])
        y_loading_q[:, pls_component] = y_loading_q_tmp
        autoscaled_X_tmp = autoscaled_X_tmp - score_t_tmp.values.reshape(score_t_tmp.shape[0], 1).dot(
            x_loading_p_tmp.values.reshape(1, x_loading_p_tmp.shape[0]))
        autoscaled_y_tmp = autoscaled_y_tmp - score_t_tmp * y_loading_q_tmp
    regression_coefficient = weight_w.dot(np.linalg.inv(x_loading_p.transpose().dot(weight_w))).dot(
        y_loading_q.transpose())
    calculated_y = np.ndarray.flatten(np.array(autoscaled_X.dot(regression_coefficient)))
elif regression_method_flag == 3:  # PLS(simpls)
    weight_w = np.empty((autoscaled_X.shape[1], pls_component_number))
    scoreT = np.empty((autoscaled_X.shape[0], pls_component_number))
    x_loading_p = np.empty((autoscaled_X.shape[1], pls_component_number))
    y_loading_q = np.empty((1, pls_component_number))
    R = np.empty((autoscaled_X.shape[1], pls_component_number))
    V = np.empty((autoscaled_X.shape[1], pls_component_number))
    covarianceXy = (autoscaled_X.transpose().dot(autoscaled_y)).values.reshape(autoscaled_X.shape[1], 1)
    for pls_component in range(0, pls_component_number):
        y_loading_q_tmp = covarianceXy.transpose().dot(covarianceXy)
        R_tmp = covarianceXy.dot(y_loading_q_tmp)
        score_t_tmp = autoscaled_X.dot(R_tmp)
        score_t_tmp = score_t_tmp - score_t_tmp.mean()
        R_tmp = R_tmp / np.linalg.norm(score_t_tmp)
        score_t_tmp = score_t_tmp / np.linalg.norm(score_t_tmp)
        x_loading_p_tmp = autoscaled_X.transpose().dot(score_t_tmp)
        y_loading_q_tmp = autoscaled_y.transpose().dot(score_t_tmp)
        V_tmp = x_loading_p_tmp.copy()
        if pls_component > 0:
            V_tmp = V_tmp - V[:, :pls_component - 1].dot(V[:, :pls_component - 1].transpose().dot(x_loading_p_tmp))
        V_tmp = V_tmp / np.ndarray.flatten(np.array((V_tmp.transpose().dot(V_tmp)) ** 0.5))
        covarianceXy = covarianceXy - V_tmp.dot(V_tmp.transpose().dot(covarianceXy))
        R[:, pls_component] = np.array(R_tmp).reshape(R_tmp.shape[0])
        V[:, pls_component] = np.array(V_tmp).reshape(V_tmp.shape[0])
        weight_w[:, pls_component] = np.array(weight_w_tmp).reshape(weight_w_tmp.shape[0])
        scoreT[:, pls_component] = np.array(score_t_tmp).reshape(score_t_tmp.shape[0])
        x_loading_p[:, pls_component] = np.array(x_loading_p_tmp).reshape(x_loading_p_tmp.shape[0])
        y_loading_q[:, pls_component] = y_loading_q_tmp
    regression_coefficient = R.dot(y_loading_q.transpose())
    calculated_y = np.ndarray.flatten(np.array(autoscaled_X.dot(regression_coefficient)))
if do_autoscaling:
    calculated_y = calculated_y * y.std(ddof=1) + y.mean()

# r2
print('r2: {0}'.format(float(1 - sum((y - calculated_y) ** 2) / sum((y - y.mean()) ** 2))))

# yy-plot
plt.figure(figsize=figure.figaspect(1))
plt.scatter(y, calculated_y)
YMax = np.max(np.array([np.array(y), calculated_y]))
YMin = np.min(np.array([np.array(y), calculated_y]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel('Actual Y')
plt.ylabel('Calculated Y')
plt.show()
