# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
課題9: logSdataset1290.csvを読み込み、オートスケーリングしてからlogSと構造記述子との間で最小二乗法による重回帰分析を行え。
計算できない場合は、その原因について調べて対処法を実行せよ。r2の計算、logSの実測値と計算値とのプロット、標準回帰係数の
standard_regression_coefficients.csvへの保存を実施せよ。

課題10: logSdataset1290.csvを読み込み、オートスケーリングしてからlogSと構造記述子との間でPLSを行え。成分数は2とする。
r2の計算、logSの実測値と計算値とのプロット、標準回帰係数のstandard_regression_coefficients.csvへの保存を実施せよ。
課題9の結果と比較して考察せよ

課題11: logSdataset1290.csvを読み込み、オートスケーリングしてからlogSと構造記述子との間でPLSを行え。50成分までの
成分数ごとのr2・r2cvのプロット、最適成分数におけるlogSの実測値と計算値とのプロットおよびlogSの実測値と
クロスバリデーション推定値とのプロットを作成せよ。クロスバリデーションは2-fold クロスバリデーションとし、
最適成分数はr2CVが最大のものとする。

課題12: 課題11で50成分までにr2cvの値が小さい(たとえば負の値になる)ことがあるときはその原因を調べて対応し、
r2cvを向上させよ。ただし、化合物は削除しないこととする。
"""

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression

regression_method_flag = 3  # 1:OLS, 2:PLS(constant component number), 3:PLS,
pls_component_number = 2
max_pls_component_number = 50
fold_number = 5
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
# raw_data_with_y = raw_data_with_y.drop(['Ipc','Kappa3'],axis=1) #Ipc:1139だけ外れ値をもつ記述子、Kappa3:889だけ外れ値をもつ記述子

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
    print('{0} / {1}'.format(num, rawX.shape[1]))
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

if regression_method_flag == 1:  # Ordinary Least Squares
    regression_model = LinearRegression()
elif regression_method_flag == 2:  # Partial Least Squares with constant component
    regression_model = PLSRegression(n_components=pls_component_number)
elif regression_method_flag == 3:  # Partial Least Squares
    pls_components = np.arange(1, min(np.linalg.matrix_rank(autoscaled_X) + 1, max_pls_component_number + 1), 1)
    r2all = list()
    r2cvall = list()
    for pls_component in pls_components:
        pls_model_in_cv = PLSRegression(n_components=pls_component)
        pls_model_in_cv.fit(autoscaled_X, autoscaled_y)
        calculated_y_in_cv = np.ndarray.flatten(pls_model_in_cv.predict(autoscaled_X))
        estimated_y_in_cv = np.ndarray.flatten(
            model_selection.cross_val_predict(pls_model_in_cv, autoscaled_X, autoscaled_y, cv=fold_number))
        if do_autoscaling:
            calculated_y_in_cv = calculated_y_in_cv * y.std(ddof=1) + y.mean()
            estimated_y_in_cv = estimated_y_in_cv * y.std(ddof=1) + y.mean()
        r2all.append(float(1 - sum((y - calculated_y_in_cv) ** 2) / sum((y - y.mean()) ** 2)))
        r2cvall.append(float(1 - sum((y - estimated_y_in_cv) ** 2) / sum((y - y.mean()) ** 2)))
    plt.plot(pls_components, r2all, 'bo-')
    plt.plot(pls_components, r2cvall, 'ro-')
    plt.ylim(0, 1)
    plt.xlabel('Number of PLS components')
    plt.ylabel('r2(blue), r2cv(red)')
    plt.show()
    optimal_pls_component_number = np.where(r2cvall == np.max(r2cvall))
    optimal_pls_component_number = optimal_pls_component_number[0][0] + 1
    regression_model = PLSRegression(n_components=optimal_pls_component_number)

regression_model.fit(autoscaled_X, autoscaled_y)

# calculate y
calculated_y = np.ndarray.flatten(regression_model.predict(autoscaled_X))
estimated_y = np.ndarray.flatten(
    model_selection.cross_val_predict(regression_model, autoscaled_X, autoscaled_y, cv=fold_number))
if do_autoscaling:
    calculated_y = calculated_y * y.std(ddof=1) + y.mean()
    estimated_y = estimated_y * y.std(ddof=1) + y.mean()

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

plt.figure(figsize=figure.figaspect(1))
plt.scatter(y, estimated_y)
YMax = np.max(np.array([np.array(y), estimated_y]))
YMin = np.min(np.array([np.array(y), estimated_y]))
plt.plot([YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)],
         [YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin)], 'k-')
plt.ylim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlim(YMin - 0.05 * (YMax - YMin), YMax + 0.05 * (YMax - YMin))
plt.xlabel('Actual Y')
plt.ylabel('Estimated Y')
plt.show()

# standard regression coefficients
standard_regression_coefficients = regression_model.coef_
standard_regression_coefficients = pd.DataFrame(standard_regression_coefficients)
standard_regression_coefficients.index = X.columns
standard_regression_coefficients.columns = ['standard regression coefficient']
standard_regression_coefficients.to_csv('standard_regression_coefficients.csv')
