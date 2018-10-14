# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
課題2: iris.csvを読み込み共分散と相関係数を求めそれぞれcovariance.csv, correlation_coefficient.csvに保存するプログラムを
作成せよ。自分なりの方法で各統計量があっているか確認せよ。
"""

import pandas as pd

raw_data = pd.read_csv('iris.csv', encoding='SHIFT-JIS', index_col=0)

covariance = raw_data.cov()
covariance.to_csv('covariance.csv')

correlation_coefficient = raw_data.corr()
correlation_coefficient.to_csv('correlation_coefficient.csv')
