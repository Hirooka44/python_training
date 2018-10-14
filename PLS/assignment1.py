# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
課題1: iris.csvを読み込み最大値・最小値・平均値・中央値・分散・標準偏差を求めbasic_statistics.csvに保存するプログラムを
作成せよ。自分なりの方法で各統計量があっているか確認せよ。
"""

import pandas as pd

raw_data = pd.read_csv('iris.csv', encoding='SHIFT-JIS', index_col=0)

basic_statistics = pd.concat(
    [raw_data.max(), raw_data.min(), raw_data.mean(), raw_data.median(), raw_data.var(ddof=1), raw_data.std(ddof=1)],
    axis=1)
basic_statistics.columns = ['max', 'min', 'average', 'median', 'var', 'std']
basic_statistics.to_csv('basic_statistics.csv')
