# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
課題3: まずはオートスケーリング(標準化)について調べよ。次に、iris.csvを読み込みオートスケーリングを行って
autoscaled_data.csvに保存するプログラムを作成せよ。自分なりの方法で結果があっているか確認せよ。
"""

import pandas as pd

raw_data = pd.read_csv('iris.csv', encoding='SHIFT-JIS', index_col=0)

autoscaled_data = (raw_data - raw_data.mean(axis=0)) / raw_data.std(axis=0, ddof=1)
autoscaled_data.to_csv('autoscaled_data.csv')
