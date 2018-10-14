# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
課題18: RDKitを活用して、logSdataset1290_2d.sdfを読み込み、各化学構造において複数の構造記述子を計算してdescriptors.csvに
保存するプログラムを作成せよ。ただし各サンプル名を化学構造のSMILES文字列にすること。
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

y_name = 'logS'

molecules = [molecule for molecule in Chem.SDMolSupplier('logSdataset1290_2d.sdf') if molecule is not None]
# molecules = [molecule for molecule in Chem.SmilesMolSupplier('logSdataset1290_2d.smi',
#                                                              delimiter='\t', titleLine=False)
#              if molecule is not None]

descriptor_names = [descriptor_name[0] for descriptor_name in Descriptors._descList]
print(len(descriptor_names))

descriptor_calculation = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
descriptors = pd.DataFrame([descriptor_calculation.CalcDescriptors(molecule) for molecule in molecules])
descriptors.columns = descriptor_names
molecular_names = [molecule.GetProp('_Name') for molecule in molecules]
smiles = [Chem.MolToSmiles(molecule) for molecule in molecules]
descriptors.index = smiles
descriptors.to_csv('descriptors.csv')

y = pd.DataFrame([float(molecule.GetProp(y_name)) for molecule in molecules])
y.index = smiles
y.columns = [y_name]

dataset = pd.concat([y, descriptors], axis=1)
dataset.to_csv('dataset.csv')
