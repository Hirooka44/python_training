# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
課題19: RDKitを活用して、logSdataset1290_2d.sdfを読み込み、各化学構造においてfingerprintを計算してfingerprints.csvに
保存するプログラムを作成せよ。ただし各サンプル名を化学構造のSMILES文字列にすること。いろいろな種類のfingerprintを
計算して確認すること。
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

y_name = 'logS'

molecules = [molecule for molecule in Chem.SDMolSupplier('logSdataset1290_2d.sdf') if molecule is not None]
# molecules = [molecule for molecule in Chem.SmilesMolSupplier('logSdataset1290_2d.smi',
#                                                              delimiter='\t', titleLine=False)
#              if molecule is not None]

# fingerprints = [AllChem.GetMorganFingerprintAsBitVect(molecule,5,nBits=1024) for molecule in molecules]
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(molecule, 4, nBits=1024) for molecule in molecules]
# fingerprints = [AllChem.GetMorganFingerprintAsBitVect(molecule,3,nBits=1024) for molecule in molecules]
# fingerprints = [AllChem.GetMACCSKeysFingerprint(molecule) for molecule in molecules]
# fingerprints = [Chem.RDKFingerprint(molecule) for molecule in molecules]

print(fingerprints[1].GetNumBits())
fingerprints = pd.DataFrame(np.array(fingerprints, int))
molecular_names = [molecule.GetProp('_Name') for molecule in molecules]
smiles = [Chem.MolToSmiles(molecule) for molecule in molecules]
fingerprints.index = smiles
fingerprints.to_csv('fingerprints.csv')

Y = pd.DataFrame([float(molecule.GetProp(y_name)) for molecule in molecules])
Y.index = smiles
Y.columns = [y_name]

dataset = pd.concat([Y, fingerprints], axis=1)
dataset.to_csv('dataset.csv')
