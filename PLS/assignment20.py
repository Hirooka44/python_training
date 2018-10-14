# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
課題20: RDKitを活用して、BRICSアルゴリズムにより仮想的な化学反応にもとづいて化学構造を生成して、生成した構造を
generated_structures.sdfに保存するプログラムを作成せよ。
BRICS ... http://onlinelibrary.wiley.com/doi/10.1002/cmdc.200800178/abstract
"""

import random

from rdkit import Chem
from rdkit.Chem import AllChem, BRICS

random.seed(1)
max_number_of_generated_structures = 100

molecules = [molecule for molecule in Chem.SDMolSupplier('logSdataset1290_2d.sdf') if molecule is not None]
# molecules = [molecule for molecule in Chem.SmilesMolSupplier('logSdataset1290_2d.smi',
#                                                              delimiter='\t', titleLine=False)
#              if molecule is not None]

print(len(molecules))
fragments = set()
for molecule in molecules:
    fragment = BRICS.BRICSDecompose(molecule, minFragmentSize=2)
    #    print(fragment)
    #    print(list(BRICS.FindBRICSBonds(molecule)))
    fragments.update(fragment)
print(len(fragments))
# print (fragments)

generated_structures = BRICS.BRICSBuild([Chem.MolFromSmiles(smiles) for smiles in fragments])
writer = Chem.SDWriter('generated_structures.sdf')
# writer = Chem.SmilesWriter('generated_structures.smi')
number_of_generated_structures = 0
for generated_structure in generated_structures:
    generated_structure.UpdatePropertyCache(True)
    AllChem.Compute2DCoords(generated_structure)
    writer.write(generated_structure)
    number_of_generated_structures += 1
    if number_of_generated_structures >= max_number_of_generated_structures:
        break
writer.close()
