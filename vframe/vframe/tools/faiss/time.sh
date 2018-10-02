#!/bin/bash

indexes=('Flat' 'PCA80,Flat' 'IVF512,Flat' 'IVF512,SQ4' 'IVF512,SQ8' 'PCAR8,IMI2x10,SQ8')
# indexes=('Flat' 'PCA80,Flat' 'IVF4096,Flat' 'IVF16384,Flat' 'IVF4096,SQ4' 'IVF4096,SQ8' 'IVF16384,SQ4' 'IVF16384,SQ8' 'PCAR8,IMI2x10,SQ8')
# indexes=('IVF512,SQ4' 'IVF4096,SQ8' 'PCAR8,IMI2x10,SQ8')

for index in ${indexes[*]}
do
  python train.py --dataset alexnet --factory_type $index --store_index
done
