import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import ML_lib.utils as utils
import ML_lib.plot_lib as pltl
import ML_lib.preprocessing as pre

D, L, label_dict = utils.csv_to_npy('data_raw/iris.csv')
print(D.shape)
print(L.shape)
print(label_dict)

pltl.plotHist(D, L, label_dict= label_dict, attr_names= ['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])
pltl.plotSparse(D, L, label_dict= label_dict, attr_names= ['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])

D_norm = pre.Normalizer(D).transform(D)

pltl.plotHist(D_norm, L, label_dict= label_dict, attr_names= ['Sepal length', 'Sepal width', 'Petal length', 'Petal width'], title= 'normalized')
pltl.plotSparse(D_norm, L, label_dict= label_dict, attr_names= ['Sepal length', 'Sepal width', 'Petal length', 'Petal width'], title= 'normalized')

D_stand = pre.Standardizer(D).transform(D)

pltl.plotHist(D_stand, L, label_dict= label_dict, attr_names= ['Sepal length', 'Sepal width', 'Petal length', 'Petal width'], title= 'standardized')
pltl.plotSparse(D_stand, L, label_dict= label_dict, attr_names= ['Sepal length', 'Sepal width', 'Petal length', 'Petal width'], title= 'standardized')
