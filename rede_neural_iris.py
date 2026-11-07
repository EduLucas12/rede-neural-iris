#==================================
# 1. Importação e Configurações
#==================================

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

#Setando pesos iniciais aleatórios
np.random.seed(42)