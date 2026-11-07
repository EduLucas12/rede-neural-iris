#==================================
# 1. Importação e Configurações
#==================================

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

#Setando pesos iniciais aleatórios
np.random.seed(42)

#==================================
#2. Pré-processamento de dados
#==================================

iris = load_iris()
x = iris.data                      # (150, 4) -> 4 características (comprimento e largura da sépala e da pétala)
y = iris.target.reshape(-1, 1)     # (150, 1) -> classes 0, 1 ou 2

# One-Hot Encoding: 0 -> [1,0,0] (Setosa), 1 -> [0,1,0] (Versicolor), 2 -> [0,0,1] (Virginica)
encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y)     # (150, 3)

# Divisão 80% treino / 20% teste
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=y
)

# Padronização (média 0, desvio 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Treino:", X_train.shape, Y_train.shape)
print("Teste: ", X_test.shape, Y_test.shape)