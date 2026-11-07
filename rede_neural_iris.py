#=============================================
# 1. Importação e Configurações
#=============================================
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

#Setando pesos iniciais aleatórios
np.random.seed(42)


#=============================================
# 2. Pré-processamento de dados
#=============================================
iris = load_iris()
X = iris.data                      # (150, 4) -> 4 características (comprimento e largura da sépala e da pétala)
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

#print("Treino:", X_train.shape, Y_train.shape)
#print("Teste: ", X_test.shape, Y_test.shape)


#=============================================
# 3. Funções de ativação e perda
#=============================================
def relu(Z):                       # Aplica a fórmula ReLU (negativos viram 0 e positivos passam iguais)
    return np.maximum(0, Z)    # np.maximum aplica a fórmula para toda a matriz

def relu_derivada(Z):        # Gera uma matriz de verdadeiro/falso
    return (Z > 0).astype(float)     # .astype(float) converte em 1.0 e 0.0

def softmax(Z):            # Implementa a fórmula Softmax
    Z_estavel = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_estavel)       # np.exp(...) deixa tudo positivo.
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def cross_entropy(Y_real, Y_prev):      # Implementa a Entropia Cruzada
    n = Y_real.shape[0]
    Y_prev = np.clip(Y_prev, 1e-12, 1.0)
    return -np.sum(Y_real * np.log(Y_prev)) / n

#### Teste ####
#print(relu(np.array([-2, 0, 3])))
#print(softmax(np.array([[3.2, 1.1, -0.5]])))


#=============================================
# 4. Arquitetura e parâmetros
#=============================================
n_entrada = 4      # 4 características do Iris
n_escondida = 8    # 8 neurônios na camada escondida
n_saida = 3        # 3 classes

# Pesos aleatórios (inicialização He, adequada para ReLU)
W1 = np.random.randn(n_entrada, n_escondida) * np.sqrt(2 / n_entrada)
b1 = np.zeros((1, n_escondida))

W2 = np.random.randn(n_escondida, n_saida) * np.sqrt(2 / n_escondida)
b2 = np.zeros((1, n_saida))

#print("W1:", W1.shape, "b1:", b1.shape)
#print("W2:", W2.shape, "b2:", b2.shape)


#=============================================
# 5. Treinamento via Gradiente Descendente
#=============================================
taxa_aprendizado = 0.1
epocas = 1000
n = X_train.shape[0]   # 120 amostras de treino

for epoca in range(epocas):
    # FORWARD PASS
    Z1 = X_train @ W1 + b1     # (120, 8) soma ponderada da camada escondida
    A1 = relu(Z1)              # (120, 8) ativação ReLU
    Z2 = A1 @ W2 + b2          # (120, 3) soma ponderada da camada de saída
    A2 = softmax(Z2)           # (120, 3) probabilidades

    # CÁLCULO DA PERDA (Mede o erro médio)
    perda = cross_entropy(Y_train, A2)

    # BACKWARD PASS (BACKPROPAGATION)
    dZ2 = (A2 - Y_train) / n                 # (120, 3)
    dW2 = A1.T @ dZ2                         # (8, 3)
    db2 = np.sum(dZ2, axis=0, keepdims=True) # (1, 3)

    dA1 = dZ2 @ W2.T                         # (120, 8)
    dZ1 = dA1 * relu_derivada(Z1)            # (120, 8)
    dW1 = X_train.T @ dZ1                    # (4, 8)
    db1 = np.sum(dZ1, axis=0, keepdims=True) # (1, 8)

    # ATUALIZAÇÃO DOS PESOS
    W2 -= taxa_aprendizado * dW2
    b2 -= taxa_aprendizado * db2
    W1 -= taxa_aprendizado * dW1
    b1 -= taxa_aprendizado * db1

    if epoca % 100 == 0 or epoca == epocas - 1:
        print(f"Época {epoca:4d} | Perda: {perda:.4f}")


#=============================================
# 4. Avaliação do Modelo
#=============================================
def prever(X):   # Mesmo Forward pass do treino, sem backward
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return A2

# Inferência no conjunto de teste
probs_teste = prever(X_test)
classes_previstas = np.argmax(probs_teste, axis=1)
classes_reais = np.argmax(Y_test, axis=1)

acuracia_teste = np.mean(classes_previstas == classes_reais)

# Acurácia no treino, para comparação
acuracia_treino = np.mean(
    np.argmax(prever(X_train), axis=1) == np.argmax(Y_train, axis=1)
)

print(f"\nAcurácia no treino: {acuracia_treino * 100:.2f}%")
print(f"Acurácia no teste:  {acuracia_teste * 100:.2f}%")