# Rede Neural do Zero: Classificação do Dataset Iris

Atividade AECO da disciplina de Inteligência Artificial (UNIPAC).

Implementação de uma rede neural artificial **do zero, usando apenas NumPy**, para classificar as três espécies de flores do dataset Iris. O Scikit-Learn é usado somente para carregar e preparar os dados. Toda a rede (ativações, função de perda, backpropagation e gradiente descendente) foi escrita manualmente.

## Arquitetura

```
Entrada (4)  ->  Camada escondida (8, ReLU)  ->  Saída (3, Softmax)
```

| Camada | Neurônios | Ativação | Parâmetros |
|---|---|---|---|
| Entrada | 4 (comprimento e largura da sépala e da pétala) | - | - |
| Escondida | 8 | ReLU | W1 (4x8), b1 (1x8) |
| Saída | 3 (Setosa, Versicolor, Virginica) | Softmax | W2 (8x3), b2 (1x3) |

## Etapas do código

1. **Pré-processamento**
   - Rótulos convertidos para One-Hot Encoding (`0 -> [1, 0, 0]`)
   - Divisão estratificada: 80% treino (120 amostras) e 20% teste (30 amostras)
   - Padronização das features (média 0, desvio 1), com estatísticas calculadas apenas no treino para evitar *data leakage*

2. **Funções implementadas com NumPy**
   - `relu` e `relu_derivada`
   - `softmax`, com subtração do máximo para estabilidade numérica
   - `cross_entropy`, com `clip` para evitar `log(0)`

3. **Inicialização**
   - Pesos aleatórios com inicialização He (adequada para ReLU)
   - Vieses iniciados com zero
   - Semente fixa (`np.random.seed(42)`) para resultados reproduzíveis

4. **Treinamento**
   - Forward pass: `Z1 = X·W1 + b1 -> ReLU -> Z2 = A1·W2 + b2 -> Softmax`
   - Perda: entropia cruzada
   - Backpropagation pela regra da cadeia
   - Atualização: `W = W - taxa_aprendizado * gradiente`
   - Hiperparâmetros: taxa de aprendizado `0.1` e `1000` épocas

5. **Avaliação**
   - Inferência no conjunto de teste e cálculo da acurácia

## Resultados

Evolução da perda (entropia cruzada) no treino:

| Época | Perda |
|---|---|
| 0 | 1.5358 |
| 100 | 0.2433 |
| 300 | 0.1021 |
| 500 | 0.0689 |
| 999 | 0.0488 |

| Conjunto | Acurácia |
|---|---|
| Treino | **98,33%** (118 de 120) |
| Teste | **96,67%** (29 de 30) |

A diferença pequena entre treino e teste indica que a rede **generalizou bem**, sem sinais de overfitting.

## Como executar

```bash
git clone https://github.com/EduLucas12/rede-neural-iris.git
cd rede-neural-iris
pip install -r requirements.txt
python rede_neural_iris.py
```

## Tecnologias

- Python 3
- NumPy (implementação da rede)
- Scikit-Learn (apenas carregamento e preparação dos dados)
