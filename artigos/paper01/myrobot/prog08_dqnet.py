### Programa para implementar DQN
### Davi Neves - Jan. 2022

###### BIBLIOTECAS E MÓDULOS
import os                             ### Sistema Operacional
import numpy as np                    ### Numérica
import pylab as pl                    ### Gráfica
from tqdm import tqdm                 ### Barra de progresso
### PyTorch
from torch import nn, Tensor, optim

### Pra evitar warnings irrelevantes
import warnings
warnings.filterwarnings('ignore')

### Diretórios
DIRD = "prog08/data/"
DIRT = "prog08/tmp/"
DIRI = "prog08/imgs/"

###### PARÂMETROS
qtable1 = np.genfromtxt(DIRD+"qtable001.csv", delimiter=",")
R = 2                   ### Alcance do braço -> vide mylib07 
B = R/np.sqrt(2)        ### Diagonal do alcance
dd = 0.04               ### Distância entre estados -> vide mylib07
NL = len(qtable1)       ### Número de linhas/estados = 3868
AN = 9                  ### Número de ações
NS = NL*AN              ### Tamanho da saída = 34812
ND = 300                ### Quantidade de Dados (Q-tabelas)
NT = ND-1               ### Quantidade usadas no Trenamento
QT = 20                 ### Quantidade de treinos

###### CONSTRUINDO A REDE NEURAL ARTIFICIAL
### Rede Neural: 2-4-8-16-32-64-128-512-2048-8192-16384->dim
camadas = []
camadas.append(nn.Linear(2, 4))
camadas.append(nn.ReLU())
camadas.append(nn.Linear(4, 8))
camadas.append(nn.ReLU())
camadas.append(nn.Linear(8, 16))
camadas.append(nn.ReLU())
camadas.append(nn.Linear(16, 32))
camadas.append(nn.ReLU())
camadas.append(nn.Linear(32, 64))
camadas.append(nn.ReLU())
camadas.append(nn.Linear(64, 128))
camadas.append(nn.ReLU())
camadas.append(nn.Linear(128, 512))
camadas.append(nn.ReLU())
camadas.append(nn.Linear(512, 2048))
camadas.append(nn.ReLU())
camadas.append(nn.Linear(2048, 8192))
camadas.append(nn.ReLU())
camadas.append(nn.Linear(8192, 16384))
camadas.append(nn.ReLU())
camadas.append(nn.Linear(16384, NS))
camadas.append(nn.Tanhshrink())
rede = nn.Sequential(*camadas)

###### PREPARANDO DADOS
### Entradas
print("\nPreparando as entradas:")
state1 = np.genfromtxt(DIRD+"state001.csv", delimiter=",")
state1 = np.array([[state1[0]], [state1[1]]]).T
state2 = np.genfromtxt(DIRD+"state002.csv", delimiter=",")
state2 = np.array([[state2[0]], [state2[1]]]).T
pilha = np.concatenate((state1, state2), axis=0)
for i in tqdm(range(2, ND)):
  state = np.genfromtxt(DIRD+"state{:03d}.csv".format(i+1), delimiter=",")
  state = np.array([[state[0]], [state[1]]]).T
  pilha = np.concatenate((pilha, state), axis=0)
entradas = Tensor(pilha)
### Saidas
print("\nPreparando as saídas:")
qtab1 = np.genfromtxt(DIRD+"qtable001.csv", delimiter=",")
qtab1 = np.reshape(qtab1, (1, NS))
qtab2 = np.genfromtxt(DIRD+"qtable002.csv", delimiter=",")
qtab2 = np.reshape(qtab2, (1, NS))
stack = np.concatenate((qtab1, qtab2), axis=0)
for i in tqdm(range(2, ND)):
  qtab = np.genfromtxt(DIRD+"qtable{:03d}.csv".format(i+1), delimiter=",")
  qtab = np.reshape(qtab, (1, NS))
  stack = np.concatenate((stack, qtab), axis=0)
saidas = Tensor(stack)

###### TREINAMENTO
### Funções e listas para o Treinamento
lis_perda = []
perda = nn.L1Loss()                                 ### Função perda
gradiente = optim.SGD(rede.parameters(), lr=0.01)   ### Ajuste de pesos
### Treinando!
print("\nTreinos com ", NT," dados:")
for treino in tqdm(range(QT)):
  rotulos = rede(entradas[0:NT,:])         ### Previsões com pesos atuais
  erros = perda(rotulos, saidas[0:NT,:])   ### Erros das previsões
  lis_perda.append(erros.item())          ### Anotando os erros
  gradiente.zero_grad()                    ### Atualiza os pesos
  erros.backward()                         ### Backpropagation
  gradiente.step()                         ### Ajusta os parâmetros da rede

### Função perda
pl.plot(lis_perda, 'r-')
pl.xlabel('train steps')
pl.ylabel('loss function')
pl.savefig(DIRI+'lossfunction.png', dpi=200)

###### TESTANDO A Q-TABLE
from mylib04 import plot_estados, testeQ
### Criando os argumentos de testeQ()
saved = DIRT+"estados.png"
EN, TAG, TXY = plot_estados(R, dd, saved)
xo = entradas.detach().numpy()[NT,0]
yo = entradas.detach().numpy()[NT,1]
xb = -0.96    ### Estado inicial
yb = -0.72    ### do braço robótico
saveme = DIRI+"resultado.png"

### Criando a Q-table específica para xo e yo usando
### a  rede neural - parte mais importante do código
Q_table = rede(entradas[NT,:])

### Preparando a Q-table
qtable = Q_table.detach().numpy()
qtable = np.reshape(qtable, (NL, AN))
### Testando
le = testeQ(qtable, TXY, xb, yb, xo, yo, dd, saveme, ND)
np.savetxt("prog08/qtabprev.csv", qtable, delimiter=",")

print("Estados da trajetória:\n", le[0:5])

###### Deletando os arquivos temporários
os.system("rm -rf prog08/tmp/*")

### FIM
