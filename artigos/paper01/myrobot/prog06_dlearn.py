###### Programa para gerar trajetórias a partir de condições de contorno
###### Davi Neves - Ouro Preto, Brasil - Jan. 2022

###### Módulos e bibliotecas
from imageio import imread
import numpy as np
import pylab as pl
from tqdm import tqdm
### PyTorch
import torch
from torch import nn, flatten, cat, optim
from torchvision.transforms import transforms
### Pra evitar warnings irrelevantes
import warnings
warnings.filterwarnings('ignore')

###### Parâmetros
num_entradas = 4
num_saidas = 3969
NT = 999           ### Treinamento/Aprendizagem
NF = 500           ### Número de figuras
N95 = int(0.95*NF)

###### Rede Neural Artificial: (entrada: 4) -> ... -> (saída: 3969)
camadas = []
camadas.append(nn.Linear(num_entradas, 8))    # Camada de entrada (4 -> 8)
camadas.append(nn.ReLU())
camadas.append(nn.Linear(8, 32))              # Camada 2
camadas.append(nn.ReLU())
camadas.append(nn.Linear(32, 128))            # Camada 3
camadas.append(nn.ReLU())
camadas.append(nn.Linear(128, 512))           # Camada 4
camadas.append(nn.ReLU())
camadas.append(nn.Linear(512, 2048))          # Camada 5
camadas.append(nn.ReLU())
camadas.append(nn.Linear(2048, num_saidas))   # Camada de saída (2048 -> 3069)
camadas.append(nn.Sigmoid())
rede = nn.Sequential(*camadas)
### (Condição de contorno) 4 -> 8 -> 32 -> 128 -> 512 -> 2048 -> 3969 (Trajetória)

###### Dados
transforme = transforms.ToTensor()   ### Transformar imagens em tensores
### entradas
print("\nPreparando as entradas ... ")
# Primeira entrada (condição de contorno: x_braco, y_braco, x_objeto, y_objeto)
imag1 = np.genfromtxt("prog06/data/in_0001.csv", delimiter=',')
tensor1 = transforme(imag1)                  # Transforma a imagem num tensor
tensor1 = flatten(tensor1, start_dim=1)      # Achata o vetor: 2D -> 1D
entradas = tensor1                           # Inicia a construção da lista
# Tensor (lista) com as demais entradas
for i in tqdm(range(1, NF)):  # A partir da segunda figura: 002
  img = np.genfromtxt("prog06/data/in_{:04d}.csv".format(i+1), delimiter=',')
  tensor = transforme(img)                        # Transforma
  tensor_achatado = flatten(tensor, start_dim=1)  # Achata
  entradas = cat((entradas, tensor_achatado), 0)  # Concatena
### saídas
# Primeira saída (trajetória: de y_min até y_max, sendo y(x) a função)
print("\nPreparando as saidas ... ") 
imag2 = np.genfromtxt("prog06/data/out_0001.csv", delimiter=',')  # Importa a imagem
tensor2 = transforme(imag2)                       # Transforma em tensor
polimento = nn.MaxPool2d(kernel_size=4, stride=2) # Polimento da matriz
tensor2 = polimento(tensor2)                      
dim = tensor2.size()[1]                           # Informação fundamental p/ resultados
tensor2 = flatten(tensor2, start_dim=1)           # Achata
saidas = tensor2                                  # Primeiro item da lista
for i in tqdm(range(1, NF)):  # A partir o segundo item
  img = np.genfromtxt("prog06/data/out_{:04d}.csv".format(i+1), delimiter=',')
  tensor = transforme(img)                          # Transformação
  polimento = nn.MaxPool2d(kernel_size=4, stride=2) # Polimento
  tensor_polido = polimento(tensor)           
  tensor_achatado = flatten(tensor_polido, start_dim=1) # Achatamento
  saidas = cat((saidas, tensor_achatado), 0)            # Concatena
### Necessário: Converter o formato dos dados: double -> float
entradas = torch.tensor(entradas).float()
saidas = torch.tensor(saidas).float()

###### Treinamento, agora!
### Função de erro (perda) e otimizador de pesos (cálculo do gradiente)
lis_perda = []
perda = nn.L1Loss()                                 ### Função perda
gradiente = optim.SGD(rede.parameters(), lr=0.01)   ### Ajuste de pesos
### Treinando!
print("\nTreino:")
for treino in tqdm(range(NT)):
  rotulos = rede(entradas[0:N95, :])        ### Previsões com pesos atuais
  erros = perda(rotulos, saidas[0:N95, :])  ### Erros das previsões
  lis_perda.append(erros.item())            ### Anotando os erros
  gradiente.zero_grad()                     ### Atualiza os pesos
  erros.backward()                          ### Backpropagation
  gradiente.step()                          ### Ajusta os parâmetros da rede

###### Função perda
pl.plot(lis_perda, 'r-')
pl.xlabel("steps")
pl.ylabel("loss function value")
pl.ylim(5, 15)
pl.savefig("prog06/lossfunction.png", dpi=200)

###### Resultados
print("\nResultados:")
for i in tqdm(range(N95+1, NF)):
  ### Construindo os dados para entradas e verificações
  imag3 = np.genfromtxt("prog06/data/out_{:04d}.csv".format(i), delimiter=',')
  tensor3 = transforme(imag3)
  polimento = nn.MaxPool2d(kernel_size=4, stride=2, return_indices=True)
  tensor3, indice = polimento(tensor3)
  ### Previsão com a rede
  resultado = rede(entradas[i-1, :])
  resultado = resultado.view(1, dim, dim)
  despolimento = nn.MaxUnpool2d(kernel_size=4, stride=2)
  resultado = despolimento(resultado, indice)
  resultado = resultado.view(128, 128)
  ### Salvando as figuras da previsões
  figura = resultado.detach().numpy()
  pl.imshow(figura)
  pl.savefig("prog06/resultados/prev_{}.png".format(i))
  pl.clf()
  ### Figuras referentes aos dados reais
  pl.imshow(imag3)
  pl.savefig("prog06/resultados/real_{}.png".format(i))
  pl.clf()

### Montagem interessante
num1, num2 = 481, 497
pl.figure(figsize=(15,10))
imagem = imread("prog06/imgs/traj_{:04d}.png".format(num1))
pl.subplot(2, 3, 1)
pl.imshow(imagem)
pl.title("( A )", y=-0.15)
imagem = imread("prog06/resultados/real_{:03d}.png".format(num1))
pl.subplot(2, 3, 2)
pl.axis('off')
pl.imshow(imagem)
pl.title("( B )", y=-0.1)
imagem = imread("prog06/resultados/prev_{:03d}.png".format(num1))
pl.subplot(2, 3, 3)
pl.axis('off')
pl.imshow(imagem)
pl.title("( C )", y=-0.1)
imagem = imread("prog06/imgs/traj_{:04d}.png".format(num2))
pl.subplot(2, 3, 4)
pl.imshow(imagem)
pl.title("( D )", y=-0.15)
imagem = imread("prog06/resultados/real_{:03d}.png".format(num2))
pl.subplot(2, 3, 5)
pl.axis('off')
pl.imshow(imagem)
pl.title("( E )", y=-0.1)
imagem = imread("prog06/resultados/prev_{:03d}.png".format(num2))
pl.subplot(2, 3, 6)
pl.axis('off')
pl.imshow(imagem)
pl.title("( F )", y=-0.1)
pl.savefig("prog06/montagem.png")

### FIM
