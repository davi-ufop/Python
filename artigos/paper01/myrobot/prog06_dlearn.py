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
  imag3 = np.genfromtxt("prog06/data/out_{:04d}.csv".format(i), delimiter=",")
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
  np.savetxt("prog06/data/prev_{:04d}.csv".format(i), figura, delimiter=",") 
  pl.imshow(figura)
  pl.savefig("prog06/resultados/prev_{}.png".format(i))
  pl.clf()
  ### Figuras referentes aos dados reais
  pl.imshow(imag3)
  pl.savefig("prog06/resultados/real_{}.png".format(i))
  pl.clf()
  ### Correlação - Dados e Previsão
  rotulos = ['Data', 'Forecast']
  COR = []
  for k in range(128):
    MCK = np.corrcoef(imag3[:, k], figura[:, k])
    MCK = np.nan_to_num(MCK, nan=1.0)
    COR.append(MCK[0,1])
  CORM = np.mean(COR)
  MCS = np.array([[1, CORM], [CORM, 1]]) 
  # Plotando e salvando
  pl.matshow(abs(MCS), cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Columns)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig("prog06/resultados/corr_{:03d}.png".format(i), dpi=500)
  pl.clf()

### Montages interessante
# imagens
num1, num2 = 481, 497
imagem1 = imread("prog06/imgs/traj_{:04d}.png".format(num1))
imagem2 = imread("prog06/resultados/real_{:03d}.png".format(num1))
imagem3 = imread("prog06/resultados/prev_{:03d}.png".format(num1))
imagem4 = imread("prog06/imgs/traj_{:04d}.png".format(num2))
imagem5 = imread("prog06/resultados/real_{:03d}.png".format(num2))
imagem6 = imread("prog06/resultados/prev_{:03d}.png".format(num2))
# figura settings
pl.figure(figsize=(15,10), dpi=400)
# 1
pl.subplot(2, 3, 1)
pl.imshow(imagem1)
pl.title("( A )", y=-0.15)
# 2
pl.subplot(2, 3, 2)
pl.axis('off')
pl.imshow(imagem2)
pl.title("( B )", y=-0.1)
# 3
pl.subplot(2, 3, 3)
pl.axis('off')
pl.imshow(imagem3)
pl.title("( C )", y=-0.1)
# 4
pl.subplot(2, 3, 4)
pl.imshow(imagem4)
pl.title("( D )", y=-0.15)
# 5
pl.subplot(2, 3, 5)
pl.axis('off')
pl.imshow(imagem5)
pl.title("( E )", y=-0.1)
# 6
pl.subplot(2, 3, 6)
pl.axis('off')
pl.imshow(imagem6)
pl.title("( F )", y=-0.1)
# save
pl.savefig("prog06/montagem1.png")
pl.clf()

### Correlação
# imagens
imagem1 = imread("prog06/imgs/traj_{:04d}.png".format(num1))
imagem2 = imread("prog06/resultados/real_{:03d}.png".format(num1))
imagem3 = imread("prog06/resultados/prev_{:03d}.png".format(num1))
imagem4 = imread("prog06/resultados/corr_{:03d}.png".format(num1))
# figura settings
pl.figure(figsize=(10,10), dpi=500)
# 1
pl.subplot(2, 2, 1)
pl.imshow(imagem1)
pl.title("( A )", y=-0.15)
# 2
pl.subplot(2, 2, 2)
pl.axis('off')
pl.imshow(imagem2)
pl.title("( B )", y=-0.1)
# 3
pl.subplot(2, 2, 3)
pl.axis('off')
pl.imshow(imagem3)
pl.title("( C )", y=-0.1)
# 4
pl.subplot(2, 2, 4)
pl.axis('off')
pl.imshow(imagem4)
pl.title("( D )", y=-0.1)
# save
pl.savefig("prog06/montagem2.png")
pl.clf()

### FIM
