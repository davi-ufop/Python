### Código para produzir figuras de trajetórias pendulares
### Davi Neves - Ouro Preto, Brasil - Jan., 2022 

### Importando módulos necessários
from mylib04 import *
import os
import cv2
from tqdm import tqdm
from random import seed, randint, uniform
import warnings
warnings.filterwarnings('ignore')

### Parâmetros e variáveis do processo
R = 2             ### Dimensões do ambiente 
B = R/np.sqrt(2)  ### Alcance máximo do braço
dteta = 0.02      ### Tamanho da variação angular, valor adequado! Não aletere!
tamanho = 0.05    ### Dimensão do objeto
P = 2             ### Precisão das medidas
L1, L2 = R/2, R/2

###### Plotando as trajetórias dos estados: angular e cartesiano
def plot_traj(vxa, vya, caminho):
  ### Plot simples
  pl.figure(figsize=(4, 4), dpi=32)
  pl.plot(vxa, vya, 'r.')
  pl.axis('off')
  pl.savefig(caminho)
  pl.clf()

############ INICIANDO A DIVERSÃO
seed(238)
### Pegando 500 objetos:
for i in tqdm(range(500)):
  ########## CONSTRUINDO AS CONDIÇÕES DE CONTORNO
  ### Definindo a posição do objeto, na mesa!
  xo = round(tamanho*(uniform(-B, B)//tamanho), P)
  yo = round(tamanho*(uniform(-B, 0)//tamanho), P)
  ### Definindo a posição do braço
  xb = round(uniform(-B, B), P)
  yb = round(uniform(-B, 0), P)
  ########## SALVANDO AS ENTRADAS
  matriz = np.array([[xb, yb], [xo, yo]])
  np.savetxt("prog06/data/in_{:04d}.csv".format(i+1), matriz, fmt='%1.2f', delimiter=',')
  ########## LISTAS DE AÇÕES ANGULARES
  ### Determinando os ângulos
  [o1, o2] = angulos_ponto(xo, yo, L1, L2)
  [a1, a2] = angulos_ponto(xb, yb, L1, L2)
  ### Lista dos ângulos das juntas
  la1 =  acoes_listas(o1, a1, dteta)
  la2 =  acoes_listas(o2, a2, dteta)
  la1, la2 = igualistas(la1, la2)
  ########## ESTADOS DO SISTEMA -> CARTESIANOS E ANGULARES
  ### Estados do sistema em ângulos
  va1, va2 = varia_estados(a1, a2, la1, la2)
  ### Trajetória do pêndulo
  vxa = L1*(np.sin(va1)) + L2*(np.sin(va2))
  vya = L1*(np.cos(va1)) + L2*(np.cos(va2))
  ### Plotando as duas trajetórias
  salvem = "prog06/tmp/traj_{:04d}.png".format(i+1)
  plot_traj(vxa, vya, salvem)
  ########## CONSTRUINDO A MATRIZ DE SAÍDAS (TRAJETÓRIAS)
  ### Tranformando em matriz esparsa
  imagem = cv2.imread("prog06/tmp/traj_{:04d}.png".format(i+1))
  cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
  cinza[cinza != 255] = 128
  cinza[cinza == 255] = 0
  ### Saida -> CSV
  np.savetxt("prog06/data/out_{:04d}.csv".format(i+1), cinza, fmt='%1.0f', delimiter=',') 

os.system('rm -rf prog06/tmp/*')
### FIM
