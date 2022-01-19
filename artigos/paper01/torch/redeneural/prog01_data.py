### Programa para montar uma rede neural
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Bibliotecas e Modulos
import os
import cv2
import numpy as np
import pylab as pl
import random as rm
from tqdm import tqdm

### Warnings irrelevantes
### Pra evitar warnings irrelevantes
import warnings
warnings.filterwarnings('ignore')

### Parâmetros
Lx = 35
dx = 0.1
pix = 32

### Curva original
x = np.arange(0, Lx, dx)
y = np.round(x*np.sin(x),1)
N = len(y)

### Salvando a curva original
pl.figure(figsize=(4,4), dpi=pix)
pl.plot(x, y, 'b.')
pl.savefig("fig01.png")
pl.axis('off')
pl.clf()

##### Criando um dataset
for i in tqdm(range(200)):
  ### Entrada -> CSV
  i1 = rm.randint(1, N-1)
  i2 = rm.randint(1, N-1)
  [i1, i2] = [i1, i2] if (i1 < i2) else [i2, i1]
  x1, y1, x2, y2 = x[i1], y[i1], x[i2], y[i2]
  matriz = np.array([[x1, y1], [x2, y2]])
  np.savetxt("data/in_{:03d}.csv".format(i+1), matriz, fmt='%1.2f', delimiter=',')
  ### Saida -> PNG
  pl.figure(figsize=(4,4), dpi=pix)
  pl.plot(x[i1:i2], y[i1:i2], 'r.')
  pl.axis('off')
  pl.savefig("fig02.png")
  pl.clf()
  ### Tranformando em matriz esparsa
  imagem = cv2.imread("fig02.png")
  cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
  cinza[cinza != 255] = 128
  cinza[cinza == 255] = 0
  ### Saida -> CSV
  np.savetxt("data/out_{:03d}.csv".format(i+1), cinza, fmt='%1.0f', delimiter=',') 

os.system('rm *.png')
### FIM
