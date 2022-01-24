### Auxiliar de prog11, Davi Neves, Jan. 2022
import numpy as np
import pylab as pl

###### Transformando imagens em trajetórias
def fig_to_curve1(figXY, caminho):
  X, Y = [], []     ## Listas pra dados
  for i in range(len(figXY)):      # Linhas estão em Y e neste caso Y<0, vide problema do pêndulo
    for j in range(len(figXY)):    # Colunas estão no eixo X, que neste caso [-2, +2]
      if (figXY[i,j] == 128):      # Condição para os Pontos da curva
        X.append(j/64)             # Registrando com Correções adequadas
        Y.append(-i/64)
  ### Vetorização
  X = np.array(X)
  Y = np.array(Y)
  ### Ajuste de ordem
  XY = np.vstack((X, Y)).T
  XY = XY[np.argsort(XY[:,0])]
  ### Reconstrução
  X = XY[:,0]
  Y = XY[:,1]
  ### Conferindo
  pl.figure(figsize=(10,4), dpi=70)
  pl.subplot(1, 2, 1)
  pl.imshow(figXY)
  pl.axis('off')
  pl.subplot(1, 2, 2)
  pl.plot(X, Y, 'r-')
  pl.xlabel('q1')
  pl.ylabel('q2')
  pl.savefig(caminho)
  pl.clf()
  ### Retorno
  return Y

###### Transformando imagens em trajetórias
def fig_to_curve2(figXY, caminho):
  X, Y = [], []     ## Listas pra dados
  for i in range(len(figXY)):      
    for j in np.arange(1, len(figXY), 2):     # Condição pertinente
      if (figXY[i,j] != 0.0):      
        X.append(j/64)             
        Y.append(-i/64)
  ### Vetorização
  X = np.array(X)
  Y = np.array(Y)
  ### Ajuste de ordem
  XY = np.vstack((X, Y)).T
  XY = XY[np.argsort(XY[:,0])]
  ### Reconstrução
  X = XY[:,0]
  Y = XY[:,1]
  ### Conferindo
  pl.figure(figsize=(10,4), dpi=70)
  pl.subplot(1, 2, 1)
  pl.imshow(figXY)
  pl.axis('off')
  pl.subplot(1, 2, 2)
  pl.plot(X, Y, 'b.')
  pl.xlabel('q1')
  pl.ylabel('q2')
  pl.savefig(caminho)
  pl.clf()
  ### Retorno
  return Y

### FIM
