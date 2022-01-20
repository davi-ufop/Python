### Programa que utiliza o operador de Koopman para reconstruir
### as trajetórias no espaço de fases 
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Bibliotecas e módulos
import numpy as np          ### Numérica
import pylab as pl          ### Gráfica
import pykoopman as kp      ### Método de Koopman
from tqdm import tqdm       ### Barras de progresso
from imageio import imread  ### Imagens

### Diretórios
DDP = "prog07/data/dpendulo/"
DDS = "prog07/data/duffing/"
DDL = "prog07/data/lorenz/"
DIP = "prog07/imgs/dpendulo/"
DIS = "prog07/imgs/duffing/"
DIL = "prog07/imgs/lorenz/"
DKP = "prog07/koopman/dpendulo/"
DKS = "prog07/koopman/duffing/"
DKL = "prog07/koopman/lorenz/"

### Parâmetros de mylib06,py
N = 50

###### PÊNDULO DUPLO
### Importando dados
print("\nPêndulo duplo:")
for i in tqdm(range(N)):
  ### Condições iniciais
  x0 = np.genfromtxt(DDP+"entrada{:02d}.csv".format(i+1), delimiter=",")
  ### Trajetórias -> Dados: Data-Drive
  x = np.genfromtxt(DDP+"saida{:02d}.csv".format(i+1), delimiter=",")
  teta1, teta2, omega1, omega2 = x[:,0], x[:,1], x[:,2], x[:,3]
  ### Plotando o espaço de fases 1
  pl.plot(teta1, omega1, 'b.')
  pl.title(str(list(np.round(x0,2))))
  pl.xlabel('teta1')
  pl.ylabel('omega1')
  pl.savefig(DIP+"efases1{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Plotando o espaço de fases 2
  pl.plot(teta2, omega2, 'b-')
  pl.title(str(list(np.round(x0,2))))
  pl.xlabel('teta2')
  pl.ylabel('omega2')
  pl.savefig(DIP+"efases2{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ###### OPERADOR DE KOOPMAN
  ### Adequando os dados para usarmos no modelo -> Vide [1]
  Field = np.array([teta1, teta2, omega1, omega2]).T
  ### Determinando o operador (matriz) de Koopman
  modelo = kp.Koopman()             ### Modelo do método
  modelo.fit(Field)                 ### Ajustando os dados
  K = modelo.koopman_matrix.real    ### Determinando a matrix
  ### Salvando o operador
  np.savetxt(DKP+"koopman{:02d}.csv".format(i+1), K, delimiter=",")
  pl.matshow(K)
  pl.savefig(DIP+"operador{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Fazendo a reconstrução (rebuild) das trajetórias
  Frb = np.vstack((Field[0], modelo.simulate(Field[0], n_steps=Field.shape[0]-1)))
  x1 = Frb[:,0].real
  x2 = Frb[:,1].real
  x3 = Frb[:,2].real
  x4 = Frb[:,3].real
  ### Espaço de fases 1 reconstruido
  pl.plot(x1, x3, 'r.')
  pl.xlabel('teta1')
  pl.ylabel('omega1')
  pl.savefig(DIP+"koopman1{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Espaço de fases 2 reconstruido
  pl.plot(x2, x4, 'r-')
  pl.xlabel('teta2')
  pl.ylabel('omega2')
  pl.savefig(DIP+"koopman2{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  
###### Combinando imagens
num = 32
pl.figure(figsize=(8, 8), dpi=128)
imagem1 = imread(DIP+"efases1{:02d}.png".format(num))
imagem2 = imread(DIP+"efases2{:02d}.png".format(num))
imagem3 = imread(DIP+"koopman1{:02d}.png".format(num))
imagem4 = imread(DIP+"koopman2{:02d}.png".format(num))
### Espaço de fases 1
pl.subplot(2, 2, 1)
pl.axis('off')
pl.imshow(imagem1)
pl.title('(A)', y=-0.1)
### Espaço de fases 2
pl.subplot(2, 2, 2)
pl.axis('off')
pl.imshow(imagem2)
pl.title('(B)', y=-0.1)
### Espaço de fases reconstruido 1
pl.subplot(2, 2, 3)
pl.axis('off')
pl.imshow(imagem3)
pl.title('(C)', y=-0.1)
### Espaço de fases reconstruido 2
pl.subplot(2, 2, 4)
pl.axis('off')
pl.imshow(imagem4)
pl.title('(D)', y=-0.1)
### Salvando
pl.savefig("prog07/montagem_dpendulo.png")
pl.clf()

###### SISTEMA DE LORENZ
### Importando dados
print("\nLorenz:")
for i in tqdm(range(N)):
  ### Condições iniciais
  x0 = np.genfromtxt(DDL+"entrada{:02d}.csv".format(i+1), delimiter=",")
  ### Trajetórias -> Dados: Data-Drive
  x = np.genfromtxt(DDL+"saida{:02d}.csv".format(i+1), delimiter=",")
  C1, T1, T2 = x[:,0], x[:,1], x[:,2]
  ### Plotando o espaço de fases 1
  pl.plot(C1, T1, 'b.')
  pl.title(str(list(np.round(x0,2))))
  pl.xlabel('convecção')
  pl.ylabel('temperatura horizontal')
  pl.savefig(DIL+"efases1{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Plotando o espaço de fases 2
  pl.plot(C1, T2, 'b-')
  pl.title(str(list(np.round(x0,2))))
  pl.xlabel('convecção')
  pl.ylabel('temperatura vertical')
  pl.savefig(DIL+"efases2{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ###### OPERADOR DE KOOPMAN
  ### Adequando os dados para usarmos no modelo -> Vide [1]
  Field = np.array([C1, T1, T2]).T
  ### Determinando o operador (matriz) de Koopman
  modelo = kp.Koopman()             ### Modelo do método
  modelo.fit(Field)                 ### Ajustando os dados
  K = modelo.koopman_matrix.real    ### Determinando a matrix
  ### Salvando o operador
  np.savetxt(DKL+"koopman{:02d}.csv".format(i+1), K, delimiter=",")
  pl.matshow(K)
  pl.savefig(DIL+"operador{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Fazendo a reconstrução (rebuild) das trajetórias
  Frb = np.vstack((Field[0], modelo.simulate(Field[0], n_steps=Field.shape[0]-1)))
  x1 = Frb[:,0].real
  x2 = Frb[:,1].real
  x3 = Frb[:,2].real
  ### Espaço de fases 1 reconstruido
  pl.plot(x1, x2, 'r.')
  pl.xlabel('convecção')
  pl.ylabel('temperatura horizontal')
  pl.savefig(DIL+"koopman1{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Espaço de fases 2 reconstruido
  pl.plot(x1, x3, 'r-')
  pl.xlabel('convecção')
  pl.ylabel('temperatura vertical')
  pl.savefig(DIL+"koopman2{:02d}.png".format(i+1), dpi=200)
  pl.clf()

###### Combinando imagens
num = 40
pl.figure(figsize=(8, 8), dpi=128)
imagem1 = imread(DIL+"efases1{:02d}.png".format(num))
imagem2 = imread(DIL+"efases2{:02d}.png".format(num))
imagem3 = imread(DIL+"koopman1{:02d}.png".format(num))
imagem4 = imread(DIL+"koopman2{:02d}.png".format(num))
### Espaço de fases 1
pl.subplot(2, 2, 1)
pl.axis('off')
pl.imshow(imagem1)
pl.title('(A)', y=-0.1)
### Espaço de fases 2
pl.subplot(2, 2, 2)
pl.axis('off')
pl.imshow(imagem2)
pl.title('(B)', y=-0.1)
### Espaço de fases reconstruido 1
pl.subplot(2, 2, 3)
pl.axis('off')
pl.imshow(imagem3)
pl.title('(C)', y=-0.1)
### Espaço de fases reconstruido 2
pl.subplot(2, 2, 4)
pl.axis('off')
pl.imshow(imagem4)
pl.title('(D)', y=-0.1)
### Salvando
pl.savefig("prog07/montagem_lorenz.png")
pl.clf()

###### SISTEMA DE DUFFING
### Importando dados
print("\nDuffing:")
for i in tqdm(range(N)):
  ### Condições iniciais
  x0 = np.genfromtxt(DDS+"entrada{:02d}.csv".format(i+1), delimiter=",")
  ### Trajetórias -> Dados: Data-Drive
  x = np.genfromtxt(DDS+"saida{:02d}.csv".format(i+1), delimiter=",")
  teta, omega = x[:,0], x[:,1]
  ### Plotando o espaço de fases
  pl.plot(teta, omega, 'b.')
  pl.title(str(list(np.round(x0,2))))
  pl.xlabel('teta')
  pl.ylabel('omega')
  pl.savefig(DIS+"efases1{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ###### OPERADOR DE KOOPMAN
  ### Adequando os dados para usarmos no modelo -> Vide [1]
  Field = np.array([teta, omega]).T
  ### Determinando o operador (matriz) de Koopman
  modelo = kp.Koopman()             ### Modelo do método
  modelo.fit(Field)                 ### Ajustando os dados
  K = modelo.koopman_matrix.real    ### Determinando a matrix
  ### Salvando o operador
  np.savetxt(DKS+"koopman{:02d}.csv".format(i+1), K, delimiter=",")
  pl.matshow(K)
  pl.savefig(DIS+"operador{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Fazendo a reconstrução (rebuild) das trajetórias
  Frb = np.vstack((Field[0], modelo.simulate(Field[0], n_steps=Field.shape[0]-1)))
  x1 = Frb[:,0].real
  x2 = Frb[:,1].real
  ### Espaço de fases reconstruido
  pl.plot(x1, x2, 'r.')
  pl.xlabel('teta')
  pl.ylabel('omega')
  pl.savefig(DIS+"koopman1{:02d}.png".format(i+1), dpi=200)
  pl.clf()

###### Combinando imagens
num1 = 6
num2 = 29
pl.figure(figsize=(8, 8), dpi=128)
imagem1 = imread(DIS+"efases1{:02d}.png".format(num1))
imagem2 = imread(DIS+"efases1{:02d}.png".format(num2))
imagem3 = imread(DIS+"koopman1{:02d}.png".format(num1))
imagem4 = imread(DIS+"koopman1{:02d}.png".format(num2))
### Espaço de fases 1
pl.subplot(2, 2, 1)
pl.axis('off')
pl.imshow(imagem1)
pl.title('(A)', y=-0.1)
### Espaço de fases 2
pl.subplot(2, 2, 2)
pl.axis('off')
pl.imshow(imagem2)
pl.title('(B)', y=-0.1)
### Espaço de fases reconstruido 1
pl.subplot(2, 2, 3)
pl.axis('off')
pl.imshow(imagem3)
pl.title('(C)', y=-0.1)
### Espaço de fases reconstruido 2
pl.subplot(2, 2, 4)
pl.axis('off')
pl.imshow(imagem4)
pl.title('(D)', y=-0.1)
### Salvando
pl.savefig("prog07/montagem_duffing.png")
pl.clf()

### FIM 
