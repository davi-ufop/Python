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
dt = 0.01

###### PÊNDULO DUPLO
### Importando dados
print("\nPêndulo duplo:")
for i in tqdm(range(N)):
  ###### DADOS E AJUSTES
  ### Condições iniciais
  x0 = np.genfromtxt(DDP+"entrada{:02d}.csv".format(i+1), delimiter=",")
  ### Trajetórias -> Dados: Data-Drive
  X = np.genfromtxt(DDP+"saida{:02d}.csv".format(i+1), delimiter=",")
  X1, X2, X3, X4 = X[:,0], X[:,1], X[:,2], X[:,3]
  ### Adequando os dados para usarmos no modelo -> Vide [1]
  F = np.array([X1, X2, X3, X4]).T  ### Field 
  ###### OPERADOR DE KOOPMAN
  ### Determinando o operador (matriz) de Koopman
  modelo = kp.Koopman()             ### Modelo de Koopman Simples
  modelo.fit(F)                     ### Ajustando os dados
  K = modelo.koopman_matrix         ### Determinando a matrix do operador
  ### Fazendo a reconstrução (rebuild) das trajetórias
  Xp = modelo.predict(F)                   ### Com o predict()
  Xs = modelo.simulate(F[0], n_steps=700)  ### Com o simulate()
  ### Salvando as trajetorias previstas e simuladas
  np.savetxt(DDP+"prev_{:02d}.csv".format(i+1), Xp.real, delimiter=",")
  np.savetxt(DDP+"sims_{:02d}.csv".format(i+1), Xs.real, delimiter=",")
  ### Previsão das medidas
  X1p, X2p, X3p, X4p = Xp[:,0], Xp[:,1], Xp[:,2], Xp[:,3]
  ### Correlação entre dados e previsões e dos métodos do pacote
  X1s = Xs[:,0]
  MCDP1 = np.corrcoef(X1, X1p).real        ### Correlação entre dados e previsões: X1
  MCDP2 = np.corrcoef(X2, X2p).real        ### Correlação entre dados e previsões: X2
  MCDP3 = np.corrcoef(X3, X3p).real        ### Correlação entre dados e previsões: X3
  MCDP4 = np.corrcoef(X4, X4p).real        ### Correlação entre dados e previsões: X4
  MC100 = np.corrcoef(X1p[1:101], X1s[0:100]).real   ### Correlação de 100 pontos
  MC300 = np.corrcoef(X1p[1:301], X1s[0:300]).real   ### Para 500 pontos
  MC500 = np.corrcoef(X1p[1:501], X1s[0:500]).real   ### Para 500 pontos
  MC700 = np.corrcoef(X1p[1:701], X1s).real          ### Para 900 pontos
  ###### APRESENTANDO RESULTADOS
  ### Salvando as matrizes e plotando seus valores
  np.savetxt(DKP+"koopman{:02d}.csv".format(i+1), K, delimiter=",")
  np.savetxt(DKP+"corrprev1{:02d}.csv".format(i+1), MCDP1, delimiter=",")
  np.savetxt(DKP+"corrprev2{:02d}.csv".format(i+1), MCDP2, delimiter=",")
  np.savetxt(DKP+"corrprev3{:02d}.csv".format(i+1), MCDP3, delimiter=",")
  np.savetxt(DKP+"corrprev4{:02d}.csv".format(i+1), MCDP4, delimiter=",")
  np.savetxt(DKP+"corr100{:02d}.csv".format(i+1), MC100, delimiter=",")
  np.savetxt(DKP+"corr300{:02d}.csv".format(i+1), MC300, delimiter=",")
  np.savetxt(DKP+"corr500{:02d}.csv".format(i+1), MC500, delimiter=",")
  np.savetxt(DKP+"corr700{:02d}.csv".format(i+1), MC700, delimiter=",")
  ### Koopman Eigenvetors an Eigenvalues
  autova, autove = np.linalg.eig(K)      ## Determinando
  #autova_cont = autova/dt               ## Ajustando para valores continuos [1]
  ### Representando os eigenvetors
  EK = autove.real                       ## Matriz de autovetroes
  pl.matshow(EK)
  pl.title('(Koopman eigenvectors)', y=-0.1)
  pl.savefig(DIP+"operador{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Representando os eigenvalues
  Xva = [xva.real for xva in autova]     ## Partes real e imaginária
  Yva = [yva.imag for yva in autova]     ## dos autovalores
  pl.figure(figsize=(6,5), dpi=300)
  circulo = pl.Circle((0,0), 1, color='red', ls="--", lw=1.5, fill=False)
  pl.scatter(Xva, Yva, color='blue', s=60)
  pl.gca().add_patch(circulo)
  pl.xlim(-1.2, 1.2)
  pl.ylim(-1.3, 1.3)
  pl.xlabel("real")
  pl.ylabel("imaginary")
  pl.savefig(DIP+"valores{:02d}.png".format(i+1))
  pl.clf()
  ### Corelação Dados e Previsões - X1
  rotulos = ['Data', 'Forecast']
  pl.matshow(MCDP1, cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Corr-Teta 1)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig(DIP+"corrprev1{:02d}.png".format(i+1), dpi=400)
  pl.clf()
  ### Corelação Dados e Previsões - X2
  pl.matshow(MCDP2, cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Corr-Teta 2)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig(DIP+"corrprev2{:02d}.png".format(i+1), dpi=400)
  pl.clf()
  ### Corelação Dados e Previsões - X3
  pl.matshow(MCDP3, cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Corr-Omega 1)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig(DIP+"corrprev3{:02d}.png".format(i+1), dpi=400)
  pl.clf()
  ### Corelação Dados e Previsões - X4
  pl.matshow(MCDP4, cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Corr-Omega 2)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig(DIP+"corrprev4{:02d}.png".format(i+1), dpi=400)
  pl.clf()
  ### Corelação 100
  pl.matshow(MC100, cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Corr100)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig(DIP+"corr100{:02d}.png".format(i+1), dpi=400)
  pl.clf()
  ### Corelação 300
  pl.matshow(MC300, cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Corr300)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig(DIP+"corr300{:02d}.png".format(i+1), dpi=400)
  pl.clf()
  ### Corelação 500
  pl.matshow(MC500, cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Corr500)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig(DIP+"corr500{:02d}.png".format(i+1), dpi=400)
  pl.clf()
  ### Corelação 700
  pl.matshow(MC700, cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Corr700)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig(DIP+"corr700{:02d}.png".format(i+1), dpi=400)
  pl.clf()
  ### Plotando o espaço de fases 1
  pl.plot(X1, X3, 'b.')
  pl.title(str(list(np.round(x0,2))))
  pl.xlabel('teta-1')
  pl.ylabel('omega-1')
  pl.savefig(DIP+"efases1{:02d}.png".format(i+1), dpi=400)
  pl.clf()
  ### Plotando o espaço de fases 2
  pl.plot(X2, X4, 'b-')
  pl.title(str(list(np.round(x0,2))))
  pl.xlabel('teta-2')
  pl.ylabel('omega-2')
  pl.savefig(DIP+"efases2{:02d}.png".format(i+1), dpi=400)
  pl.clf()
  ### Espaço de fases 1 reconstruido
  pl.plot(X1p.real, X3p.real, 'r.')
  pl.title(str(list(np.round(x0,2))))
  pl.xlabel('teta-1')
  pl.ylabel('omega-1')
  pl.savefig(DIP+"koopman1{:02d}.png".format(i+1), dpi=400)
  pl.clf()
  ### Espaço de fases 2 reconstruido
  pl.plot(X2p.real, X4p.real, 'r-')
  pl.title(str(list(np.round(x0,2))))
  pl.xlabel('teta-2')
  pl.ylabel('omega-2')
  pl.savefig(DIP+"koopman2{:02d}.png".format(i+1), dpi=400)
  pl.clf()

###### Combinando imagens - trajetórias
num = 32
pl.figure(figsize=(8, 8), dpi=600)
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

###### Combinando matrizes de correlação
pl.figure(figsize=(12, 4), dpi=800)
imagem1 = imread(DIP+"corrprev1{:02d}.png".format(num))
imagem2 = imread(DIP+"corrprev2{:02d}.png".format(num))
imagem3 = imread(DIP+"corrprev3{:02d}.png".format(num))
imagem4 = imread(DIP+"corrprev4{:02d}.png".format(num))
imagem5 = imread(DIP+"corr100{:02d}.png".format(num))
imagem6 = imread(DIP+"corr300{:02d}.png".format(num))
imagem7 = imread(DIP+"corr500{:02d}.png".format(num))
imagem8 = imread(DIP+"corr700{:02d}.png".format(num))
### Matriz Corr-X1
pl.subplot(2, 4, 1)
pl.axis('off')
pl.imshow(imagem1)
pl.title('(A)', y=-0.1)
### Matriz Corr-X2
pl.subplot(2, 4, 2)
pl.axis('off')
pl.imshow(imagem2)
pl.title('(B)', y=-0.1)
### Matriz Cor-X3
pl.subplot(2, 4, 3)
pl.axis('off')
pl.imshow(imagem3)
pl.title('(C)', y=-0.1)
### Matriz Cor-X4
pl.subplot(2, 4, 4)
pl.axis('off')
pl.imshow(imagem4)
pl.title('(D)', y=-0.1)
### Matriz Corr100
pl.subplot(2, 4, 5)
pl.axis('off')
pl.imshow(imagem5)
pl.title('(E)', y=-0.1)
### Matriz Corr300
pl.subplot(2, 4, 6)
pl.axis('off')
pl.imshow(imagem6)
pl.title('(F)', y=-0.1)
### Matriz Corr500
pl.subplot(2, 4, 7)
pl.axis('off')
pl.imshow(imagem7)
pl.title('(G)', y=-0.1)
### Matriz Corr700
pl.subplot(2, 4, 8)
pl.axis('off')
pl.imshow(imagem8)
pl.title('(H)', y=-0.1)
### Salvando
pl.savefig("prog07/matrizes_dpendulo.png")
pl.clf()

###### SISTEMA DE LORENZ
### Importando dados
print("\nLorenz:")
for i in tqdm(range(N)):
  ###### DADOS E AJUSTES
  ### Condições iniciais
  x0 = np.genfromtxt(DDL+"entrada{:02d}.csv".format(i+1), delimiter=",")
  ### Trajetórias -> Dados: Data-Drive
  X = np.genfromtxt(DDL+"saida{:02d}.csv".format(i+1), delimiter=",")
  X1, X2, X3 = X[:,0], X[:,1], X[:,2]
  ### Adequando os dados para usarmos no modelo -> Vide [1]
  F = np.array([X1, X2, X3]).T  ### Field 
  ###### OPERADOR DE KOOPMAN
  ### Determinando o operador (matriz) de Koopman
  modelo = kp.Koopman()             ### Modelo de Koopman Simples
  modelo.fit(F)                     ### Ajustando os dados
  K = modelo.koopman_matrix         ### Determinando a matrix do operador
  ### Fazendo a reconstrução (rebuild) das trajetórias
  Xp = modelo.predict(F)                   ### Com o predict()
  Xs = modelo.simulate(F[0], n_steps=900)  ### Com o simulate()
  ### Salvando as trajetorias previstas e simuladas
  np.savetxt(DDL+"prev_{:02d}.csv".format(i+1), Xp, delimiter=",")
  np.savetxt(DDL+"sims_{:02d}.csv".format(i+1), Xs, delimiter=",")
  ### Previsão das medidas
  X1p, X2p, X3p = Xp[:,0], Xp[:,1], Xp[:,2]
  ### Correlação entre dados e previsões e dos métodos do pacote
  X1s = Xs[:,0]
  MCDP1 = np.corrcoef(X1, X1p).real        ### Correlação entre dados e previsões: X1
  MCDP2 = np.corrcoef(X2, X2p).real        ### Correlação entre dados e previsões: X2
  MCDP3 = np.corrcoef(X3, X3p).real        ### Correlação entre dados e previsões: X3
  MC100 = np.corrcoef(X1p[1:101], X1s[0:100]).real   ### Correlação de 100 pontos
  MC500 = np.corrcoef(X1p[1:501], X1s[0:500]).real   ### Para 500 pontos
  MC900 = np.corrcoef(X1p[1:901], X1s).real          ### Para 900 pontos
  ###### APRESENTANDO RESULTADOS
  ### Salvando as matrizes e plotando seus valores
  np.savetxt(DKL+"koopman{:02d}.csv".format(i+1), K, delimiter=",")
  np.savetxt(DKL+"corrprev1{:02d}.csv".format(i+1), MCDP1, delimiter=",")
  np.savetxt(DKL+"corrprev2{:02d}.csv".format(i+1), MCDP2, delimiter=",")
  np.savetxt(DKL+"corrprev3{:02d}.csv".format(i+1), MCDP3, delimiter=",")
  np.savetxt(DKL+"corr100{:02d}.csv".format(i+1), MC100, delimiter=",")
  np.savetxt(DKL+"corr500{:02d}.csv".format(i+1), MC500, delimiter=",")
  np.savetxt(DKL+"corr900{:02d}.csv".format(i+1), MC900, delimiter=",")
  ### Koopman Eigenvetors an Eigenvalues
  autova, autove = np.linalg.eig(K)      ## Determinando
  #autova_cont = autova/dt               ## Ajustando para valores continuos [1]
  ### Representando os eigenvetors
  EK = autove.real                       ## Matriz de autovetroes
  pl.matshow(EK)
  pl.title('(Koopman eigenvectors)', y=-0.1)
  pl.savefig(DIL+"operador{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Representando os eigenvalues
  Xva = [xva.real for xva in autova]     ## Partes real e imaginária
  Yva = [yva.imag for yva in autova]     ## dos autovalores
  pl.figure(figsize=(6,5), dpi=300)
  circulo = pl.Circle((0,0), 1, color='red', ls="--", lw=1.5, fill=False)
  pl.scatter(Xva, Yva, color='blue', s=60)
  pl.gca().add_patch(circulo)
  pl.xlim(-1.2, 1.2)
  pl.ylim(-1.3, 1.3)
  pl.xlabel("real")
  pl.ylabel("imaginary")
  pl.savefig(DIL+"valores{:02d}.png".format(i+1))
  pl.clf()
  ### Corelação Dados e Previsões - X1
  rotulos = ['Data', 'Forecast']
  pl.matshow(MCDP1, cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Corr-X1)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig(DIL+"corrprev1{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Corelação Dados e Previsões - X2
  pl.matshow(MCDP2, cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Corr-X2)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig(DIL+"corrprev2{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Corelação Dados e Previsões - X3
  pl.matshow(MCDP3, cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Corr-X3)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig(DIL+"corrprev3{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Corelação 100
  pl.matshow(MC100, cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Corr100)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig(DIL+"corr100{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Corelação 500
  pl.matshow(MC500, cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Corr500)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig(DIL+"corr500{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Corelação 900
  pl.matshow(MC900, cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Corr900)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig(DIL+"corr900{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Plotando o espaço de fases 1
  pl.plot(X1, X2, 'b.')
  pl.title(str(list(np.round(x0,2))))
  pl.xlabel('convecção')
  pl.ylabel('temperatura horizontal')
  pl.savefig(DIL+"efases1{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Plotando o espaço de fases 2
  pl.plot(X1, X3, 'b-')
  pl.title(str(list(np.round(x0,2))))
  pl.xlabel('convecção')
  pl.ylabel('temperatura vertical')
  pl.savefig(DIL+"efases2{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Espaço de fases 1 reconstruido
  pl.plot(X1p.real, X2p.real, 'r.')
  pl.title(str(list(np.round(x0,2))))
  pl.xlabel('convecção')
  pl.ylabel('temperatura horizontal')
  pl.savefig(DIL+"koopman1{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Espaço de fases 2 reconstruido
  pl.plot(X1p.real, X3p.real, 'r-')
  pl.title(str(list(np.round(x0,2))))
  pl.xlabel('convecção')
  pl.ylabel('temperatura vertical')
  pl.savefig(DIL+"koopman2{:02d}.png".format(i+1), dpi=200)
  pl.clf()

###### Combinando imagens - trajetórias
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

###### Combinando matrizes de correlação
pl.figure(figsize=(12, 4), dpi=128)
imagem1 = imread(DIL+"corrprev1{:02d}.png".format(num))
imagem2 = imread(DIL+"corrprev2{:02d}.png".format(num))
imagem3 = imread(DIL+"corrprev3{:02d}.png".format(num))
imagem4 = imread(DIL+"corr100{:02d}.png".format(num))
imagem5 = imread(DIL+"corr500{:02d}.png".format(num))
imagem6 = imread(DIL+"corr900{:02d}.png".format(num))
### Matriz Corr-X1
pl.subplot(2, 3, 1)
pl.axis('off')
pl.imshow(imagem1)
pl.title('(A)', y=-0.1)
### Matriz Corr-X2
pl.subplot(2, 3, 2)
pl.axis('off')
pl.imshow(imagem2)
pl.title('(B)', y=-0.1)
### Matriz Cor-X3
pl.subplot(2, 3, 3)
pl.axis('off')
pl.imshow(imagem3)
pl.title('(C)', y=-0.1)
### Matriz Corr100
pl.subplot(2, 3, 4)
pl.axis('off')
pl.imshow(imagem4)
pl.title('(D)', y=-0.1)
### Matriz Corr500
pl.subplot(2, 3, 5)
pl.axis('off')
pl.imshow(imagem5)
pl.title('(E)', y=-0.1)
### Matriz Corr900
pl.subplot(2, 3, 6)
pl.axis('off')
pl.imshow(imagem6)
pl.title('(F)', y=-0.1)
### Salvando
pl.savefig("prog07/matrizes_lorenz.png")
pl.clf()

###### SISTEMA DE DUFFING
### Importando dados
print("\nDuffing:")
for i in tqdm(range(N)):
  ###### DADOS E AJUSTES
  ### Condições iniciais
  x0 = np.genfromtxt(DDS+"entrada{:02d}.csv".format(i+1), delimiter=",")
  ### Trajetórias -> Dados: Data-Drive
  X = np.genfromtxt(DDS+"saida{:02d}.csv".format(i+1), delimiter=",")
  X1, X2 = X[:,0], X[:,1]
  ### Adequando os dados para usarmos no modelo -> Vide [1]
  F = np.array([X1, X2]).T   ### Field 
  ###### OPERADOR DE KOOPMAN
  ### Determinando o operador (matriz) de Koopman
  modelo = kp.Koopman()             ### Modelo de Koopman Simples
  modelo.fit(F)                     ### Ajustando os dados
  K = modelo.koopman_matrix         ### Determinando a matrix do operador
  ### Fazendo a reconstrução (rebuild) das trajetórias
  Xp = modelo.predict(F)                   ### Com o predict()
  Xs = modelo.simulate(F[0], n_steps=900)  ### Com o simulate()
  ### Salvando as trajetorias previstas e simuladas
  np.savetxt(DDS+"prev_{:02d}.csv".format(i+1), Xp, delimiter=",")
  np.savetxt(DDS+"sims_{:02d}.csv".format(i+1), Xs, delimiter=",")
  ### Previsão das medidas
  X1p, X2p = Xp[:,0], Xp[:,1]
  ### Correlação entre dados e previsões e dos métodos do pacote
  X1s = Xs[:,0]
  MCDP1 = np.corrcoef(X1, X1p).real        ### Correlação entre dados e previsões: X1
  MCDP2 = np.corrcoef(X2, X2p).real        ### Correlação entre dados e previsões: X2
  MC100 = np.corrcoef(X1p[1:101], X1s[0:100]).real   ### Correlação de 100 pontos
  MC900 = np.corrcoef(X1p[1:901], X1s).real          ### Para 900 pontos
  ###### APRESENTANDO RESULTADOS
  ### Salvando as matrizes e plotando seus valores
  np.savetxt(DKS+"koopman{:02d}.csv".format(i+1), K, delimiter=",")
  np.savetxt(DKS+"corrprev1{:02d}.csv".format(i+1), MCDP1, delimiter=",")
  np.savetxt(DKS+"corrprev2{:02d}.csv".format(i+1), MCDP2, delimiter=",")
  np.savetxt(DKS+"corr100{:02d}.csv".format(i+1), MC100, delimiter=",")
  np.savetxt(DKS+"corr900{:02d}.csv".format(i+1), MC900, delimiter=",")
  ### Koopman Eigenvetors an Eigenvalues
  autova, autove = np.linalg.eig(K)      ## Determinando
  #autova_cont = autova/dt               ## Ajustando para valores continuos [1]
  ### Representando os eigenvetors
  EK = autove.real                       ## Matriz de autovetroes
  pl.matshow(EK)
  pl.title('(Koopman eigenvectors)', y=-0.1)
  pl.savefig(DIS+"operador{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Representando os eigenvalues
  Xva = [xva.real for xva in autova]     ## Partes real e imaginária
  Yva = [yva.imag for yva in autova]     ## dos autovalores
  pl.figure(figsize=(6,5), dpi=300)
  circulo = pl.Circle((0,0), 1, color='red', ls="--", lw=1.5, fill=False)
  pl.scatter(Xva, Yva, color='blue', s=60)
  pl.gca().add_patch(circulo)
  pl.xlim(-1.2, 1.2)
  pl.ylim(-1.3, 1.3)
  pl.xlabel("real")
  pl.ylabel("imaginary")
  pl.savefig(DIS+"valores{:02d}.png".format(i+1))
  pl.clf()
  ### Corelação Dados e Previsões - X1
  rotulos = ['Data', 'Forecast']
  pl.matshow(MCDP1, cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Corr-Teta)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig(DIS+"corrprev1{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Corelação Dados e Previsões - X2
  pl.matshow(MCDP2, cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Corr-Omega)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig(DIS+"corrprev2{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Corelação 100
  pl.matshow(MC100, cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Corr100)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig(DIS+"corr100{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Corelação 900
  pl.matshow(MC900, cmap=pl.cm.BrBG, vmin=-1., vmax=1.)
  pl.colorbar()
  pl.title('(Corr900)', y=-0.1)
  pl.gca().set_xticklabels(['']+rotulos)
  pl.gca().set_yticklabels(['']+rotulos)
  pl.savefig(DIS+"corr900{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Plotando o espaço de fases 
  pl.plot(X1, X2, 'b.')
  pl.title(str(list(np.round(x0,2))))
  pl.xlabel('Teta')
  pl.ylabel('Omega')
  pl.savefig(DIS+"efases1{:02d}.png".format(i+1), dpi=200)
  pl.clf()
  ### Espaço de fases reconstruido
  pl.plot(X1p.real, X2p.real, 'r.')
  pl.title(str(list(np.round(x0,2))))
  pl.xlabel('Teta')
  pl.ylabel('Omega')
  pl.savefig(DIS+"koopman1{:02d}.png".format(i+1), dpi=200)
  pl.clf()

###### Combinando imagens - trajetórias
num1 = 17 
num2 = 18
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

###### Combinando matrizes de correlação
pl.figure(figsize=(8, 8), dpi=128)
imagem1 = imread(DIS+"corrprev1{:02d}.png".format(num))
imagem2 = imread(DIS+"corrprev2{:02d}.png".format(num))
imagem3 = imread(DIS+"corr100{:02d}.png".format(num))
imagem4 = imread(DIS+"corr900{:02d}.png".format(num))
### Matriz Corr-X1
pl.subplot(2, 2, 1)
pl.axis('off')
pl.imshow(imagem1)
pl.title('(A)', y=-0.1)
### Matriz Corr-X2
pl.subplot(2, 2, 2)
pl.axis('off')
pl.imshow(imagem2)
pl.title('(B)', y=-0.1)
### Matriz Corr100
pl.subplot(2, 2, 3)
pl.axis('off')
pl.imshow(imagem3)
pl.title('(C)', y=-0.1)
### Matriz Corr900
pl.subplot(2, 2, 4)
pl.axis('off')
pl.imshow(imagem4)
pl.title('(D)', y=-0.1)
### Salvando
pl.savefig("prog07/matrizes_duffing.png")
pl.clf()

### FIM 
### [1] https://pykoopman.readthedocs.io/en/latest/
