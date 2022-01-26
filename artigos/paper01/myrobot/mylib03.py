#### Programa para implementar funções usadas noutros códigos
#### Davi Neves - Ouro Preto, Brasil - Jan., 2022

### Módulos de Python usados
import numpy as np                  ## Numérica
import matplotlib.pyplot as pl      ## Gráfica
from scipy.optimize import fsolve   ## Não Lienar
from matplotlib.patches import Rectangle  ## Área factível
from mylib01 import *               ## Primeira biblioteca

### Para igualar as listas de ações nos ângulos
def igualistas(da1, da2):
  ### Medindo as listas
  lda1 = len(da1)
  lda2 = len(da2)
  dl = abs(lda1 - lda2)
  ### Tornando as duas listas do mesmo tamanho
  if (lda1 > lda2):
    for i in range(dl):
      da2.append(0)
  elif (lda2 > lda1):
    for i in range(dl):
      da1.append(0)
  ### Retornando o resultado
  return da1, da2

###### Apresentando os movimentos de acordo com
###### estas ações:
### Movimento do braço1:
def move_braco(xo, yo, a1b, a2b, da1, da2, R, caminho, tamanho):
  ### Parâmetros:
  L1, L2 = R/2, R/2
  LM = round(1.1*(R/np.sqrt(2)), 2) 
  passo = 0
  dt = 2
  ### Retângulo que representa a áera útil
  rect = Rectangle((-LM,-LM), 2*LM, LM, linewidth=2, edgecolor='brown', facecolor='none')
  ### Igualando as listas de ações (mesmo número de incrementos angulares)
  da1, da2 = igualistas(da1, da2)
  ### Realizando os movimentos sincronizadamente:
  for ac in zip(da1, da2):
    ### Atualizando o valor do ângulo 1
    a1b += ac[0]
    a2b += ac[1]
    ### Posição do semi-braço, chamado também de primeiro braço 
    x1 = L1*np.sin(a1b)
    y1 = L1*np.cos(a1b)
    ### Ponto onde se encontra a extremidade do braço robótico
    x2 = L1*np.sin(a1b) + L2*np.sin(a2b)
    y2 = L1*np.cos(a1b) + L2*np.cos(a2b)
    ### Diferença das coordenadas referidas
    dx = x1 - x2
    dy = y1 - y2
    ### Distância do objeto ao braço
    dob = np.sqrt((xo - x2)**2 + (yo - y2)**2)
    ### Plotando a tentativa do braço pegar o objeto
    brx, bry = [0, x1, x2], [0, y1, y2]                 ## Braços
    pl.plot(brx, bry, 'b-X', linewidth=2.5, ms=12, mfc='k', mec='k')                    
    pl.scatter(xo, yo, color='red', marker='D', s=50)   ## Objeto
    pl.xlim(-R, R)    ## Limites da figura
    pl.ylim(-R, 0.5)
    pl.gca().add_patch(rect)         ## Adicionando uma Mesa (Retângulo)
    pl.gcf().set_size_inches(10, 6)  ## Estabelecendo o tamanho da figura
    ### Textos na figura, começando pelo estado e passo da simulação
    ang1 = 180*a1b/np.pi
    ang2 = 180*a2b/np.pi
    texto1 = str("Angles: {:.1f}° and {:.1f}° ".format(ang1, ang2))
    texto2 = str("  => Step: {}".format(passo+1))
    textoT = texto1 + texto2
    pl.title(textoT)
    ### Agora conferindo o tamanho dos braços, no eixo X
    texto3 = str("|b1| = {:.1f}".format(np.sqrt(x1*x1 + y1*y1))) 
    texto4 = str(" and |b2| = {:.1f}".format(np.sqrt(dx*dx + dy*dy)))
    textoX = texto3 + texto4
    pl.xlabel(textoX)
    ### Por fim a distância objeto--braço e a posição do objeto no eixo Y
    texto5 = str("|ob| = {:.3f}  if ".format(dob))
    texto6 = str("[x, y] = [{:.2f}, {:.2f}]".format(xo, yo))
    textoY = texto5 + texto6
    pl.ylabel(textoY)
    ### Um extra, adicionando as ações no canto direito e o estado no esquerdo
    txtaco = str("Action:\nArm 1 = {:.2f}\nArm 2 = {:.2f}".format(ac[0], ac[1]))
    pl.text(R-0.99, 0.15, txtaco)
    txtstt = str("State:\nX = {:.2f}\nY = {:.2f}".format(x2, y2))
    pl.text(-R+0.4, 0.15, txtstt)
    ### Condição de parada
    if (dob < tamanho):
      ### Salve a figura
      pl.savefig(caminho, dpi=600)
      break
    ### Fazendo a figura se comportar como um GIF
    pl.show(block=False)
    pl.pause(dt)
    pl.cla()
    ### Apenas a primeira figura vai demorar 2s, as d+ 0.004s
    if (dt > 0.5):
      dt = dt/500
    ### Contando os passos:
    passo += 1
  ### Pronto!
  pl.close()

### FIM
