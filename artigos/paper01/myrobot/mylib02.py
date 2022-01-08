#### Programa para implementar funções usadas noutros códigos
#### Davi Neves - Ouro Preto, Brasil - Jan., 2022

### Módulos de Python usados
import numpy as np                  ## Numérica
import matplotlib.pyplot as pl      ## Gráfica
from scipy.optimize import fsolve   ## Não Lienar
from matplotlib.patches import Rectangle  ## Área factível
from mylib01 import *               ## Primeira biblioteca

###### Apresentando os movimentos de acordo com
###### estas ações:
### Movimento do braço1:
def move_braco(da1, da2, a1b, a2b, R, xo, yo, caminho):
  ### Parâmetros:
  tamanho = 0.04
  passo = 0
  L1, L2 = R/2, R/2
  dt = 2
  ### Retângulo que representa a áera útil
  rect = Rectangle((-1.7,-1.8), 3.4, 1.8, linewidth=2, edgecolor='brown', facecolor='none')
  for ac1 in da1:
    ### Atualizando o valor do ângulo 1
    a1b += ac1
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
    texto1 = str("Estado: {:.1f}° e {:.1f}° ".format(ang1, ang2))
    texto2 = str("  => Passo: {}".format(passo+1))
    textoT = texto1 + texto2
    pl.title(textoT)
    ### Agora conferindo o tamanho dos braços, no eixo X
    texto3 = str("|b1| = {:.1f}".format(np.sqrt(x1*x1 + y1*y1))) 
    texto4 = str(" e |b2| = {:.1f}".format(np.sqrt(dx*dx + dy*dy)))
    textoX = texto3 + texto4
    pl.xlabel(textoX)
    ### Por fim a distância objeto--braço e a posição do objeto no eixo Y
    texto5 = str("|ob| = {:.3f}  se ".format(dob))
    texto6 = str("[x, y] = [{:.2f}, {:.2f}]".format(xo, yo))
    textoY = texto5 + texto6
    pl.ylabel(textoY)
    ### Fazendo a figura se comportar como um GIF
    pl.show(block=False)
    pl.pause(dt)
    pl.cla()
    ### Apenas a primeira figura vai demorar 2s, as d+ 0.004s
    if (dt > 0.5):
      dt = dt/500
    ### Contando os passos:
    passo += 1
  ### Movimento do braço2:
  for ac2 in da2:
    ### Atualizando o valor do ângulo 1
    a2b += ac2
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
    texto1 = str("Estado: {:.1f}° e {:.1f}° ".format(ang1, ang2))
    texto2 = str("  => Passo: {}".format(passo+1))
    textoT = texto1 + texto2
    pl.title(textoT)
    ### Agora conferindo o tamanho dos braços, no eixo X
    texto3 = str("|b1| = {:.1f}".format(np.sqrt(x1*x1 + y1*y1))) 
    texto4 = str(" e |b2| = {:.1f}".format(np.sqrt(dx*dx + dy*dy)))
    textoX = texto3 + texto4
    pl.xlabel(textoX)
    ### Por fim a distância objeto--braço e a posição do objeto no eixo Y
    texto5 = str("|ob| = {:.3f}  se ".format(dob))
    texto6 = str("[x, y] = [{:.2f}, {:.2f}]".format(xo, yo))
    textoY = texto5 + texto6
    pl.ylabel(textoY)
    ### Condição de parada
    if (dob < tamanho):
      ### Salve a figura
      pl.savefig(caminho, dpi=600)
      break
    ### Fazendo a figura se comportar como um GIF
    pl.show(block=False)
    pl.pause(dt)
    pl.cla()
    ### Contando os passos:
    passo += 1
  ### Pronto!
  pl.close()

### FIM
