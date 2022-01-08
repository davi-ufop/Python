### Programa para elucidar o controle de um 
### braço robótico em modo determinístico
### Davi Neves - Ouro Preto, Brasil - Jan., 2022

### Importando módulos necessários
from mylib01 import *
from random import uniform

### Adicionando uma mesa na figura
from matplotlib.patches import Rectangle
rect = Rectangle((-1.7,-1.8), 3.4, 1.8, linewidth=2, edgecolor='brown', facecolor='none')

### Parâmetros e variáveis do processo
dteta = 0.01    ### Tamanho do passo das ações, vide around() em mylib
R = 2                ### Dimensões do problema 
L1, L2 = R/2, R/2 
FK = 0.7             ### Evita problemas
passo = 0            ### Passo da simulação
tamanho = 0.04       ### Tamanho do objeto

### Definindo a posição do objeto
xo = round(FK*uniform(-R, R), 3)
yo = round(FK*uniform(-R, 0), 3)
print("x_obj=",xo,"y_obj=",yo)

### Definindo os ângulos alvos, do objeto
[a1o, a2o] = angulos_ponto(xo, yo, 1, 1)
print("a1_alvo=",a1o,"a2_alvo=",a2o)

### Definindo a posição do braço
xb = round(uniform(-R, R), 3)
yb = round(uniform(-R, 0), 3)
print("x_robo=",xb,"y_robo=",yb)

### Definindo o estado do sistema
[a1b, a2b] = angulos_ponto(xb, yb, 1, 1)
print("a1_robo=",a1b,"a2_robo=",a2b)

### Determinando o número de ações
na1 = int( abs(a1b-a1o)//dteta )
na2 = int( abs(a2b-a2o)//dteta )
nat = na1 + na2
print("Serão realizadas ", nat, " ações, divididas assim: ", [na1, na2])

###### Construindo as ações - MUITO IMPORTANTE
### Listas para as ações nas juntas do braço
da1 = acoes_junta(a1o, a1b, na1, dteta)
da2 = acoes_junta(a2o, a2b, na2, dteta)

###### Apresentando os movimentos de acordo com
###### estas ações:
### Movimento do braço1:
dt = 2
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
  ### Apenas a primeira figura vai demorar 2s, as d+ 0.02s
  if (dt > 1):
    dt = dt/100
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
    pl.savefig("prog02/pegou.png", dpi=600)
    break
  ### Fazendo a figura se comportar como um GIF
  pl.show(block=False)
  pl.pause(dt)
  pl.cla()
  ### Contando os passos:
  passo += 1

### Pronto
pl.close()

### FIM
