#### Programa 01 para o artigo
#### Davi Neves - Jan., 2022

### Módulos de Python usados
import numpy as np                  ## Numérica
import matplotlib.pyplot as pl      ## Gráfica
from scipy.optimize import fsolve   ## Não Lienar
from random import uniform          ## Randômica

### Adicionando uma mesa na figura
from matplotlib.patches import Rectangle
rect = Rectangle((-1.7,-1.8), 3.4, 1.8, linewidth=2, edgecolor='brown', facecolor='none')

### Parâmetros e variáveis
N = 10000                       ## Número de estados
R, KF = 2, 0.4                  ## Valores para geração de estados e objetos
L1, L2 = R/2, R/2               ## Comprimentos dos braços
pontos_x = np.array([])         ## Arrays para armazenar as coordenadas
pontos_y = np.array([])         ## dos estados
tamanho = 0.04                  ## Tamanho do objeto (m)
num_obj = 0                     ## Contagem de objetos alcançados 

### Função para determinar se o ponto está na áera útil
def util(x, y):
  d2 = x**2 + y**2   ## Quadrado da distância do ponto
  r2 = R**2          ## Raio da área de alcance do robô
  if (d2 <= r2):     ## Condição de verificação
    return 1         ## Tá dentro
  else:
    return 0         ## Tá fora

### Determinando os ângulos aproximados do braço robótico 
### para alcançar um ponto definido pelas coordenadas: xp2 e yp2
def angulos_ponto(xp2, yp2):
  ### Função para usar em fsolve(), contendo o sistema não linear
  def ponto(teta):
    return [ L1*np.sin(teta[0]) + L2*np.sin(teta[1]) - xp2,
             L1*np.cos(teta[0]) + L2*np.cos(teta[1]) - yp2]
  ### Resolvendo o sistema não linear com fsolve
  angulos = fsolve(ponto, [0, 0])
  ### Ajustando os ângulos para ficarem no intervalo: [0, 2pi]
  angulos = angulos%(2*np.pi)
  ### Pronto, retorno um vetor com dois valores [teta1 e teta2]
  return angulos

### Gerando o espaço de estados para o braço robótico
for i in range(N):
  kx = uniform(-R, R)       ## Posições x e y
  ky = uniform(-R, 0)
  ku = util(kx, ky)         ## Verificando a possibilidade 
  if (ku == 1):             ## deste ponto
    pontos_x = np.append(pontos_x, [kx])   ## Adicionando
    pontos_y = np.append(pontos_y, [ky])
  else:     ## Recuperando a contagem, queremos N estados!
    N = N-1

### Entre com as coordenandas do primeiro ponto
px = eval(input("Digite a coordenada x([-1,1]) do objeto: "))
py = eval(input("Digite a coordenada y([-1,0]) do objeto: "))

### Simulação randômica para que o braço pegue três objetos
for i in range(N):
  ### Ponto onde se encontra a extremidade do braço robótico
  x2 = pontos_x[i]
  y2 = pontos_y[i]
  ### Ângulos que definem o estado atual do braço robótico
  tetas = angulos_ponto(pontos_x[i], pontos_y[i])
  ### Posição do semi-braço, chamado também de segunda junta 
  x1 = L1*np.sin(tetas[0])
  y1 = L1*np.cos(tetas[0])
  ### Diferença das coordenadas referidas
  dx = x1 - x2
  dy = y1 - y2
  ### Distância do objeto ao braço
  dob = np.sqrt((px - x2)**2 + (py - y2)**2)
  ### Plotando a tentativa do braço pegar o objeto
  brx, bry = [0, x1, x2], [0, y1, y2]                 ## Braços
  pl.plot(brx, bry, 'b-X', linewidth=2.5, ms=12, mfc='k', mec='k')                    
  pl.scatter(px, py, color='red', marker='D', s=50)   ## Objeto
  pl.xlim(-R, R)    ## Limites da figura
  pl.ylim(-R, 0.5)
  pl.gca().add_patch(rect)         ## Adicionando uma Mesa (Retângulo)
  pl.gcf().set_size_inches(10, 6)  ## Estabelecendo o tamanho da figura
  ### Textos na figura, começando pelo estado e passo da simulação
  ang1 = 180*tetas[0]/np.pi
  ang2 = 180*tetas[1]/np.pi
  texto1 = str("Estado: {:.1f}° e {:.1f}° ".format(ang1, ang2))
  texto2 = str("  => Passo: {}".format(i+1))
  textoT = texto1 + texto2
  pl.title(textoT)
  ### Agora conferindo o tamanho dos braços, no eixo X
  texto3 = str("|b1| = {:.1f}".format(np.sqrt(x1*x1 + y1*y1))) 
  texto4 = str(" e |b2| = {:.1f}".format(np.sqrt(dx*dx + dy*dy)))
  textoX = texto3 + texto4
  pl.xlabel(textoX)
  ### Por fim a distância objeto--braço e a posição do objeto no eixo Y
  texto5 = str("|ob| = {:.3f}  se ".format(dob))
  texto6 = str("[x, y] = [{:.2f}, {:.2f}]".format(px, py))
  textoY = texto5 + texto6
  pl.ylabel(textoY)
  ### Condição de parada
  if (dob < tamanho):
    ### Salve a figura
    pl.savefig("prog01/pegou{:01d}.png".format(num_obj+1), dpi=600)
    ### Novo objeto:
    px = KF*uniform(-R, R)
    py = KF*uniform(-R, 0)
    ### Conte o novo objeto
    num_obj += 1
    ### Quando pegar três objetos, pare!
    if (num_obj > 2):
      break
  ### Fazendo a figura se comportar como um GIF
  pl.show(block=False)
  pl.pause(0.001)
  pl.cla()

### FIM
