### BIBLIOTECA CENTRAL PARA AS SIMULAÇÕES
### Davi Neves - Ouro Preto, Brasil - UFOP/2022
### Módulos de Python utilizados:
import numpy as np                           ## Numérica
import matplotlib.pyplot as pl               ## Gráfica
from tqdm import tqdm                        ## Barra de progresso
from random import choice, randint, seed     ## Randômico
from scipy.optimize import fsolve   ## Não Lienar
from matplotlib.patches import Rectangle  ## Área factível

###### Função para determinar se o ponto está na área circular útil
def util(x, y, R):
  d2 = x**2 + y**2   ## Quadrado da distância do ponto
  r2 = R**2          ## Raio da área de alcance do robô
  if (d2 <= r2):     ## Condição de verificação
    return 1         ## Tá dentro
  else:
    return 0         ## Tá fora

###### Função para determinar se o ponto está na área circular útil
def factivel(x, y, R):
  d2 = x**2 + y**2   ## Quadrado da distância do ponto
  r2 = R**2          ## Raio da área de alcance do robô
  if (d2 <= r2):     ## Condição de verificação
    return x, y         ## Tá dentro
  else:
    return None, None   ## Tá fora

###### Para igualar o tamanho de duas listas -> listas de acoes
def igualistas(l1, l2):
  ### Medindo as listas
  n1 = len(l1)
  n2 = len(l2)
  dn = abs(n1 - n2)
  ### Tornando as duas listas do mesmo tamanho
  if (n1 > n2):
    for n in range(dn):
      l2.append(0)
  elif (n1 < n2):
    for n in range(dn):
      l1.append(0)
  ### Retornando o resultado
  return l1, l2

###### Determina os ângulos do pêndulo duplo com a ponta (L1+L2) situada em xp e yp
def angulos_ponto(xp, yp, L1, L2):
  ### Usando a função fsolve() no seguinte sistema não linear
  def ponto(teta):
    return [ L1*np.sin(teta[0]) + L2*np.sin(teta[1]) - xp,
             L1*np.cos(teta[0]) + L2*np.cos(teta[1]) - yp]
  ### Resolvendo o sistema não linear com fsolve
  angulos = fsolve(ponto, [0, 0])
  ### Ajustando os ângulos para ficarem no intervalo: [0, 2pi]
  angulos = np.around(angulos%(2*np.pi), 2) 
  ### Pronto, retorno um vetor com dois valores [teta1 e teta2]
  return angulos

###### Função para construir listas de ações, ou seja, ações em uma listas:
def acoes_listas(n1, n2, dn):     ### n1 refere-se ao alvo e n2 ao braço
  ### Número de termos da lista
  N = int(abs(n1-n2)//dn)
  ### Iniciando a lista
  la = [0]
  ### Montando a lista de ações
  if (n1 < n2):
    for n in range(N):
      la.append(-dn)
  elif (n1 > n2):
    for n in range(N):
      la.append(dn)
  return la

###### Função para variar linearmente os estados
def varia_estados(xk, yk, lx, ly):
  ### Listas dos estados
  l1 = []
  l2 = []
  ### Estado inicial:
  xp = xk
  yp = yk
  ### Variação até o estado final:
  for xy in zip(lx, ly):
    xp += xy[0]
    yp += xy[1]
    l1.append(xp)
    l2.append(yp)
  ### Convertendo em arrays
  vx, vy = np.array(l1), np.array(l2)
  return vx, vy

###### Plotando as trajetórias dos estados: angular e cartesiano
def plot_trajetorias(vxa, vya, vxp, vyp, caminho):
  ### Plot simples
  pl.clf()
  pl.plot(vxa, vya, 'r-.', label='Angular')
  pl.plot(vxp, vyp, 'b-.', label='Cartesiano')
  pl.title("Trajetórias dos estados")
  pl.xlabel("X")
  pl.ylabel("Y")
  pl.legend()
  pl.savefig(caminho, dpi=200)

###### Apresentando os movimentos dos braços robóticos
def move_braco(xo, yo, a1, a2, la1, la2, R, caminho, tamanho):
  ### Parâmetros:
  L1, L2 = R/2, R/2
  LM = round((R/np.sqrt(2)), 2) 
  passo = 0
  dt = 0.5
  dt2 = dt/2
  ### Retângulo que representa a áera útil
  rect = Rectangle((-LM,-LM), 2*LM, LM, linewidth=2, edgecolor='brown', facecolor='none')
  ### Realizando os movimentos sincronizadamente:
  for ac in zip(la1, la2):
    ### Atualizando o valor do ângulo 1
    a1 += ac[0]
    a2 += ac[1]
    ### Posição do semi-braço, chamado também de primeiro braço 
    x1 = L1*np.sin(a1)
    y1 = L1*np.cos(a1)
    ### Ponto onde se encontra a extremidade do braço robótico
    x2 = L1*np.sin(a1) + L2*np.sin(a2)
    y2 = L1*np.cos(a1) + L2*np.cos(a2)
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
    ang1 = 180*a1/np.pi
    ang2 = 180*a2/np.pi
    texto1 = str("Estado: {:.1f}°({:.2f}) e {:.1f}°({:.2f}) ".format(ang1, a1, ang2, a2))
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
    ### Um extra, adicionando as ações no canto direito e o estado no esquerdo
    txtaco = str("Ação:\nBraço1 = {:.2f}\nBraço2 = {:.2f}".format(ac[0], ac[1]))
    pl.text(R-0.99, 0.15, txtaco)
    txtstt = str("Ponta:\nX = {:.2f}\nY = {:.2f}".format(x2, y2))
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
    if (dt > dt2):
      dt = dt/5000
    ### Contando os passos:
    passo += 1
  pl.close()
  ### Pronto!

###### Determinando os estados
def estados(R, dd):
  ### Listas para plotagem
  lx1, lx2 = [], []
  la1, la2 = [], []
  L1, L2 = R/2, R/2   ### Parâmetros necessários
  ### Pontos que constituem os estados 
  xp = np.arange(-R, R, dd)
  yp = np.arange(-R, 0, dd)
  ### Verificando condições
  for xk in xp:
    for yk in yp:
      ### Está dentro do semi-círculo?
      xu, yu = factivel(xk, yk, R)
      if (xu != None):
        lx1.append(xu)
        lx2.append(yu)
  ### Agora a lista dos ângulos
  for xa in zip(lx1, lx2):
    [a1, a2] = angulos_ponto(xa[0], xa[1], L1, L2)
    la1.append(a1)
    la2.append(a2)
  ### Retorno:
  return lx1, lx2, la1, la2

###### Plotando os estados contínuos do sistema (pêndulo duplo)
def plot_estados(R, dd, caminho):
  ### Precisão dos dados
  P = 2
  ### Listas para plotagem
  lx1, lx2, la1, la2 = estados(R, dd) 
  ### Determinando resultados importantes 
  NE = int(len(lx1))            ### Número de estados
  TAG = np.array([la1, la2])    ### Vetor dos estados angulares
  TXY = np.array([lx1, lx2])    ### Vetor dos estados cartesianos
  TAG = np.round(TAG.T, P)
  TXY = np.round(TXY.T, P)
  ### Retângulo que representa a áera útil
  LM = round((R/np.sqrt(2)), P) 
  rect = Rectangle((-LM,-LM), 2*LM, LM, linewidth=3, edgecolor='black', facecolor='none')
  ### Plotando os estados
  pl.scatter(lx1, lx2, color='red', marker='.')
  pl.xlim(-R, R)
  pl.ylim(-R, 0.5)
  pl.title("Estados angulares factíveis")
  pl.xlabel("X")
  pl.ylabel("Y")
  pl.text(0, 0.2, "Número de estados = {}".format(NE))
  pl.gca().add_patch(rect)         ## Adicionando uma Mesa (Retângulo)
  pl.savefig(caminho, dpi=200)     ## Salvando
  pl.close()
  ### Retorno útil
  return NE, TAG, TXY

###### Função para conferir a completeza do espaço de estados
def completo(R, dd, TAB):
  P = 2   ### Precisão
  Vx = np.arange(-R, R, dd)   ### Vetores de X e Y
  Vy = np.arange(-R, 0, dd)
  bom, ruim = 0, 0   ### Contadores
  ### Loop para comparar:
  for xi in Vx:
    for yj in Vy:
      ###  Verificando se são pontos factíveis
      xk, yk = factivel(xi, yj, R)
      if (xk != None):
        ### Arredondando pra evitar erros
        xk = round(xk, P)
        yk = round(yk, P)
        ### Contando erros
        soma = (TAB == (xk, yk)).all(-1).sum()
        if (soma == 1):
          bom += 1
        else:
          ruim += 1
  ### Resultado
  return bom, ruim

###### Função para definir as ações
def acoes(TXY, estado, acao, dx):
  ### Definindo as coordenadas do estados conforme
  ### as respecitvas ações
  if (acao == 0):
    x1 = TXY[estado, 0] - dx    ### [-1, -1]      
    x2 = TXY[estado, 1] - dx       
  elif (acao == 1):
    x1 = TXY[estado, 0] - dx    ### [-1, +1]    
    x2 = TXY[estado, 1] + dx      
  elif (acao == 2):
    x1 = TXY[estado, 0] - dx    ### [-1, 0]     
    x2 = TXY[estado, 1]        
  elif (acao == 3):
    x1 = TXY[estado, 0] + dx      
    x2 = TXY[estado, 1] - dx      
  elif (acao == 4):
    x1 = TXY[estado, 0] + dx       
    x2 = TXY[estado, 1] + dx       
  elif (acao == 5):
    x1 = TXY[estado, 0] + dx       
    x2 = TXY[estado, 1]        
  elif (acao == 6):
    x1 = TXY[estado, 0]        
    x2 = TXY[estado, 1] - dx       
  elif (acao == 7):
    x1 = TXY[estado, 0]        
    x2 = TXY[estado, 1] + dx      
  elif (acao == 8):
    x1 = TXY[estado, 0]        
    x2 = TXY[estado, 1] 
  else:
    print("Ação mão existe!")
  ### Ajustando as coordenadas
  x1 = round(x1, 2)
  x2 = round(x2, 2)
  ### Determinando o próximo estado
  proximo = np.where(np.all(TXY == (x1, x2), axis=1))[0]
  if (len(proximo) > 0):
    proximo = proximo[0]
  else:
    proximo = estado
  ### Resultado
  return proximo

###### Função para estabelecer as recompensas
def recompensas(acao, xi, yi, xo, yo, B):
  ### Recompensa por achar o objeto
  if (xi == xo and yi == yo):
    if (acao == 8):       ### Se encontrou, tem que escolher
      recompensa = 10
    else:                 ### Senão será punido
      recompensa = -10
  ### Mas se não achar
  else:
    ### Pontos fora da mesa e na zona proibida
    if (abs(xi) > B or 0 < yi < -B):
      recompensa  = -1.5    ## Punição por sair da mesa
    ### Pontos sobre a amesa
    elif (abs(xi) <= B or  0 >= yi >= -B):
      if (acao == 8):
        recompensa = -5     ## Punição pra não travar num lugar só
      else:
        recompensa = 0.05   ## Recompensa por estar na mesa
  ### Retorno
  return recompensa

###### Função para realizar o treino
def treino(qtab, K, TXY, xo, yo, dx, B):
  ### Parâmetros e variáveis
  alfa = 0.55    ### Convergência do algoritmo
  gama = 0.75
  NE = len(qtab)        ### Estados:
  ves = np.arange(0, NE)
  ### Realizando o treino K vezes
  for i in tqdm(range(K)):
    ### Possibilidade de ações
    #diagonais = [0, 1, 3, 4, 8]
    quadradas = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    ### Escolhendo a k-esima acao
    #acao = choice(diagonais)
    acao = choice(quadradas)
    ### Escolhendo o estado
    estado = choice(ves)
    ### Determinando a recompensa
    xi = TXY[estado, 0]
    yi = TXY[estado, 1]
    recompensa = recompensas(acao, xi, yi, xo, yo, B)
    ### Definindo o proximo estado
    proximo = acoes(TXY, estado, acao, dx)
    ### Calculando a nova recompensa - Bellman:
    atual = qtab[estado, acao]
    maior = np.max(qtab[proximo])
    novov = (1-alfa)*atual + alfa*(recompensa + gama*maior)
    ### Atualizando a Q-table
    qtab[estado, acao] = round(novov, 3)
  ### Resultado
  return qtab

### Testando a Q-Table
def testeQ(qtab, TXY, xi, yi, xo, yo, dd, caminho, NT):
  ### Contagem e limite de operações
  KO, LO = 0, 500
  eps = 2*dd
  ### Lista para plotar a trajetória inligente
  lx, ly, le = [], [], []
  ### Definindo o estado inicial:
  ie = np.where(np.all(TXY == (xi, yi), axis=1))[0]
  if (len(ie) > 0):
    estado = int(ie[0])
  else:
    print("\nEstado não existe!")
    estado = randint(0, len(qtab))
    print("Escolhemos outro: ", estado)
  ### Salvando o estado inicial
  le.append(estado)
  ### Definindo as coordenadas, caso tenhamos escolhido outro
  xi = TXY[estado, 0]
  yi = TXY[estado, 1]
  ### Buscando o objeto com Q-Table
  while (abs(xi - xo)>eps or abs(yi - yo)>eps):
    ### Passo/Estado
    #print("Estado atual = ", estado)
    #print("x=", xi, " y=", yi)
    ### Determinando a ação atual
    acao = np.argmax(qtab[estado])
    #print("Acao = ", acao)
    ### Determinando o próximo estado
    proximo = acoes(TXY, estado, acao, dd)
    ### Atualizando os valores de xi e yi
    xi = TXY[proximo, 0]
    yi = TXY[proximo, 1]
    ### Salvando nas lista
    lx.append(xi)
    ly.append(yi)
    le.append(proximo)
    ### Atualizando o estado
    estado = proximo
    ### Controle
    KO += 1
    if (KO > LO):
      break
  ### Resultado parcial
  print("\nResultado:")
  print("Objeto alcançado em ", len(le), " passos.\nEstes:\n", le)
  ### Adicionando as coordenadas do objeto 
  lx.append(xo)
  ly.append(yo)
  ### Plotando a trajetória inteligente
  pl.plot(lx, ly, 'b-o')
  pl.scatter(xo, yo, color='red', marker='D', s=50)
  pl.title("Trajetória inteligente {}".format(NT))
  pl.xlabel("X")
  pl.ylabel("Y")
  pl.savefig(caminho, dpi=200)
  pl.close()

############################ FIM
