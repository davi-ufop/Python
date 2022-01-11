#### Programa para implementar funções usadas noutros códigos
#### Davi Neves - Ouro Preto, Brasil - Jan., 2022

### Módulos de Python usados
import numpy as np                  ## Numérica
import matplotlib.pyplot as pl      ## Gráfica
from scipy.optimize import fsolve   ## Não Lienar

### Função para determinar se o ponto está na áera útil
def util(x, y, R):
  d2 = x**2 + y**2   ## Quadrado da distância do ponto
  r2 = R**2          ## Raio da área de alcance do robô
  if (d2 <= r2):     ## Condição de verificação
    return 1         ## Tá dentro
  else:
    return 0         ## Tá fora

### Determinando os ângulos aproximados do braço robótico 
### para alcançar um ponto definido pelas coordenadas: xp2 e yp2
def angulos_ponto(xp2, yp2, L1, L2):
  ### Função para usar em fsolve(), contendo o sistema não linear
  def ponto(teta):
    return [ L1*np.sin(teta[0]) + L2*np.sin(teta[1]) - xp2,
             L1*np.cos(teta[0]) + L2*np.cos(teta[1]) - yp2]
  ### Resolvendo o sistema não linear com fsolve
  angulos = fsolve(ponto, [0, 0])
  ### Ajustando os ângulos para ficarem no intervalo: [0, 2pi]
  angulos = np.around(angulos%(2*np.pi), 2) 
  ### Pronto, retorno um vetor com dois valores [teta1 e teta2]
  return angulos

### Função para construir listas de ações, ou seja, ações em uma listas:
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

### FIM
