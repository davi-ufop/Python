### Programa para resolver o oscilado de Van der Pol
### Davi Neves - Ouro Preto, Brasil - Jan., 2022

### Importando módulos
import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import odeint

### Parâmetros iniciais da simulação
### Ref.: https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
mu = 8.53     ## Fator de não linearidade
tmax = 200    ## Tempo máximo de integração
dt = 0.01     ## Passo temporal de integração
tp = np.arange(0, tmax+dt, dt)  ## Mesh temporal
lp = tp.size  ## Número de pontos integrados

### Condições iniciais - Posição e Velocidade
x0 = np.array([3.6, -13.7])

### Sistema de equações diferenciais acoplados
def sist_edos(x, tp):
  ## Entrada de dados:
  x1, x2 = x
  ## Derivadas primeiras do sistema:
  dx1_dt = x2
  dx2_dt = mu*(1 - x1*x1)*x2 - x1
  ## Retorno do sistema:
  return dx1_dt, dx2_dt

### Integrando o sistema de ODEs
x = odeint(sist_edos, x0, tp)

### Preparando os resultados para plotar as soluções
px, vx = x[:,0], x[:,1]   ## Posições e Velocidades
pmax = max(px) + abs(max(px)*0.2) ## Limites da posição
pmin = min(px) - abs(min(px)*0.2)
vmax = max(vx) + abs(max(vx)*0.2) ## Limites da velocidade
vmin = min(vx) - abs(min(vx)*0.2)

### Plotando a solução da posição
def plot_pos(k):
  t100 = 100*k
  pl.plot(tp[0:t100], px[0:t100], 'b-.')
  pl.xlim(0, tmax)
  pl.ylim(pmin, pmax)
  pl.xlabel("Tempo (s)")
  pl.ylabel("Posição (m)")
  pl.savefig("posicao/{:03d}.png".format(k), dpi=72)
  pl.cla()

### Realizando a plotagem da posição
np = lp//100  ## Número de pontos plotados
for i in range(1, np+1):
  print("Plot ", i,"/", np, "da posição.")
  plot_pos(i)


### Plotando o retrato de fases
def plot_fases(k):
  t100 = 100*k
  pl.plot(px[0:t100], vx[0:t100], 'r-.')
  pl.xlim(pmin, pmax)
  pl.ylim(vmin, vmax)
  pl.xlabel("Posição (m)")
  pl.ylabel("Velocidade (m/s)")
  pl.savefig("efases/{:03d}.png".format(k), dpi=72)
  pl.cla()

### Realizando a plotagem da posição
for i in range(1, np+1):
  print("Plot ", i,"/", np, "das fases.")
  plot_fases(i)

### FIM
