### Programa para resolver o sistema de Lorenz
### Davi Neves - Ouro Preto, Brasil - Jan., 2022

### Importando os modulos
import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import odeint

### Parâmetros do sistema
### Ref.: https://en.wikipedia.org/wiki/Lorenz_system
sigma, beta, rho = 10, 8/3, 28

### Parâmetros e variáveis temporais
tmax = 50
dt = 0.001
t = np.arange(0, tmax+dt, dt)
lp = t.size  ## Número total de pontos da solução

### Condições iniciais
y0 = np.array([1.9, 0.45, 2.6])

### Sistema de Lorenz
def sist_edos(y, t, S, B, R):
  ## Condição
  y1, y2, y3 = y
  ## Sistema: Taxa de convecção (1) + 
  ## Temperaturas horizontais (2) e verticais (3)
  dy1_dt = S*(y2 - y1)
  dy2_dt = y1*(R - y3) - y2
  dy3_dt = y1*y2 - B*y3
  ## Resultado
  return dy1_dt, dy2_dt, dy3_dt

### Integrando o sistema de ODEs
y = odeint(sist_edos, y0, t, args=(sigma, beta, rho))

### Parâmetros para plotagem
z1, z2, z3 = y[:,0], y[:,1], y[:,2]   ## Vetores referentes as soluções
z1max = max(z1) + abs(max(z1)*0.2)    ## Limites para a convecção
z1min = min(z1) - abs(min(z1)*0.2)
z2max = max(z2) + abs(max(z2)*0.2)    ## Limites da Temperatura horizontal
z2min = min(z2) - abs(min(z2)*0.2)
z3max = max(z3) + abs(max(z3)*0.2)    ## Limites da Temperatura vertical
z3min = min(z3) - abs(min(z3)*0.2)

### Plotando a solução (Convecção vs Temperatura horizontal)
### a cada duezentos pontos
def plot_200pts(k):
  t200 = 200*k
  pl.plot(z1[0:t200], z2[0:t200], 'r-.')
  pl.xlim(z1min, z1max)
  pl.ylim(z2min, z2max)
  pl.title("Lorenz")
  pl.xlabel("Convecção")
  pl.ylabel("Temperatura Horizontal")
  pl.savefig("frames/{:03d}.png".format(k), dpi=72)
  pl.cla()

### Gerando os frames para solução em vídeo
np = lp//200  ## Número máximo de frames=250
for i in range(1, np+1):
  print("Passo: ", i, "/", np)
  plot_200pts(i)

### FIM
