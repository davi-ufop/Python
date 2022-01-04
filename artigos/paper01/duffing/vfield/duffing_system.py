### Programa para resolver o sistema do oscilador de Duffing
### Davi Neves - Ouro Preto, Brasil - Jan., 2022

### Importando os módulos necessários
import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import odeint
from plot_vetor import *  ## plot_campo

### Parâmetros iniciais da simulação
tmax = 200          ## Tempo máximo simulado
dt = 0.01           ## Passo temporal
tp = np.arange(0, tmax+dt, dt) ## Mesh de integração
lp = tp.size        ## Número de pontos integrados
K = 0.2             ## Ampliação das dimensões da figura

### Condições iniciais
x0 = np.array([1.4, 0.3])

### Definindo o sistema de equações
def sist_edos(x, tp):
  ## Entrada das variáveis
  x1, x2 = x
  ## Equações diferenciais acopladas -> Pêndulo Simples O(3)
  dx1_dt = x2
  dx2_dt = x1 - x1**3
  ## Retorno do sistema
  return dx1_dt, dx2_dt

### Integrando o sistema de EDOs
x = odeint(sist_edos, x0, tp)

### Parâmetros para plotar
px, vx = x[:,0], x[:,1]   ## Posição e Velocidade
pmax = max(px) + (K*abs(max(px)))   ## Limites para a figura
pmin = min(px) - (K*abs(min(px)))
vmax = max(vx) + (K*abs(max(vx)))
vmin = min(vx) - (K*abs(min(vx)))

### Campo Vetorial
duffing = ["Y", "X - X**3"]

### Plotando o retrato de fases
pl.figure()
plot_campo(duffing, xran=[pmin, pmax], yran=[vmin, vmax])
pl.plot(px[0:lp], vx[0:lp], 'r-.')
pl.title("Oscilador de Duffing - Pêndulo")
pl.xlabel("Posição (m)")
pl.ylabel("Velocidade (m/s)")
pl.savefig("retrato_fases.png")

### FIM
