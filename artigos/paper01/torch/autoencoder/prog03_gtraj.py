### Programa para resolver o sistema dinâmico do pêndulo duplo
### Davi Neves - Ouro Preto, Brasil - Jan., 2022

### Importnado módulos necessários
import matplotlib.pyplot as pl
from numpy import arange, array, sin, cos, pi, savetxt
from scipy.integrate import odeint
from random import uniform

### Parâmetros e variáveis da simulação
tmax = 30           ## Tempo da simulaçao (s)
dt = 0.001          ## Passo temporal de integração
t = arange(0, tmax+dt, dt)  ## Mesh temporal
lp = t.size         ## Comprimento total de pontos da solução
g = 9.82            ## Aceleração gravitacional
l1, l2 = 1, 1       ## Comprmento do pêndulo 1
m1, m2 = 1, 1       ## Massa do pêndulo 1
FP = 3000           ## Número de pontos plotados por frame

### Condições Iniciais - ALTERE ISSO!
ang1 = uniform(-pi/3, pi/3)
ang2 = uniform(-pi, pi)
x0 = array([ang1, 0, ang2, 0])

### Definindo o sistema de equações
### Ref.: https://en.wikipedia.org/wiki/Double_pendulum
def sistema(x, t):
  ### Entrada de dados
  q1, q2, p1, p2 = x
  ### Sistema de EDOs acopladas - Vide Referência
  dq1_dt = (6/(m1*l1*l1))*((2*p1 - 3*cos(q1-q2)*p2) / (16 - 9*((cos(p1-p2))**2)))
  dq2_dt = (6/(m2*l2*l2))*((8*p2 - 3*cos(q1-q2)*p1) / (16 - 9*((cos(p1-p2))**2)))
  dp1_dt = (-0.5*m1*l1*l1)*(dq1_dt*dq2_dt*sin(q1-q2) + (3*(g/l1))*sin(q1))
  dp2_dt = (-0.5*m2*l2*l2)*(-dq1_dt*dq2_dt*sin(q1-q2) + (g/l2)*sin(q2))
  ### Retorno do sistema
  return dq1_dt, dq2_dt, dp1_dt, dp2_dt

### Integrando o sistema de EDOs
x = odeint(sistema, x0, t)

### Parâmetros para plotagem
teta1, teta2, omega1, omega2 = x[:,0], x[:,1], x[:,2], x[:,3]
xk = l1*sin(teta1) + l2*sin(teta2)            ## Posições cartesianas da ponta do pêndulo
yk = l1*cos(teta1) + l2*cos(teta2)
xmax = max(xk) + (0.2*abs(max(xk)))           ## Limites para X
xmin = min(xk) - (0.2*abs(min(xk)))

### Função para plotar a solução temporal -> Trajetórias
def sol_FPk(k):
  tp = FP*k    ## Número de pontos plotados
  pl.figure(figsize=(4, 2), dpi=32)
  pl.plot(xk[0:tp], -abs(yk[0:tp]), 'k-')
  pl.xlim(xmin, xmax)
  pl.ylim(-2, 0)
  pl.axis('off')
  pl.savefig(".trajetorias/{:04d}.png".format(k+10))
  pl.clf()

### Plotando a trejetória
np = lp//FP
for i in range(1, np+1):
  print("Passo ", i, "/", np, " da trajetória.")
  sol_FPk(i)

### FIM  
