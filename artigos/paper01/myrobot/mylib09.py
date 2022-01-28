### Auxiliar do prog10
### Davi Neves - Jan. 2022

from mylib04 import igualistas, acoes_listas, varia_estados
import numpy as np
import pylab as pl

### Cria lista de ângulos
def liang(o1, o2, b1, b2):
  dteta = np.amax([abs(b1-o1)/50, abs(b2-o2)/50])
  da1 = acoes_listas(o1, b1, dteta)  # mylib01
  da2 = acoes_listas(o2, b2, dteta)
  da1, da2 = igualistas(da1, da2)    # mylib03
  la1, la2 = varia_estados(b1, b2, da1, da2)
  return la1, la2

### Sistema de EDOs do pêndulo duplo
def dpendulo(x, t):
  ### Parâmetros 
  l1, l2, m1, m2 = 1, 1, 1, 1
  g = 9.82
  ### Entrada de dados
  q1, q2, p1, p2 = x
  ### Sistema de EDOs acopladas - Vide Referência
  dq1_dt = (6/(m1*l1*l1))*((2*p1 - 3*np.cos(q1-q2)*p2) / (16 - 9*((np.cos(p1-p2))**2)))
  dq2_dt = (6/(m2*l2*l2))*((8*p2 - 3*np.cos(q1-q2)*p1) / (16 - 9*((np.cos(p1-p2))**2)))
  dp1_dt = (-0.5*m1*l1*l1)*(dq1_dt*dq2_dt*np.sin(q1-q2) + (3*(g/l1))*np.sin(q1))
  dp2_dt = (-0.5*m2*l2*l2)*(-dq1_dt*dq2_dt*np.sin(q1-q2) + (g/l2)*np.sin(q2))
  ### Retorno do sistema
  return dq1_dt, dq2_dt, dp1_dt, dp2_dt

### Busca por trajetorias no meio ao caos
def busca_listas(xi, yi, xf, yf):
  ### Lista de registro
  Lista = []
  ### Importando e ajustando os dados
  dados = np.genfromtxt("prog10/data/dados_caos.csv", delimiter=",")
  X1, X2 = np.round(dados[:,0], 2), np.round(dados[:,1], 2)
  ### Procurando xi e yi
  a, b = np.where(X1 == xi), np.where(X2 == yi)
  i1 = np.intersect1d(a, b)
  a, b = np.where(X1 == xf), np.where(X2 == yf)
  i2 = np.intersect1d(a, b)
  ### Garantindo a solução
  if (len(i1) < 1 or len(i2) < 1):
    A = np.array([])
    return A
  else:
    ik1, ik2 = i1[0], i2[0]
  ### Construindo a lista 
  if (ik2 > ik1):
    for i in range(ik1, ik2+1):
      Lista.append([X1[i], X2[i]])
  else:
    for i in range(ik2, ik1+1):
      Lista.append([X1[i], X2[i]])
  ### Pronto, agora basta retornar como array
  A = np.array(Lista)
  return A

### Plota trajetórias caóticas
def plot_traj_caos(A, caminho):
  ### Lista de ângulos
  L1 = A[:,0]
  L2 = A[:,1]
  N = len(L1)-1
  ### Plotando no máximo 500 pontos
  if (N > 501):
    L1N, L2N = [], []
    dN = int(N/500)
    for i in range(0, N, dN):
      L1N.append(L1[i])
      L2N.append(L2[i])
    L1 = np.array(L1N)
    L2 = np.array(L2N)
    N = len(L1)-1
  else:
    pass
  ### Coordenadas X-Y
  X = np.cos(L1) + np.cos(L2)
  Y = np.sin(L1) + np.sin(L2)
  ### Plotando a trajetória
  pl.figure(figsize=(5,4), dpi=300)
  pl.scatter(X[0], Y[0], color='red', marker='D', s=60, label="first")
  pl.scatter(X[N], Y[N], color='black', marker='D', s=60, label="end")
  pl.plot(X, Y, 'b-', alpha=0.45)
  pl.title("Chaotic Path")
  pl.xlabel("X")
  pl.ylabel("Y")
  pl.legend()
  pl.savefig(caminho)
  pl.clf()

### FIM
