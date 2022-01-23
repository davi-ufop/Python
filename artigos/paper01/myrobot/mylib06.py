### Programa para auxiliar o prog07_koopman.py
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

###### Bibliotecas e módulos
import numpy as np
from scipy.integrate import odeint
from tqdm import tqdm

### Semente randômica
np.random.seed(888)
### Número de simulações
N = 50      
tmax = 20
dt = 0.01
t = np.arange(0, tmax, dt)

###### SISTEMA: PÊNDULO DUPLO
### Parâmetros 
l1, l2, m1, m2 = 1, 1, 1, 1
g = 9.82
### Sistema de EDOs do pêndulo duplo
def dpendulo(x, t):
  ### Entrada de dados
  q1, q2, p1, p2 = x
  ### Sistema de EDOs acopladas - Vide Referência
  dq1_dt = (6/(m1*l1*l1))*((2*p1 - 3*np.cos(q1-q2)*p2) / (16 - 9*((np.cos(p1-p2))**2)))
  dq2_dt = (6/(m2*l2*l2))*((8*p2 - 3*np.cos(q1-q2)*p1) / (16 - 9*((np.cos(p1-p2))**2)))
  dp1_dt = (-0.5*m1*l1*l1)*(dq1_dt*dq2_dt*np.sin(q1-q2) + (3*(g/l1))*np.sin(q1))
  dp2_dt = (-0.5*m2*l2*l2)*(-dq1_dt*dq2_dt*np.sin(q1-q2) + (g/l2)*np.sin(q2))
  ### Retorno do sistema
  return dq1_dt, dq2_dt, dp1_dt, dp2_dt

###### RESOLUÇÃO DO PÊNDULO DUPLO
### Resolução
print("\nPêndulo duplo:")
for i in tqdm(range(N)):
  ### Condições iniciais
  teta1 = np.random.uniform(-np.pi/3, np.pi/3)
  teta2 = np.random.uniform(-np.pi/2, np.pi/2)
  x0 = np.array([teta1, 0, teta2, 0]) 
  ### Integrando o sistema de EDOs
  x = odeint(dpendulo, x0, t)
  ### Salvando as informações
  DIR = "prog07/data/dpendulo/"
  np.savetxt(DIR+"entrada{:02d}.csv".format(i+1), x0, delimiter=",")
  np.savetxt(DIR+"saida{:02d}.csv".format(i+1), x, delimiter=",")


###### SISTEMA DE DUFFING
### Sistema de EDOS de Duffing (Pêndulo Simples)
def duffing(x, t):
  ## Entrada das variáveis
  x1, x2 = x
  ## Equações diferenciais acopladas -> Pêndulo Simples O(3)
  dx1_dt = x2
  dx2_dt = x1 - x1**3
  ## Retorno do sistema
  return dx1_dt, dx2_dt

###### RESOLUÇÃO DO SISTEMA DE DUFFING
### Resolução
print("\nPêndulo simples:")
for i in tqdm(range(N)):
  ### Condições iniciais
  teta1 = np.random.uniform(-np.pi/3, np.pi/3)
  omega1 = np.random.uniform(0, 0.5)
  x0 = np.array([teta1, omega1])
  ### Integrando o sistema de EDOs
  x = odeint(duffing, x0, t)
  ### Salvando as informações
  DIR = "prog07/data/duffing/"
  np.savetxt(DIR+"entrada{:02d}.csv".format(i+1), x0, delimiter=",")
  np.savetxt(DIR+"saida{:02d}.csv".format(i+1), x, delimiter=",")


######  SISTEMA DE LORENZ
### Sistema de Lorenz
def lorenz(x, t, S, B, R):
  ## Condição
  x1, x2, x3 = x
  ## Sistema: Taxa de convecção (1) + 
  ## Temperaturas horizontais (2) e verticais (3)
  dx1_dt = S*(x2 - x1)
  dx2_dt = x1*(R - x3) - x2
  dx3_dt = x1*x2 - B*x3
  ## Resultado
  return dx1_dt, dx2_dt, dx3_dt

###### RESOLUÇÃO DO SISTEMA DE LORENZ
### Parâmetros
sigma, beta, rho = 10, 8/3, 28
### Resolução
print("\nSistema atmosférico:")
for i in tqdm(range(50)):
  ### Condições iniciais
  x01 = np.random.uniform(0.8, 2.3)
  x02 = np.random.uniform(0.2, 1.1)
  x03 = np.random.uniform(1.3, 3.2)
  x0 = np.array([x01, x02, x03])
  ### Integrando o sistema de ODEs
  x = odeint(lorenz, x0, t, args=(sigma, beta, rho))
  ### Salvando as informações
  DIR = "prog07/data/lorenz/"
  np.savetxt(DIR+"entrada{:02d}.csv".format(i+1), x0, delimiter=",")
  np.savetxt(DIR+"saida{:02d}.csv".format(i+1), x, delimiter=",")

### FIM 
