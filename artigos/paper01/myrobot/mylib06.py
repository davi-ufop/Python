### Programa para auxiliar o prog07_koopman.py
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

###### Bibliotecas e módulos
import numpy as np
import pylab as pl
from scipy.integrate import odeint
import pykoopman as kp

### Condições iniciais
t = np.arange(0, 30, 0.01)
x0 = np.array([-0.8, 0, -0.5, 0])  
l1, l2, m1, m2 = 1, 1, 1, 1
g = 9.82

### Sistema de EDOs do pêndulo duplo: Vide Wikipédia
def sistema(x, t):
  ### Entrada de dados
  q1, q2, p1, p2 = x
  ### Sistema de EDOs acopladas - Vide Referência
  dq1_dt = (6/(m1*l1*l1))*((2*p1 - 3*np.cos(q1-q2)*p2) / (16 - 9*((np.cos(p1-p2))**2)))
  dq2_dt = (6/(m2*l2*l2))*((8*p2 - 3*np.cos(q1-q2)*p1) / (16 - 9*((np.cos(p1-p2))**2)))
  dp1_dt = (-0.5*m1*l1*l1)*(dq1_dt*dq2_dt*np.sin(q1-q2) + (3*(g/l1))*np.sin(q1))
  dp2_dt = (-0.5*m2*l2*l2)*(-dq1_dt*dq2_dt*np.sin(q1-q2) + (g/l2)*np.sin(q2))
  ### Retorno do sistema
  return dq1_dt, dq2_dt, dp1_dt, dp2_dt

### Integrando o sistema de EDOs
x = odeint(sistema, x0, t)

### Ajustando a solução
teta1, teta2, omega1, omega2 = x[:,0], x[:,1], x[:,2], x[:,3]
xk = l1*np.sin(teta1) + l2*np.sin(teta2)            ## Posições cartesianas da ponta do pêndulo
yk = l1*np.cos(teta1) + l2*np.cos(teta2)

pl.plot(xk, yk, 'b-')
pl.xlabel('x')
pl.ylabel('y')
pl.show()
pl.clf()

Field = np.array([xk, yk]).T

modelo = kp.Koopman()
modelo.fit(Field)

K = modelo.koopman_matrix.real
print(K)
#pl.matshow(K)
#pl.show()
#pl.clf()

Fprev = np.vstack((Field[0], modelo.simulate(Field[0], n_steps=Field.shape[0]-1)))

xp = Fprev[:,0].real
yp = Fprev[:,1].real

pl.plot(xp, yp, 'r-')
pl.xlabel('x')
pl.ylabel('y')
pl.show()
pl.clf()

dy = abs(yp[1:50] -yk[1:50])

pl.plot(dy, 'k.')
pl.show()
pl.clf()

### FIM 
