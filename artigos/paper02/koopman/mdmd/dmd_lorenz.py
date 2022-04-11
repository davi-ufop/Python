### Implementação do método dynamic mode decomposition
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Módulos e bibliotecas
import numpy as np
import pylab as pl
from mpl_toolkits import mplot3d
from numpy.linalg import svd, eig, pinv

### Importando os dados
t = np.arange(0, 50, 0.01)  # Vide ../dados/gendata.py
X = np.genfromtxt("../dados/lorenz.csv", delimiter=",")
Y1, Y2, Y3 = X[:,0], X[:,1], X[:,2]
### Conferindo com visual
pl.figure()
ax = pl.axes(projection='3d')
ax.scatter3D(t, Y1, Y3, c=t, cmap='Greens')
pl.show()
pl.clf()

### Parâmetros
r = 3
N = 10
p = 1

### Vetores pra testar A
X = X.T
print("X:\n", np.round(X[:,N-2:N+2],p))
x0 = X[:,0]
x1 = X[:,N]
x2 = X[:,N+1]
x3 = X[:,N+2]
x4 = X[:,N+3]

### Construindo as matrízes de entrada pro DMD
X1 = X[:, :N]
X2 = X[:, 1:N+1]


### Decomposição pro valor singular -> SVD
Ui, Si, Vi = svd(X1, full_matrices=False)
Ur = Ui[:, :r].conj().T
Sr = np.reciprocal(Si[:r])
Vr = Vi[:r, :].conj().T
 
### Cálculo do operador de Koopman, truncando USV com ordem r
K = Ur@X2@Vr*Sr
print("K:\n", np.round(K,p))

### Autovetores e autovalores de Koopman
Phi, Q = eig(K)

### Cálculo da matriz de evolução A
Psi = X2@Vr@np.diag(Sr)@Q
A = Psi@np.diag(Phi)@pinv(Psi)
### Retornando A
print("A:\n", np.round(A.real,p))

### Testando
print("x1:\n", np.round(x1,p))
x2_a = (A@x1).real
print("\nx2:\n", np.round(x2,p))
print("x2_A:\n", np.round(x2_a,p))
x3_a = (A@x2).real
print("\nx3:\n", np.round(x3,p))
print("x3_A:\n", np.round(x3_a,p))
x4_a = (A@x3).real
print("\nx4:\n", np.round(x4,p))
print("x4_A:\n", np.round(x4_a,p))

### Teste de verdade
xA = x0
XA = []
N5 = N+5
for i in range(N5):
  xA = A@xA
  XA.append(xA)
VA = np.array(XA)

### Plot A
A1, A3 = VA[:, 0], VA[:, 2]
pl.plot(Y1[:N5], Y3[:N5], 'k-', label='exact')
pl.plot(A1, A3, 'b.', label='forecast')
pl.xlabel('Convection')
pl.ylabel('Vertical Temperature')
pl.legend()
pl.show()
pl.clf()

### FIM
