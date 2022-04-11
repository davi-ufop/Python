### Implementação do método dynamic mode decomposition
### Davi Neves - Ouro Preto, Brasil - Jan. 2022

### Módulos e bibliotecas
import numpy as np
import pylab as pl
from numpy.linalg import svd, eig, pinv

### Definindo os dados
r = 3 # 2
N = 11
X = np.zeros((r, N))
print("X_empty:\n", X)

### Função para compor os dados
def G(xx, yy):
    #res = xx+yy
    #res = xx*yy - 2
    res = xx*xx - 2*yy
    return res

### Compondo os dados
X[0,:] = np.arange(1,N+1)
X[1,:] = np.arange(2,N+2)
X[2,:] = G(X[0,:], X[1,:])

print("X_full:\n", X)

### Construindo as matrízes de entrada pro DMD
X1 = X[:, :-1]
X2 = X[:, 1:]
print('X1: \n', X1)
print('X2: \n', X2)

### Decomposição pro valor singular -> SVD
U, S, V = svd(X1)

### Truncamento pro cálculo do operador de Koopman
Ur = U[:, :r].conj().T
Sr = np.reciprocal(S[:r])
Vr = V[:r, :].conj().T
 
### Operador de Koopman
K = Ur@X2@Vr*Sr
print("K:\n", np.round(K,1))

### Autovetores e autovalores de Koopman
Phi, Q = eig(K)

### Elaboração das matriz constituintes de A
DSr = np.diag(Sr)     # Diagonal dos autovalores de X1
DPhi = np.diag(Phi)   # Diagonal dos autovalores de Koopman
Psi = X2@Vr@DSr@Q     # Autovetores de A
Psinv = pinv(Psi)     # Pseudoinversa dos Autovetores de A

### Calculando A -> x_i+1 = A*x_i
A = Psi@DPhi@Psinv

### Mostrando A Real
print("A.real:\n", np.round(A.real,1))

### Testando -> Resultados:
print("\nÚltimo vetor:")
x15 = X[:,N-1]
print("x:\n", x15)
print("\nPrevisões:")
x16 = A@x15
print("x+1:\n", np.round(x16.real,1))
x17 = A@x16
print("x+2:\n", np.round(x17.real,1))
x18 = A@x17
print("x+3:\n", np.round(x18.real,1))

### Conferindo
print("\nConferindo:")
x16_3 = G(x16[0], x16[1])
x17_3 = G(x17[0], x17[1])
x18_3 = G(x18[0], x18[1])
print("x+1[3] = ", np.round(x16_3.real,1))
print("x+2[3] = ", np.round(x17_3.real,1))
print("x+3[3] = ", np.round(x18_3.real,1))

### Analisando a não linearidade
pl.plot(X[2,:], 'r.')
pl.title("X3")
pl.xlabel("pontos")
pl.ylabel("valores")
pl.show()

### FIM
