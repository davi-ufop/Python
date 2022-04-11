### Program to adjust Lorenz model in Koopman method
### Davi Neves - Abril/2022 - UFOP

### Importando a biblioteca numérica
import numpy as np

### Dados
r = 3
P0 = 90
P1 = P0 + 1
N = P0 + 5
N1 = N+1
X = np.genfromtxt("../../dados/lorenz.csv", delimiter=",")
Y1, Y2, Y3 = X[:,0], X[:,1], X[:,2]
X1 = np.array([Y1[P0:N],Y2[P0:N],Y3[P0:N]])
X2 = np.array([Y1[P1:N1],Y2[P1:N1],Y3[P1:N1]])

### Decomposição pro valor singular -> SVD
U, S, V = np.linalg.svd(X1, full_matrices=False)
Ur = U[:, :r].conj().T
Sr = np.reciprocal(S[:r])
Vr = V[:r, :].conj().T

### Determinando o operador K e seus autos
K = Ur@X2@Vr*Sr
D, Wr = np.linalg.eig(K)

### Autovetor de Koopman e seu pseudoinverso
Phi = X2@Vr@np.diag(Sr)@Wr
IPhi = np.linalg.pinv(Phi)
### Matrizes dos autovalores de Koopman
LB = np.diag(D)
### Matriz de transição de estados A
A = Phi@LB@IPhi

### Estado inicial dos dados acima
x1 = X1[:,0]

### Determinando o último estado de X2
x2 = A.real@x1
print("x2:\n", x2)

### Indo além de 2
XN = []
xi = x1
for i in range(P0, N+1):
  XN.append(xi.real)
  xi = A@xi
XK = np.array(XN)
print("XN:\n", XK.T)  

### FIM
