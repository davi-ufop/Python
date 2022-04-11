### Programa que replica a metodologia do dmd.m
### Davi Neves - Abril/2022 - UFOP

### Importando a biblioteca numérica
import numpy as np

### Dados
dt = 1  
r = 3
X1 = np.array([[1,2,3,4,5],[2,3,4,5,6],[3,5,7,9,11]])
print("X1:\n", X1)
X2 = np.array([[2,3,4,5,6],[3,4,5,6,7],[5,7,9,11,13]])

### Decomposição SVD e truncamento (r)
U, S, V = np.linalg.svd(X1)
Ur = U[:, :r]
Sr = np.diag(S)
Vr = V.T[:, :r]

### Matrizes úteis para cálculos posteriores
ISr = np.linalg.inv(Sr)
UrT = Ur.conj().T

### Determinando a matriz A_til
Atil = UrT@X2@Vr@ISr
D, Wr = np.linalg.eig(Atil)

### Autovetor de Koopman e seu pseudoinverso
Phi = X2@Vr@ISr@Wr
IPhi = np.linalg.pinv(Phi)

### Matrizes dos autovalores de Koopman
LB = np.diag(D)
om = np.log(D)*(1/dt) 

### Estado inicial dos dados acima
x1 = X1[:,0]
### Amplitude dos modos de Fourier
b = IPhi@x1

### Parâmetros para a integração temporal
m1 = X1.shape[1]
time = np.zeros((r, m1))
t = np.arange(0, m1)*dt

### Integração temporal
for it in range(m1):
  time[:,it] = np.exp(om*t[it])*b

### Espaço de estados reconstruído
Xd = Phi@time
print("X1_dmd:\n", np.round(Xd,0))

### Matriz de transição de estados A
A = Phi@LB@IPhi
### Determinando o último estado de X2
x2f = A.real@X1[:,m1-1]
print("x2f:\n", np.round(x2f,0))

### FIM
