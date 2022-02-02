### Program illustrate eigen decomposition of a non-square matrix (A)
### Davi Neves - Ouro Preto, Brazil - Feb. 2022

### Modules
from numpy import array, zeros, dot, fill_diagonal, round, sqrt
from numpy.linalg import inv, eig
from numpy.random import randn

### Make random matrix
m, n, p = 2, 3, 2          # row, column and precision
#A = round(randn(m, n), p)  # A matrix
A = array([[3,3,2],[2,3,-2]])
S = zeros((m,n))           # Singular Matrix
print("A:\n", A)

### Transpose to Left and Right
GU = A@A.T  # Left -> Generate of U -> mxm
GV = A.T@A  # Right -> Generate of V -> nxn
print("GU:\n", GU)
print("GV:\n", GV)

### Eigenvalues and Eigenvectors
LU, U = eig(GU)
LV, V = eig(GV)

### Fill singular matrix with eigenvalues
if (len(LU)>len(LV)):
  L = LV
else:
  L = LU
fill_diagonal(S, sqrt(L))

### Verificate
print("U:\n", U)
print("VT:\n", V.T)
print("S:\n", S)

### A Composte
AC = U@S@V.T

### Results
print("A matrix:\n", round(A, p))
print("A composte:\n", round(AC.real, p))

### End
