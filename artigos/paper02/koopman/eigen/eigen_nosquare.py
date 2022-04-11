### Program illustrate eigen decomposition of a non-square matrix (A)
### Davi Neves - Ouro Preto, Brazil - Feb. 2022

### Modules
from numpy import zeros, diag, round
from numpy.linalg import inv, eig, svd
from numpy.random import randint

### Make random matrix
l, m, n, p = 9, 3, 4, 1      # length, m-row, n-column and precision
A = randint(l, size=(m, n))  # A matrix: n > m or change line 29: 0 <-> 1
S = zeros((m,n))             # singular matrix
print("A:\n", A)

### Transpose to Left and Right
GU = A@A.T    # Left -> Generate of U -> mxm
GV = A.T@A    # Right -> Generate of V -> nxn
print("A.AT -> Left Transpose:\n", GU)
print("AT.A -> Right Transpose:\n", GV)

### Eigenvalues and Eigenvectors
LU, U = eig(GU)     # Left
LV, V = eig(GV)     # Right

### Using svd() function by numpy library
Uf, s, Vf = svd(A) 

### Fill singular matrix with eigenvalues
S[:A.shape[0], :A.shape[0]] = diag(s)
### S matrix use s values, that are square root of LU (or LV)

### Bad Extimated 
print("U:\n", round(U, p))
print("VT:\n", round(V.T, p))

### Determinated with numpy
print("U with numpy:\n", round(Uf, p))
print("VT with numpy:\n", round(Vf, p))

### A Composte
AC = U@S@V.T

### A Composte with numpy
ACf = Uf@S@Vf

### Results
print("A matrix:\n", round(A, p))
print("A composte:\n", round(AC.real, p))
print("A composte by numpy:\n", round(ACf.real, p))
### End
