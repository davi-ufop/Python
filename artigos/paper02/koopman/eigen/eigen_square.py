### Program illustrate eigen decomposition of a square matrix (A)
### Davi Neves - Ouro Preto, Brazil - Feb. 2022

### Modules
from numpy import array, diag, round
from numpy.linalg import inv, eig
from numpy.random import randn

### Make random matrix
n, p = 5, 2        # order and precision
A = round(randn(n, n), p)  # A matrix
AI = inv(A)  # A inverse

### Decomposition: A.x = lambda.x -> A.X = X.LB -> A = X.LB.XI
lb, X = eig(A)    # egenvalues, eigenvectors
LB = diag(lb)     # eigenvalues matrix
LI = diag(1/lb)   # pseudoinverse
XI = inv(X)       # eigenvectors inverse matrix

### Compositions
AC = X@LB@XI     # A matrix composte
AIC = X@LI@XI    # A inverse composte

### Results
print("A matrix:\n", round(A.real, p))
print("A composte:\n", round(AC.real, p))
print("A inverse:\n", round(AI.real, p))
print("A inverse composte:\n", round(AIC.real, p))

### End
