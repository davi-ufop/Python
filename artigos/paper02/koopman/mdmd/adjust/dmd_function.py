### Program to build DMD method based in Koopman's operator
### Davi Neves - Abril/2022 - UFOP

### Importing libraries and methods
import numpy as np
from numpy.linalg import eig, svd, pinv

### Function to calc A matrix: Data and rank
def dmd_model(X1, X2, r):

  ### Decomposição pro valor singular -> SVD
  U, S, V = svd(X1)
  ### Truncate according to matrix rank (r)
  Ur = U[:, :r].conj().T
  Sr = np.reciprocal(S[:r])
  Vr = V[:r, :].conj().T

  ### Determinando o operador K e seus autos
  K = Ur@X2@Vr*Sr
  R, W = eig(K)  # K autovalores e autovetores

  ### Autovetores de Koopman e seus pseudoinversos
  Phi = X2@Vr@np.diag(Sr)@W
  IPhi = pinv(Phi)
  ### Matrizes dos autovalores de Koopman
  LB = np.diag(R)
  ### Matriz de transição de estados A
  A = Phi@LB@IPhi

  ### Retorno
  return A

### FIM
