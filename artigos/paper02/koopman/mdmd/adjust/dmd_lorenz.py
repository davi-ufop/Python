### Program to adjust Lorenz model in Koopman method
### Davi Neves - Abril/2022 - UFOP

### Importing libraries and methods
import numpy as np
import pylab as pl
from dmd_function import dmd_model 

### Importing and defining data
r = 3         # matriz rank
N = 5         # number of values - 1
N1 = N+1
P = int(2E3)  # total amount of points
### Import data of lorenz model
X = np.genfromtxt("../../dados/lorenz.csv", delimiter=",")
Y1, Y2, Y3 = X[:,0], X[:,1], X[:,2]           # adjusting data
XN = np.array([Y1[0:N1],Y2[0:N1],Y3[0:N1]])   # starting XN

for p in range(P):
  # points parameters
  pn = p + N             # end point of X1
  p1 = p + 1             # starting point of X2
  pn1 = pn + 1           # end point of X2 
  # X1 and X2 data updated
  #X1 = XN[:, p:pn]
  #X2 = XN[:, p1:pn1]
  X1 = np.array([Y1[p:pn],Y2[p:pn],Y3[p:pn]])
  X2 = np.array([Y1[p1:pn1],Y2[p1:pn1],Y3[p1:pn1]])
  ### Matrix A: X2 = A*X1
  A = dmd_model(X1, X2, r)
  ### End State Evaluated -> End XN-Data
  xn = XN[:, pn]
  ### Append xn in XN
  xn = A.real@xn
  XN = np.c_[XN, xn]

### Calculated results
Z1 = XN[0,:]
Z2 = XN[1,:]
Z3 = XN[2,:]

### Show results
fig, (ax1, ax2) = pl.subplots(1,2)   # Two in one
ax1.plot(Y1[0:P], Y3[0:P], 'b-')     # Left figure
ax1.set_title("Original")
ax2.plot(Z1, Z3, 'r-')               # Right figure
ax2.set_title("Calculated")
pl.savefig("solution.png")           # Saving

### FIM
