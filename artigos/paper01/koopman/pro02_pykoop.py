import matplotlib.pyplot as plt
import numpy as np
import warnings
import matplotlib.cm as cm
import pykoopman as pk
from pykoopman.common  import advance_linear_system

warnings.filterwarnings('ignore')

A = np.array([[1.5, 0],[0, 0.1]])
B = np.array([[1],[0]])

x0 = np.array([4,7])
u = np.array([-4, -2, -1, -0.5, 0, 0.5, 1, 3, 5])
n = len(u)+1
x,_ = advance_linear_system(x0,u,n,A,B)
X1 = x[:-1,:]
X2 = x[1:,:]
C = u[:,np.newaxis]
print('X1 = ', X1)
print('X2 = ', X2)
print('C = ', C)

U, s, Vh = np.linalg.svd(X1.T, full_matrices=False)
Aest = np.dot(X2.T,np.dot(Vh.T*(s**(-1)),U.T))
print('A = ', Aest)

DMDc = pk.regression.DMDc(svd_rank=3, control_matrix=B)

model = pk.Koopman(regressor=DMDc)
model.fit(x,C)
Aest = model.state_transition_matrix
Best = model.control_matrix

print(Aest)
np.allclose(A,Aest)


DMDc = pk.regression.DMDc(svd_rank=3)

model = pk.Koopman(regressor=DMDc)
model.fit(x,C)
Aest = model.state_transition_matrix
Best = model.control_matrix

print(Aest)
print(Best)
np.allclose(B,Best)
np.allclose(A,Aest)

xpred = model.simulate(x[0,:], C, n_steps=n-1)

fig = plt.figure(figsize=(6, 3))
ax = fig.add_subplot()
ax.plot(np.linspace(0,n-1,n),x[:,0],'-o', color='lightgrey', label='True')
ax.plot(np.linspace(0,n-1,n),x[:,1],'-o', color='lightgrey')
ax.plot(np.linspace(1,n-1,n-1),xpred[:,0],'--or', label='DMDc')
ax.plot(np.linspace(1,n-1,n-1),xpred[:,1],'--or')
ax.set(
        ylabel=r'$x$',
        xlabel=r'steps $k$')
ax.legend()
