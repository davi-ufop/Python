""" Dynamic mode decomposition on a linear system
    https://pykoopman.readthedocs.io/en/latest/examples/index.html """
### Davi Neves - DECOM/UFOP/Brasil - Jan. 2022
import matplotlib.pyplot as plt
import numpy as np
import warnings
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pykoopman as pk

warnings.filterwarnings('ignore')

tArray = np.linspace(0, 4*np.pi, 200)  # Time array for solution
dt = tArray[1]-tArray[0] # Time step
xArray = np.linspace(-10,10,400)
[Xgrid, Tgrid] = np.meshgrid(xArray, tArray)

def sech(x):
    return 1./np.cosh(x)

omega1 = 2.3
omega2 = 2.8
f1 = np.multiply(sech(Xgrid+3), np.exp(1j*omega1*Tgrid))
f2 = np.multiply( np.multiply(sech(Xgrid), np.tanh(Xgrid)), 2*np.exp(1j*omega2*Tgrid))
f = f1 + f2


def plot_dynamics(Xgrid, Tgrid, f, fig=None, title='', subplot=111):
    if fig is None:
        fig = plt.figure(figsize=(12, 4))

    time_ticks = np.array([0, 1*np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
    time_labels = ('0', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$')

    ax = fig.add_subplot(subplot, projection='3d')
    surf = ax.plot_surface(Xgrid, Tgrid, f, rstride=1)
    cset = ax.contourf(Xgrid, Tgrid, f, zdir='z', offset=-1.5, cmap=cm.ocean)
    ax.set(
        xlabel=r'$x$',
        ylabel=r'$t$',
        title=title,
        yticks=time_ticks,
        yticklabels=time_labels,
        xlim=(-10, 10),
        zlim=(-1.5, 1),
    )
    ax.autoscale(enable=True, axis='y', tight=True)

fig = plt.figure(figsize=(12,4))
fig.suptitle('Spatiotemporal dynamics of mixed signal')
plot_dynamics(Xgrid, Tgrid, f, fig=fig, title=r'$f(x, t) = f_1(x,t) + f_2(x,t)$', subplot=131)
plot_dynamics(Xgrid, Tgrid, f1, fig=fig, title=r'$f_1(x,t)$', subplot=132)
plot_dynamics(Xgrid, Tgrid, f2, fig=fig, title=r'$f_2(x,t)$', subplot=133)
plt.show()
plt.clf()


model = pk.Koopman()
model.fit(f)


# Let's look at the Koopman matrix
K = model.koopman_matrix.real
plt.matshow(K)
plt.show()
plt.clf()

# Let's have a look at the eigenvalues of the Koopman matrix
evals, evecs = np.linalg.eig(K)
evals_cont = np.log(evals)/dt
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111)
ax.plot(evals_cont.real, evals_cont.imag, 'bo', label='pykoopman')
print(omega1,omega2)
plt.clf()

f_predicted = np.vstack((f[0], model.simulate(f[0], n_steps=f.shape[0] - 1)))
fig = plt.figure(figsize=(8, 4))
fig.suptitle('PyKoopman simulation')
plot_dynamics(Xgrid, Tgrid, f, fig=fig, title=r'$f(x, t)$', subplot=121)
plot_dynamics(Xgrid, Tgrid, f_predicted, fig=fig, title='PyKoopman', subplot=122)
plt.clf()
plt.close()
