### Programa para resolver o sistema referente ao pêndulo duplo
### Davi C. Neves - Ouro Preeto, MG, Brasil - UFOP/DECOM - Jan.,2022

### Importando módulos
import sys                              ## Linux
import numpy as np                      ## Numpy
from scipy.integrate import odeint      ## Integador de ODEs
import matplotlib.pyplot as pl          ## Plotagem Simples
from matplotlib.patches import Circle   ## Trajetórias

### Parâmetros do pêndulo: comprimentos (m), massas (kg).
L1, L2 = 1, 1
m1, m2 = 1, 1
### Aceleração gravitacional
g = 9.82
### Tempo máximo, passo temporal e mesh de integração (no tempo - s):
tmax, dt = 20, 0.01     ## tmax*10 é o número de figuras geradas
t = np.arange(0, tmax+dt, dt)
### Condições iniciais para o sistema dinâmico - ALTERE ISSO!
y0 = np.array([8.0*np.pi/7, 0, 8.0*np.pi/4, 0])

### Definindo o conjunto de equações: campo vetorial do sistema dinâmico
def deriv(y, t, L1, L2, m1, m2):
    theta1, z1, theta2, z2 = y
    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)
    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot

### Cálculo da energia do sistema:
def calc_E(y):
    th1, th1d, th2, th2d = y.T
    V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
    T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
            2*L1*L2*th1d*th2d*np.cos(th1-th2))
    return T + V

### Integrando o sistema de ODEs:
y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))

### Checando a conservação de energia:
EDRIFT = 0.05
E = calc_E(y0)
if np.max(np.sum(np.abs(calc_E(y) - E))) > EDRIFT:
    sys.exit('Energia máxima superou o limite tolerável: {}.'.format(EDRIFT))

### Enpacotando z e theta como funções temporais:
theta1, theta2 = y[:,0], y[:,2]

### Transformação de coordenadas para realizar a plotagem
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

# Plotando círculos
r = 0.05
trail_secs = 1
max_trail = int(trail_secs / dt)
def make_plot(i):
    ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k')
    c0 = Circle((0, 0), r/2, fc='k', zorder=10)
    c1 = Circle((x1[i], y1[i]), r, fc='b', ec='b', zorder=10)
    c2 = Circle((x2[i], y2[i]), r, fc='r', ec='r', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)
    ns = 20
    s = max_trail // ns
    for j in range(ns):
        imin = i - (ns-j)*s
        if imin < 0:
            continue
        imax = imin + s + 1
        alpha = (j/ns)**2
        ax.plot(x2[imin:imax], y2[imin:imax], c='r', solid_capstyle='butt',
                lw=2, alpha=alpha)
    ax.set_xlim(-L1-L2-r, L1+L2+r)
    ax.set_ylim(-L1-L2-r, L1+L2+r)
    ax.set_aspect('equal', adjustable='box')
    pl.axis('off')
    pl.savefig('frames/_img{:04d}.png'.format(i//di), dpi=72)
    pl.cla()

### Criando os frames em cada passo temporal
fps = 10
di = int(1/fps/dt)
fig = pl.figure(figsize=(8.3, 8.05), dpi=72)
ax = fig.add_subplot(111)

### Executando a plotagem
for i in range(0, t.size, di):
    print(i // di, '/', t.size // di)
    make_plot(i)

### FIM
